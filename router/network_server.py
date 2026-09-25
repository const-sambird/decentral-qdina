import grpc
import time
import threading
import numpy as np
import torch
import csv
import os

from protos import qdina_pb2
from protos import qdina_pb2_grpc

from router.environment_global import GlobalRoutingEnv
from router.router_agent import RouterAgent
from common.replay_memory import ReplayMemory


class QDinaServerServicer(qdina_pb2_grpc.QDinaServiceServicer):
    def __init__(self, n_replicas, n_templates=22, batch_size=16, metrics_file=None,
                 param_layers=10, steps_per_episode=5, router_mode='learned'):
        '''
        gRPC Server Servicer coordinating decentralized worker nodes.
        '''
        self.n_templates = n_templates
        self.n_replicas = n_replicas
        self.steps_per_episode = steps_per_episode
        self.router_mode = router_mode

        self.env = GlobalRoutingEnv(n_templates=n_templates, n_replicas=n_replicas)

        if router_mode == 'learned':
            self.agent = RouterAgent(
                n_templates=n_templates,
                n_replicas=n_replicas,
                n_actions=self.env.n_actions,
            )
        elif router_mode == 'static' or router_mode == 'heuristic':
            # No learned agent in these modes. The routing table is either kept
            # fixed ('static') or overwritten by the DINA heuristic ('heuristic')
            # directly in SubmitMetricsAndGetWorkload.
            self.agent = None
        else:
            raise ValueError(f"Unknown router mode: {router_mode}")

        self.registered_workers = {}
        self.collected_metrics = {}

        self.current_workload_pool = []
        self.workload_templates_map = []

        self.routing_table_state = self.env.reset()[0]
        self.epsilon = 1.0
        self.batch_size = batch_size

        self.router_memory = ReplayMemory(capacity=50000)

        # Threading primitives for synchronization
        self.lock = threading.Condition()
        self.workers_waiting_count = 0
        self.global_step_counter = 0
        self.step_computed = False
        self.stop_training_signal = False
        self.next_workload_slices = {}

        self.global_epoch = 0

        self.last_known_costs = [0.0] * n_replicas
        self.last_known_indexes = {}
        self.collected_metrics = {}
        self.ready_to_train = False

        self.last_known_metrics = {}
        self.worker_workload_versions = {}
        self.step_computed = False

        # For tracking episode reset acknowledgments
        self.episode_reset_acks = set()

        # For storing the last valid costs per worker (used after local reset)
        self.last_valid_total_cost = {}
        self.last_valid_template_costs = {}

        self.routing_change_interval = 3
        self.steps_since_last_change = 0

        self.knapsack_metrics = {}

        # -----------------------------------------------------------------
        # DINA-faithful heuristic state
        # -----------------------------------------------------------------
        # `_last_known_cost_matrix` caches the most recent *non-zero* cost that
        # each (template, replica) pair was reported to have. This is essential
        # because a worker only reports costs for templates it actually received;
        # unreceived templates get a cost of 0, which would otherwise poison
        # `np.argmin` and cause the routing table to oscillate (see log analysis:
        # table flips between all-0 and all-1 because `argmin` picks the first
        # replica whose reported cost is 0).
        self._last_known_cost_matrix = np.zeros((n_templates, n_replicas), dtype=np.float64)

        # Templates that contain INSERT/UPDATE/DELETE statements. DINA routes
        # these to *every* replica (sentinel value -1 in `Router.evaluate()`),
        # because replicated data must stay consistent across the cluster.
        # Populated externally via `set_update_templates()`.
        self._update_templates = set()
        # -----------------------------------------------------------------

        self.metrics_file = None
        self.csv_writer = None
        self.total_actions = 0  # global counter
        if metrics_file:
            self.metrics_file = open(metrics_file, 'w', newline='')
            self.csv_writer = csv.writer(self.metrics_file)
            self.csv_writer.writerow([
                "episode", "step", "cumulative_actions",
                "makespan", "jain_index", "reward", "epsilon", "workers", "param_layers"
            ])

        self.param_layers = param_layers
        self.reset_complete = False

    # -------------------------------------------------------------------------
    # DINA-faithful helper : declare which templates are update templates
    # -------------------------------------------------------------------------
    def set_update_templates(self, templates):
        '''
        Register the set of template IDs that contain INSERT/UPDATE/DELETE
        statements. These templates are routed to every replica (broadcast),
        mirroring the `routes[template] = -1` sentinel used in DINA's
        `Router.evaluate()`.

        :param templates: iterable of integer template IDs
        '''
        self._update_templates = set(int(t) for t in templates)

    def initialize_routing_table(self):
        """
        Initialize the routing table by distributing templates evenly based
        on their frequency in the workload (greedy load balancing).
        """
        template_counts = np.zeros(self.n_templates)
        for t_id in self.workload_templates_map:
            template_counts[t_id] += 1

        sorted_templates = np.argsort(template_counts)[::-1]
        loads = np.zeros(self.n_replicas, dtype=np.int64)
        initial_routes = np.zeros(self.n_templates, dtype=np.int32)

        for t in sorted_templates:
            replica = np.argmin(loads)
            initial_routes[t] = replica
            loads[replica] += template_counts[t]

        self.routing_table_state = initial_routes
        self.env._state_routes = initial_routes
        print(f"[Router] Routing table initialized with loads: {loads}")

    def RegisterWorker(self, request, context):
        """Register a worker node with the master router.

        Args:
            request: gRPC request containing replica_id, hostname, port.
            context: gRPC context.

        Returns:
            RegistrationResponse: status and message.
        """
        with self.lock:
            worker_id = request.replica_id
            self.registered_workers[worker_id] = {
                'hostname': request.hostname,
                'port': request.port,
                'last_seen': time.time()
            }
            print(f"[gRPC Server] Worker Node {worker_id} successfully joined the cluster orchestrator.")
            return qdina_pb2.RegistrationResponse(status=True, message="Registered")

    def SubmitMetricsAndGetWorkload(self, request, context):
        """
        Called by each worker to send local metrics and receive the assigned queries
        for the next step. The router waits for all workers to submit before computing
        the next routing decision and slicing the workload.

        Returns:
            WorkloadSlice: a protobuf message containing the queries for this worker,
            or a stop signal to end the episode.
        """
        try:
            # If the router hasn't finished waiting for all workers to register,
            # send an empty slice and tell the worker not to stop yet.
            if not self.ready_to_train:
                return qdina_pb2.WorkloadSlice(stop_training=False, queries=[],
                                               epsilon=self.epsilon,
                                               param_layers=self.param_layers)

            worker_id = request.replica_id

            # Lock the condition variable to safely modify shared state.
            with self.lock:
                # Update the worker's last seen timestamp to prevent it from being
                # considered dead (timeout).
                if worker_id in self.registered_workers:
                    self.registered_workers[worker_id]['last_seen'] = time.time()

                if self.stop_training_signal:
                    # If reset is already complete, only return stop_training=True
                    if self.reset_complete:
                        return qdina_pb2.WorkloadSlice(stop_training=True, queries=[],
                                                       epsilon=self.epsilon,
                                                       param_layers=self.param_layers)

                    # Handling acknowledgments
                    if request.local_reset:
                        # retrieve metrics as before
                        self.episode_reset_acks.add(worker_id)
                        print(f"[Server] Worker {worker_id} acknowledged episode reset. "
                              f"({len(self.episode_reset_acks)}/{len(self.registered_workers)})")
                    else:
                        return qdina_pb2.WorkloadSlice(stop_training=True, queries=[],
                                                       epsilon=self.epsilon,
                                                       param_layers=self.param_layers)

                    # Check if all workers have acknowledged
                    if len(self.episode_reset_acks) >= len(self.registered_workers):
                        print("[Server] All workers acknowledged reset. Waiting for main loop to start next episode.")
                        self.reset_complete = True
                        self.episode_reset_acks.clear()
                        self.collected_metrics.clear()
                        # Reset the global step counter NOW to avoid any extra step
                        self.global_step_counter = 0
                        self.steps_since_last_change = 0

                        # NOTIFY the main loop that reset is complete
                        self.lock.notify_all()

                        # Wait for the main loop to clear the stop signal
                        while self.stop_training_signal:
                            self.lock.wait()

                        # Signal cleared – prepare slices for the new episode
                        self.reset_complete = False
                        self.next_workload_slices = {
                            w_id: self._get_routed_slice_for_node(w_id)
                            for w_id in self.registered_workers.keys()
                        }
                        print(f"[DEBUG] Next slices sizes: "
                              f"{ {w: len(self.next_workload_slices.get(w, [])) for w in self.registered_workers} }")
                        return qdina_pb2.WorkloadSlice(
                            stop_training=False,
                            queries=self.next_workload_slices.get(worker_id, []),
                            epsilon=self.epsilon,
                            param_layers=self.param_layers
                        )
                    else:
                        # Not all workers have acknowledged yet
                        return qdina_pb2.WorkloadSlice(stop_training=True, queries=[],
                                                       epsilon=self.epsilon,
                                                       param_layers=self.param_layers)

                # Store the metrics that this worker sent for the current step.
                if request.local_reset:
                    # A local reset (budget exceeded) occurred; reuse the last valid costs.
                    total_cost = self.last_valid_total_cost.get(worker_id, 0.0)
                    costs = self.last_valid_template_costs.get(worker_id, [0.0] * self.n_templates)
                    self.knapsack_metrics[worker_id] = {
                        'total_cost': total_cost,
                        'costs': costs,
                        'storage_used': request.storage_used,
                        'indexes': list(request.active_indexes),
                        'local_reset': True
                    }
                    self.last_known_metrics[worker_id] = self.knapsack_metrics[worker_id].copy()
                else:
                    total_cost = request.total_cost
                    costs = list(request.costs)
                    self.last_valid_total_cost[worker_id] = total_cost
                    self.last_valid_template_costs[worker_id] = costs

                self.collected_metrics[worker_id] = {
                    'step': self.global_step_counter,
                    'total_cost': total_cost,
                    'costs': costs,
                    'storage_used': request.storage_used,
                    'indexes': list(request.active_indexes),
                    'local_reset': request.local_reset
                }

                target_step = self.global_step_counter

                # Synchronization barrier: wait until the step advances or we receive
                # a stop signal. We cannot move forward until all workers have submitted.
                while self.global_step_counter == target_step:
                    # Remove workers that have not sent any request for more than 10 seconds.
                    now = time.time()
                    dead_workers = [wid for wid, info in self.registered_workers.items()
                                    if now - info['last_seen'] > 300.0]
                    for wid in dead_workers:
                        del self.registered_workers[wid]
                        self.collected_metrics.pop(wid, None)
                    if dead_workers:
                        self.lock.notify_all()
                        continue

                    # If all currently registered workers have submitted, proceed.
                    if len(self.collected_metrics) >= len(self.registered_workers):
                        # Ensure only one worker executes the computation (the leader).
                        if not self.step_computed:
                            self.step_computed = True
                            try:
                                # Leader computes the next routing decision.
                                # Gather total costs from all workers.
                                sorted_workers = sorted(self.collected_metrics.keys())
                                costs_array = np.array(
                                    [self.collected_metrics[w_id]['total_cost'] for w_id in sorted_workers],
                                    dtype=np.float64
                                )

                                # Get current state from the global environment.
                                state = self.env._get_obs()

                                # Decide whether to change routing or do nothing.
                                if self.router_mode == 'heuristic':
                                    # --------------------------------------------------
                                    # DINA-faithful heuristic routing.
                                    # Mirrors `Router.evaluate()` from the original
                                    # DINA codebase:
                                    #   1. argmin over per-template costs (with a
                                    #      last-known-cost fallback to avoid the
                                    #      0-cost "ghost" problem),
                                    #   2. fallback for templates with no cost at
                                    #      all -> least-loaded replica,
                                    #   3. update templates are broadcast to every
                                    #      replica (handled in _get_routed_slice_for_node).
                                    # --------------------------------------------------
                                    next_routes = self._dina_heuristic_routing(sorted_workers)
                                    self.routing_table_state = next_routes
                                    self.env._state_routes = next_routes.copy()
                                    action = 0

                                    # Recompose the per-template cost matrix the
                                    # environment expects, from the cached
                                    # non-zero matrix (keeps the observation
                                    # well-conditioned).
                                    template_costs_matrix = self._last_known_cost_matrix.copy()

                                elif self.router_mode == 'static':
                                    # Uniform static: keep the greedy-uniform table
                                    # initialised at the beginning of the episode.
                                    all_template_costs = [
                                        self.collected_metrics[w_id]['costs'] for w_id in sorted_workers
                                    ]
                                    template_costs_matrix = np.array(all_template_costs).T
                                    action = 0

                                else:
                                    # Learned router: apply the DQN every
                                    # routing_change_interval steps.
                                    all_template_costs = [
                                        self.collected_metrics[w_id]['costs'] for w_id in sorted_workers
                                    ]
                                    template_costs_matrix = np.array(all_template_costs).T

                                    self.steps_since_last_change += 1
                                    if self.steps_since_last_change >= self.routing_change_interval:
                                        action = self.agent.select_action(state, self.epsilon)
                                        self.steps_since_last_change = 0
                                    else:
                                        action = 0  # do nothing

                                # Compute current worker loads (number of queries per worker)
                                # based on the current routing table and the workload pool.
                                # Use explicit integer conversion and bounds checking.
                                worker_loads_current = np.zeros(self.n_replicas, dtype=np.int32)
                                for idx, q_text in enumerate(self.current_workload_pool):
                                    template_id = self.workload_templates_map[idx]
                                    try:
                                        template_id = int(template_id)
                                    except (TypeError, ValueError):
                                        template_id = -1
                                    if 0 <= template_id < self.n_templates:
                                        if isinstance(self.routing_table_state, np.ndarray) and \
                                                len(self.routing_table_state) > template_id:
                                            worker_id_assigned = int(self.routing_table_state[template_id])
                                            if 0 <= worker_id_assigned < self.n_replicas:
                                                worker_loads_current[worker_id_assigned] += 1
                                            else:
                                                print(f"[WARNING] Invalid worker id {worker_id_assigned} "
                                                      f"for template {template_id}")
                                        else:
                                            print(f"[WARNING] Routing table invalid for template {template_id}")
                                    else:
                                        print(f"[WARNING] Template id {template_id} out of range "
                                              f"(0-{self.n_templates - 1})")

                                # Apply the action and update the environment with costs and loads.
                                next_state, reward, _, _, info = self.env.step(
                                    action,
                                    external_costs=costs_array,
                                    external_template_costs=template_costs_matrix,
                                    worker_loads=worker_loads_current
                                )

                                if self.csv_writer is not None:
                                    self.total_actions += 1
                                    self.csv_writer.writerow([
                                        self.global_epoch,
                                        self.global_step_counter,
                                        self.total_actions,
                                        int(np.max(costs_array)),
                                        info.get('jain_index', 1.0),
                                        reward,
                                        self.epsilon,
                                        len(sorted_workers),
                                        self.param_layers
                                    ])
                                    self.metrics_file.flush()

                                # Update the routing table (which replica handles each template).
                                self.routing_table_state = np.copy(next_state[:self.env.n_templates])
                                # For each worker, compute the list of queries they will handle next.
                                self.next_workload_slices = {
                                    w_id: self._get_routed_slice_for_node(w_id)
                                    for w_id in self.registered_workers.keys()
                                }
                                # Save metrics for later export (benchmarking).
                                self.last_known_metrics = self.collected_metrics.copy()

                                # Log the current routing table and performance metrics.
                                table_str = " ".join(str(int(node)) for node in self.routing_table_state)
                                print(f"[Router State] Table : [{table_str}]")
                                print(f"[Router {self.router_mode}] Step {self.global_step_counter:2d} | "
                                      f"Makespan: {'{:,}'.format(int(np.max(costs_array))).replace(',', ' '):>18} | "
                                      f"Jain Index: {info.get('jain_index', 1.0):.4f} | "
                                      f"Reward: {reward:15.2f} | "
                                      f"Epsilon: {self.epsilon:.3f} | "
                                      f"Workers: {len(sorted_workers)}")

                                # Store the experience in the replay memory for training.
                                if self.agent is not None:
                                    self.router_memory.push(state, action, next_state, reward, None)

                                # If we have enough experiences, perform a learning step.
                                if self.agent is not None and len(self.router_memory) >= self.batch_size:
                                    self.agent.learn(self.router_memory, self.batch_size)
                                    self.agent.soft_update()

                                # Check if this was the last step (before increment)
                                if self.global_step_counter == self.steps_per_episode - 1:
                                    # Last step: signal stop, do NOT compute slices for next step
                                    self.stop_training_signal = True
                                    self.global_step_counter += 1
                                    self.next_workload_slices = {}  # empty, will not be used
                                else:
                                    # Normal step: compute slices for the next step
                                    self.routing_table_state = np.copy(next_state[:self.env.n_templates])
                                    self.next_workload_slices = {
                                        w_id: self._get_routed_slice_for_node(w_id)
                                        for w_id in self.registered_workers.keys()
                                    }
                                    self.global_step_counter += 1

                                # Clear metrics and release leader
                                self.collected_metrics.clear()
                                self.step_computed = False
                                self.lock.notify_all()
                                break

                            except Exception as e:
                                # If something goes wrong during computation, stop training.
                                print(f"[CRITICAL] Leader computation error: {e}")
                                self.step_computed = False
                                self.lock.notify_all()
                                self.stop_training_signal = True
                                return qdina_pb2.WorkloadSlice(stop_training=False, queries=[],
                                                               epsilon=self.epsilon,
                                                               param_layers=self.param_layers)
                        else:
                            # Another worker is already the leader; wait for it.
                            self.lock.wait()
                    else:
                        # Not all workers have submitted yet; wait for more.
                        self.lock.wait()

                # After the loop, if the stop signal is active, tell the worker to stop.
                if self.stop_training_signal:
                    return qdina_pb2.WorkloadSlice(stop_training=True, queries=[],
                                                   epsilon=self.epsilon,
                                                   param_layers=self.param_layers)

                # Otherwise, return the queries assigned to this specific worker.
                return qdina_pb2.WorkloadSlice(
                    stop_training=False,
                    queries=self.next_workload_slices.get(worker_id, []),
                    epsilon=self.epsilon,
                    param_layers=self.param_layers
                )

        except Exception as e:
            # Catch any unexpected error and force a stop to avoid hanging workers.
            print(f"[CRITICAL] Unhandled error in SubmitMetricsAndGetWorkload: {e}")
            return qdina_pb2.WorkloadSlice(stop_training=False, queries=[],
                                           epsilon=self.epsilon,
                                           param_layers=self.param_layers)

    # -------------------------------------------------------------------------
    # DINA-faithful heuristic routing
    # -------------------------------------------------------------------------
    def _dina_heuristic_routing(self, sorted_workers):
        '''
        Reproduces the routing decision logic from DINA's `Router.evaluate()`.

        The original code is:

            routes = np.argmin(self.times, axis=0)
            query_costs = [self.times[rep][i] for i, rep in enumerate(routes)]
            for t in update_templates:
                routes[t] = -1
            for r in range(num_replicas):
                for t in range(num_templates):
                    if routes[t] == r or routes[t] == -1:
                        replica_costs[r] += query_costs[t]
            min_replica = np.argmin(replica_costs)
            for i, c in enumerate(query_costs):
                if c == 0:
                    routes[i] = min_replica

        Adaptation for our distributed architecture:
          * Each worker only reports costs for the templates it actually received.
            Templates that were not received report 0, which would break `argmin`.
            We therefore substitute a "last known" cost when available, and +inf
            otherwise (so that unseen pairs are never selected by `argmin`).
          * Update templates are NOT encoded as -1 in `routing_table_state` (our
            slicing code uses the table directly). Instead, the broadcast is
            handled by `_get_routed_slice_for_node`, which sends update queries
            to every replica. The update templates are still counted on every
            replica when computing `replica_loads`, matching DINA's intent.

        :param sorted_workers: list of worker ids in a stable order; index
            r in the internal matrix corresponds to `sorted_workers[r]`.
        :returns: a (n_templates,) int32 array giving the chosen replica for
            each template.
        '''
        n_r = len(sorted_workers)
        matrix = np.full((self.n_templates, n_r), np.inf, dtype=np.float64)

        for r_idx, w_id in enumerate(sorted_workers):
            if w_id not in self.collected_metrics:
                continue
            costs = self.collected_metrics[w_id]['costs']
            for t in range(self.n_templates):
                c = float(costs[t]) if t < len(costs) else 0.0
                if c > 0.0:
                    matrix[t, r_idx] = c
                else:
                    # Fall back to last known non-zero cost, if any.
                    prev = self._last_known_cost_matrix[t, r_idx]
                    if prev > 0.0:
                        matrix[t, r_idx] = prev
                    # else: stays +inf, so argmin won't pick it.

        # Persist the best-known non-zero costs for the next step.
        finite_mask = np.isfinite(matrix)
        self._last_known_cost_matrix = np.where(finite_mask, matrix, self._last_known_cost_matrix)

        # Cold start: nothing to base a decision on yet -> keep the current table.
        if not np.any(finite_mask):
            return self.routing_table_state.copy()

        # 1) Greedy argmin per template (DINA).
        best_replicas = np.argmin(matrix, axis=1).astype(np.int32)

        # 2) Replica loads (DINA: updates are charged to *every* replica).
        replica_loads = np.zeros(n_r, dtype=np.float64)
        for t in range(self.n_templates):
            if t in self._update_templates:
                for rr in range(n_r):
                    if np.isfinite(matrix[t, rr]):
                        replica_loads[rr] += matrix[t, rr]
            else:
                r = best_replicas[t]
                if np.isfinite(matrix[t, r]):
                    replica_loads[r] += matrix[t, r]

        # 3) Fallback: templates that have no known cost anywhere go to the
        #    least-loaded replica (DINA's zero-cost fallback rule).
        min_replica = int(np.argmin(replica_loads))
        for t in range(self.n_templates):
            if not np.any(np.isfinite(matrix[t])):
                best_replicas[t] = min_replica

        return best_replicas

    def _get_routed_slice_for_node(self, node_id):
        """Compute the list of queries to be routed to a specific worker node.

        Depending on the execution mode ('uniform' or 'drift'), queries are either
        assigned round-robin or based on the routing table.

        Update templates (INSERT/UPDATE/DELETE) are broadcast to every replica,
        mirroring DINA's sentinel `routes[template] = -1`.

        Args:
            node_id (int): The replica ID of the worker.

        Returns:
            list[str]: List of SQL query strings assigned to this worker.
        """
        sorted_workers = sorted(self.registered_workers.keys())
        try:
            internal_id = sorted_workers.index(node_id)
        except ValueError:
            print(f"[WARNING] Node {node_id} not found in registered workers, using fallback.")
            internal_id = len(sorted_workers)  # fallback

        if hasattr(self, 'execution_mode') and self.execution_mode == 'uniform':
            sliced_queries = []
            for idx, q_text in enumerate(self.current_workload_pool):
                if idx % self.env.n_replicas == internal_id:
                    sliced_queries.append(q_text)
            return sliced_queries
        else:
            sliced_queries = []
            for idx, q_text in enumerate(self.current_workload_pool):
                template_id = self.workload_templates_map[idx]

                # DINA: update templates are routed to *every* replica.
                if template_id in self._update_templates:
                    sliced_queries.append(q_text)
                    continue

                if template_id < len(self.routing_table_state):
                    assigned_node = self.routing_table_state[template_id]
                    if assigned_node == internal_id:
                        sliced_queries.append(q_text)
            return sliced_queries

    def export_benchmark_files(self, output_dir="./output/"):
        """
        Export the routing table and index configuration to CSV files for the benchmark.
        Each index is written on a separate line with its columns separated by commas,
        exactly as expected by the benchmark (one composite index per line).
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        routes_path = os.path.join(output_dir, "routes.csv")
        config_path = os.path.join(output_dir, "config.csv")

        # Known column prefixes, sorted by descending length to avoid conflicts (e.g., 'ps_' before 'p_')
        COLUMN_PREFIXES = ['ps_', 'c_', 'l_', 'p_', 'o_', 'n_', 'r_', 's_']

        def split_columns(rest):
            """
            Decompose a compressed string like 'l_orderkey_l_shipdate' into a list
            of individual column names by recognizing known prefixes.
            Returns: ['l_orderkey', 'l_shipdate']
            """
            cols = []
            i = 0
            while i < len(rest):
                found = False
                for prefix in COLUMN_PREFIXES:
                    if rest.startswith(prefix, i):
                        start = i
                        i += len(prefix)
                        while i < len(rest):
                            if rest[i] == '_':
                                next_pos = i + 1
                                if any(rest.startswith(p, next_pos) for p in COLUMN_PREFIXES):
                                    break
                            i += 1
                        cols.append(rest[start:i])
                        found = True
                        if i < len(rest) and rest[i] == '_':
                            i += 1
                        break
                if not found:
                    cols.append(rest[i:])
                    break
            return cols

        # --- 1. Export the routing table ---
        with open(routes_path, "w", newline="") as f:
            f.write(",".join([str(int(r)) for r in self.routing_table_state]) + "\n")

        # --- 2. Export the index configuration ---
        with open(config_path, "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            for replica_id, worker_data in self.knapsack_metrics.items():
                indexes = worker_data.get('indexes', [])
                for composite in indexes:
                    parts = composite.split('_', 1)
                    if len(parts) != 2:
                        continue
                    table_full, rest = parts
                    cols = split_columns(rest)
                    if not cols:
                        continue
                    row = [replica_id - 1] + cols
                    writer.writerow(row)

        print(f"[Benchmark Export] Config exported successfully: {config_path}")

    def close_metrics(self):
        """
        Close the metrics CSV file if it was opened, ensuring all data is flushed to disk.
        """
        if self.metrics_file:
            self.metrics_file.close()
            self.metrics_file = None