# File: router/network_server.py
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
                 param_layers=10, steps_per_episode=5, router_mode='learned',
                 heuristic_theta=0.6):
        '''
        gRPC Server Servicer coordinating decentralized worker nodes.
        '''
        self.n_templates = n_templates
        self.n_replicas = n_replicas
        self.steps_per_episode = steps_per_episode
        self.router_mode = router_mode
        # Threshold for DiversityClusterDB's benefit-first load-aware routing.
        self.heuristic_theta = heuristic_theta

        self.env = GlobalRoutingEnv(n_templates=n_templates, n_replicas=n_replicas)

        if router_mode == 'learned':
            self.agent = RouterAgent(
                n_templates=n_templates,
                n_replicas=n_replicas,
                n_actions=self.env.n_actions,
            )
        elif router_mode == 'static' or router_mode == 'heuristic':
            # No learned agent in these modes. The routing table is either kept
            # fixed ('static') or overwritten by the heuristic ('heuristic')
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

        # Cache of last known non-zero cost per (template, replica).
        # Workers report 0 for templates they didn't receive, which would
        # otherwise poison `argmin` and make the routing table oscillate.
        self._last_known_cost_matrix = np.zeros((n_templates, n_replicas), dtype=np.float64)

        # Templates containing INSERT/UPDATE/DELETE; routed to every replica
        # (DINA's sentinel `routes[template] = -1` behaviour).
        self._update_templates = set()

        # Heuristic-mode state (DiversityClusterDB per-query routing):
        self._heuristic_query_routes = None
        self._heuristic_routes_pool = None
        self._full_template_counts = None
        self._no_index_costs = None

        self.metrics_file = None
        self.csv_writer = None
        self.total_actions = 0
        if metrics_file:
            self.metrics_file = open(metrics_file, 'w', newline='')
            self.csv_writer = csv.writer(self.metrics_file)
            self.csv_writer.writerow([
                "episode", "step", "cumulative_actions",
                "makespan", "jain_index", "reward", "epsilon", "workers", "param_layers"
            ])

        self.param_layers = param_layers
        self.reset_complete = False

    def set_update_templates(self, templates):
        '''
        Register the set of template IDs that contain INSERT/UPDATE/DELETE
        statements. These templates are routed to every replica (broadcast),
        mirroring DINA's `routes[template] = -1` sentinel.
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
        if self.router_mode == 'heuristic':
            if self._full_template_counts is None and np.sum(template_counts) > 0:
                self._full_template_counts = template_counts.astype(np.float64).copy()
            self._heuristic_query_routes = None
            self._heuristic_routes_pool = None
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
            msg = "Registered|router_mode=heuristic" if self.router_mode == 'heuristic' else "Registered"
            return qdina_pb2.RegistrationResponse(status=True, message=msg)

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
            if not self.ready_to_train:
                return qdina_pb2.WorkloadSlice(stop_training=False, queries=[],
                                               epsilon=self.epsilon,
                                               param_layers=self.param_layers)

            worker_id = request.replica_id

            with self.lock:
                if worker_id in self.registered_workers:
                    self.registered_workers[worker_id]['last_seen'] = time.time()

                if self.stop_training_signal:
                    if self.reset_complete:
                        return qdina_pb2.WorkloadSlice(stop_training=True, queries=[],
                                                       epsilon=self.epsilon,
                                                       param_layers=self.param_layers)

                    if request.local_reset:
                        self.episode_reset_acks.add(worker_id)
                        print(f"[Server] Worker {worker_id} acknowledged episode reset. "
                              f"({len(self.episode_reset_acks)}/{len(self.registered_workers)})")
                    else:
                        return qdina_pb2.WorkloadSlice(stop_training=True, queries=[],
                                                       epsilon=self.epsilon,
                                                       param_layers=self.param_layers)

                    if len(self.episode_reset_acks) >= len(self.registered_workers):
                        print("[Server] All workers acknowledged reset. "
                              "Waiting for main loop to start next episode.")
                        self.reset_complete = True
                        self.episode_reset_acks.clear()
                        self.collected_metrics.clear()
                        self.global_step_counter = 0
                        self.steps_since_last_change = 0
                        if self.router_mode == 'heuristic':
                            self._last_known_cost_matrix.fill(0.0)
                            self._heuristic_query_routes = None
                            self._heuristic_routes_pool = None

                        self.lock.notify_all()

                        while self.stop_training_signal:
                            self.lock.wait()

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
                        return qdina_pb2.WorkloadSlice(stop_training=True, queries=[],
                                                       epsilon=self.epsilon,
                                                       param_layers=self.param_layers)

                if request.local_reset:
                    if self.router_mode == 'heuristic' and any(float(c) > 0.0 for c in request.costs):
                        total_cost = request.total_cost
                        costs = list(request.costs)
                        self.last_valid_total_cost[worker_id] = total_cost
                        self.last_valid_template_costs[worker_id] = costs
                    else:
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

                while self.global_step_counter == target_step:
                    now = time.time()
                    dead_workers = [wid for wid, info in self.registered_workers.items()
                                    if now - info['last_seen'] > 300.0]
                    for wid in dead_workers:
                        del self.registered_workers[wid]
                        self.collected_metrics.pop(wid, None)
                    if dead_workers:
                        self.lock.notify_all()
                        continue

                    if len(self.collected_metrics) >= len(self.registered_workers):
                        if not self.step_computed:
                            self.step_computed = True
                            try:
                                sorted_workers = sorted(self.collected_metrics.keys())
                                state = self.env._get_obs()

                                if self.router_mode == 'heuristic':
                                    # DiversityClusterDB routing executed at EVERY step:
                                    #   1) Costs   : every agent reported full-workload costs under its
                                    #                current indexes (collected_metrics).
                                    #   2) Routing : Benefit-First Load-Aware routing (Hang et al. 2024,
                                    #                Algorithm 3) applied per query on the current pool.
                                    #   3) Exec    : execution cost per replica = sum of routed query costs.
                                    (template_table, query_routes,
                                     heuristic_exec_costs, heuristic_query_counts) = self._diversity_routing(
                                        sorted_workers, threshold=self.heuristic_theta
                                    )
                                    self._heuristic_query_routes = query_routes
                                    self._heuristic_routes_pool = self.current_workload_pool
                                    self.routing_table_state = template_table
                                    self.env._state_routes = template_table.copy()
                                    action = 0
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

                                # Reported worker costs are full-workload and independent of
                                # routing; derive routed cost per replica from the routing
                                # table and the cost matrix (DINA-style).
                                if self.router_mode == 'heuristic':
                                    routed_costs = np.asarray(heuristic_exec_costs, dtype=np.float64)
                                else:
                                    cost_source = template_costs_matrix

                                    routed_costs = np.zeros(self.n_replicas, dtype=np.float64)
                                    for t in range(self.n_templates):
                                        if t in self._update_templates:
                                            for r in range(self.n_replicas):
                                                c = float(cost_source[t, r]) if r < cost_source.shape[1] else 0.0
                                                if c > 0:
                                                    routed_costs[r] += c
                                        else:
                                            r = int(self.routing_table_state[t])
                                            if 0 <= r < self.n_replicas and r < cost_source.shape[1]:
                                                c = float(cost_source[t, r])
                                                if c > 0:
                                                    routed_costs[r] += c

                                costs_array = routed_costs

                                # Compute current worker loads (number of queries per worker)
                                # based on the current routing table and the workload pool.
                                if self.router_mode == 'heuristic':
                                    worker_loads_current = np.asarray(heuristic_query_counts, dtype=np.int32)
                                else:
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
                                self.last_known_metrics = self.collected_metrics.copy()

                                # Log the current routing table and performance metrics.
                                table_str = " ".join(str(int(node)) for node in self.routing_table_state)
                                print(f"[Router State] Table : [{table_str}]")
                                if self.router_mode == 'heuristic':
                                    counts_str = " ".join(str(int(c)) for c in heuristic_query_counts)
                                    exec_str = " ".join(f"{float(c):.3g}" for c in heuristic_exec_costs)
                                    print(f"[Router heuristic] Queries/replica : [{counts_str}] | "
                                          f"Exec cost/replica : [{exec_str}]")
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
                                    self.next_workload_slices = {}
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

    def _diversity_routing(self, sorted_workers, threshold=0.6):
        '''
        DiversityClusterDB Benefit-First Load-Aware Routing (Hang et al. 2024, Algorithm 3),
        executed at EVERY step in 3 explicit phases:
          1) Cost collection : build the (n_templates x n_replicas) full-workload cost matrix
                               from the costs just reported by all agents under their current indexes.
          2) Query routing   : route every query q in the current workload pool using Algorithm 3
                               (rank replicas by cost, transfer to beneficial sub-optimal replica if
                               l_j / l_i1 < theta) + makespan-aware safeguard (Eq. 4: never overload
                               a replica past another replica that can finish q sooner).
          3) Execution cost  : compute exact per-replica execution cost = sum of routed query costs
                               (Eq. 3), giving Makespan (Eq. 4) and Jain Index close to 1.0.

        :param sorted_workers: list of worker ids in a stable order.
        :param threshold: the paper's theta parameter (0 < threshold <= 1).
        :returns: (template_table, query_routes, exec_costs, query_counts)
        '''
        n_r = len(sorted_workers)
        theta = float(threshold) if threshold is not None else 0.6
        theta = min(max(theta, 1e-6), 1.0)
        benefit_tol = 1e-3

        # --- Phase 1: Build full-workload cost matrix C[t, r] from current worker reports ---
        prev_known = self._last_known_cost_matrix
        if prev_known is None or prev_known.shape != (self.n_templates, n_r):
            prev_known = np.zeros((self.n_templates, n_r), dtype=np.float64)

        matrix = np.full((self.n_templates, n_r), np.inf, dtype=np.float64)
        for r_idx, w_id in enumerate(sorted_workers):
            metrics = self.collected_metrics.get(w_id)
            if metrics is None:
                continue
            costs = metrics.get('costs', [])
            valid_full = True
            row_vals = np.zeros(self.n_templates, dtype=np.float64)
            for t in range(self.n_templates):
                c = float(costs[t]) if t < len(costs) else 0.0
                if c > 0.0:
                    matrix[t, r_idx] = c
                    row_vals[t] = c
                else:
                    valid_full = False
                    if prev_known[t, r_idx] > 0.0:
                        matrix[t, r_idx] = prev_known[t, r_idx]

            # Record true no-index baseline cost(q, empty) whenever a replica has 0 active indexes
            if valid_full and len(metrics.get('indexes', [])) == 0:
                if self._no_index_costs is None:
                    self._no_index_costs = row_vals.copy()

        finite_mask = np.isfinite(matrix)
        self._last_known_cost_matrix = np.where(finite_mask, matrix, prev_known)

        # Full-workload template counts (used to convert full-template cost into per-query cost)
        if self._full_template_counts is None:
            counts = np.zeros(self.n_templates, dtype=np.float64)
            for t in self.workload_templates_map:
                try:
                    t_int = int(t)
                except (TypeError, ValueError):
                    continue
                if 0 <= t_int < self.n_templates:
                    counts[t_int] += 1.0
            self._full_template_counts = counts
        n_full = np.maximum(self._full_template_counts, 1.0)

        q_cost = matrix / n_full[:, None]

        # Fill any missing entries with the template's max known cost (no-index proxy)
        # so no replica is ever stuck at +inf and starved of queries.
        known = q_cost[np.isfinite(q_cost)]
        default_cost = float(np.mean(known)) if known.size > 0 else 1.0
        for t in range(self.n_templates):
            row = q_cost[t]
            fin = np.isfinite(row)
            fill = float(np.max(row[fin])) if np.any(fin) else default_cost
            row[~fin] = fill

        # Baseline cost(q, empty) per query
        if self._no_index_costs is not None and len(self._no_index_costs) == self.n_templates:
            baseline = np.maximum(self._no_index_costs / n_full, np.max(q_cost, axis=1))
        else:
            baseline = np.max(q_cost, axis=1)

        # --- Phase 2: Benefit-First Load-Aware Routing (Algorithm 3), per query ---
        loads = np.zeros(n_r, dtype=np.float64)
        query_counts = np.zeros(n_r, dtype=np.int64)
        n_queries = len(self.current_workload_pool)
        query_routes = [0] * n_queries
        template_hits = np.zeros((self.n_templates, n_r), dtype=np.int64)

        pool_templates = []
        for idx in range(n_queries):
            try:
                t = int(self.workload_templates_map[idx])
            except (TypeError, ValueError, IndexError):
                t = -1
            pool_templates.append(t)

        # Broadcast update templates (INSERT/UPDATE/DELETE) to every replica
        analytical_indices = []
        for idx, t in enumerate(pool_templates):
            if 0 <= t < self.n_templates and t in self._update_templates:
                loads += q_cost[t]
                query_counts += 1
                query_routes[idx] = -1
            else:
                analytical_indices.append(idx)

        # Process heaviest queries first (LPT order) so lighter queries smooth out
        # residual load differences across replicas.
        analytical_indices.sort(
            key=lambda idx: (
                -(baseline[pool_templates[idx]] if 0 <= pool_templates[idx] < self.n_templates else default_cost),
                idx
            )
        )

        for idx in analytical_indices:
            t = pool_templates[idx]
            cost = q_cost[t] if 0 <= t < self.n_templates else np.full(n_r, default_cost)

            # Algorithm 3 Line 1-2: rank by cost; break cost ties using lowest current load.
            min_c = float(np.min(cost))
            cost_tol = 1e-4 * max(1.0, min_c)
            best_group = [r for r in range(n_r) if cost[r] <= min_c + cost_tol]
            i1 = min(best_group, key=lambda r: (loads[r], r))
            dest = i1

            # Algorithm 3 Lines 3-9: transfer to a beneficial sub-optimal replica
            # whose current load is < theta * loads[i1].
            if loads[i1] > 0.0 and 0 <= t < self.n_templates:
                others = sorted(
                    (r for r in range(n_r) if r != i1),
                    key=lambda r: (cost[r], loads[r], r)
                )
                for j in others:
                    if cost[j] < baseline[t] * (1.0 - benefit_tol) and (loads[j] / loads[i1]) < theta:
                        dest = j
                        break

            # Makespan-aware load safeguard (Eq. 4 of Hang et al. 2024):
            # If another replica can finish query q earlier than dest (l_r + cost(q, r) < l_dest + cost(q, dest)),
            # route q there so replicas without indexes yet are never starved and Jain Index stays ~1.0.
            best_mct = min(range(n_r), key=lambda r: (loads[r] + cost[r], cost[r], r))
            if loads[best_mct] + cost[best_mct] < loads[dest] + cost[dest]:
                dest = best_mct

            query_routes[idx] = dest
            loads[dest] += cost[dest]
            query_counts[dest] += 1
            if 0 <= t < self.n_templates:
                template_hits[t, dest] += 1

        # Build a balanced 22-entry summary table (for display & routes.csv export)
        template_table = np.array(self.routing_table_state, dtype=np.int32).copy()
        if template_table.shape != (self.n_templates,):
            template_table = np.zeros(self.n_templates, dtype=np.int32)
        assigned_per_replica = np.zeros(n_r, dtype=np.int64)
        for t in range(self.n_templates):
            if template_hits[t].sum() > 0:
                max_hits = int(np.max(template_hits[t]))
                top_r = [r for r in range(n_r) if template_hits[t, r] >= max(1, max_hits - 1)]
                chosen_r = min(top_r, key=lambda r: (assigned_per_replica[r], -template_hits[t, r], r))
                template_table[t] = chosen_r
                assigned_per_replica[chosen_r] += 1

        # --- Phase 3: Per-replica execution cost for this routing (Eq. 3) ---
        exec_costs = np.zeros(self.n_replicas, dtype=np.float64)
        counts_out = np.zeros(self.n_replicas, dtype=np.int32)
        m = min(n_r, self.n_replicas)
        if np.any(finite_mask):
            exec_costs[:m] = loads[:m]
        counts_out[:m] = query_counts[:m]

        return template_table, query_routes, exec_costs, counts_out

    def _get_routed_slice_for_node(self, node_id):
        """Compute the list of queries to be routed to a specific worker node.

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

        if (self.router_mode == 'heuristic'
                and self._heuristic_query_routes is not None
                and self._heuristic_routes_pool is self.current_workload_pool
                and len(self._heuristic_query_routes) == len(self.current_workload_pool)):
            sliced_queries = []
            for idx, q_text in enumerate(self.current_workload_pool):
                assigned = self._heuristic_query_routes[idx]
                if assigned == -1 or assigned == internal_id:
                    sliced_queries.append(q_text)
            return sliced_queries

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