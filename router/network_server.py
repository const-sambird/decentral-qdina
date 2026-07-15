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
    def __init__(self, n_replicas, n_templates=22, batch_size=16):
        '''
        gRPC Server Servicer coordinating decentralized worker nodes.
        '''
        self.env = GlobalRoutingEnv(n_templates=n_templates, n_replicas=n_replicas)
        self.agent = RouterAgent(n_templates=n_templates, n_replicas=n_replicas, n_actions=self.env.n_actions)

        self.registered_workers = {}
        self.collected_metrics = {}
        
        self.current_workload_pool = []  
        self.workload_templates_map = [] 
        
        self.routing_table_state = self.env.reset()[0]
        self.epsilon = 1.0
        self.batch_size = batch_size
        
        self.router_memory = ReplayMemory(capacity=2000)
        
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
        self.steps_per_episode = 20
        self.step_computed = False 

    def RegisterWorker(self, request, context):
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
        This method is called by each worker to send its local metrics and receive
        its assigned queries for the next step. The router waits for ALL workers to
        submit before computing the next routing decision and slicing the workload.

        Returns:
            WorkloadSlice: a protobuf message containing the queries for this worker, 
            or a stop signal to end the episode.
        """
        try:
            # If the router hasn't finished waiting for all workers to register,
            # send an empty slice and tell the worker not to stop yet.
            if not self.ready_to_train:
                return qdina_pb2.WorkloadSlice(stop_training=False, queries=[])

            worker_id = request.replica_id

            # Lock the condition variable to safely modify shared state.
            with self.lock:
                # Update the worker's last seen timestamp to prevent it from being
                # considered dead (timeout).
                if worker_id in self.registered_workers:
                    self.registered_workers[worker_id]['last_seen'] = time.time()

                # Store the metrics that this worker sent for the current step.
                self.collected_metrics[worker_id] = {
                    'step': self.global_step_counter,
                    'total_cost': request.total_cost,
                    'costs': list(request.costs),
                    'storage_used': request.storage_used,
                    'indexes': list(request.active_indexes)
                }

                # Remember which step we are currently waiting for.
                target_step = self.global_step_counter

                # Synchronization barrier: wait until the step advances or we receive
                # a stop signal. We cannot move forward until all workers have submitted.
                while self.global_step_counter == target_step:
                    # If the router has signaled the end of the episode, we still need
                    # to wait for all workers to finish the current step.
                    if self.stop_training_signal:
                        # If all workers have submitted, break to return the stop signal.
                        if len(self.collected_metrics) >= len(self.registered_workers):
                            break
                        else:
                            # Otherwise, wait for the remaining workers.
                            self.lock.wait()
                            continue

                    # Check for workers that have not sent any request for more than
                    # 10 seconds. Remove them to avoid infinite waiting.
                    now = time.time()
                    dead_workers = [wid for wid, info in self.registered_workers.items()
                                    if now - info['last_seen'] > 10.0]
                    for wid in dead_workers:
                        del self.registered_workers[wid]
                        self.collected_metrics.pop(wid, None)
                    if dead_workers:
                        # Wake up all threads to recalculate the number of active workers.
                        self.lock.notify_all()
                        continue

                    # If all currently registered workers have submitted, we can
                    # proceed with the step computation.
                    if len(self.collected_metrics) >= len(self.registered_workers):
                        # Ensure only one worker executes the computation (the leader).
                        if not self.step_computed:
                            self.step_computed = True
                            try:
                                # --- Leader computes the next routing decision ---
                                # Gather the total costs from all workers.
                                sorted_workers = sorted(self.collected_metrics.keys())
                                costs_array = np.array([self.collected_metrics[w_id]['total_cost'] for w_id in sorted_workers], dtype=np.float64)
                                # Sum template-level costs across workers to get global costs.
                                all_template_costs = [self.collected_metrics[w_id]['costs'] for w_id in sorted_workers]
                                template_costs_array = np.sum(all_template_costs, axis=0)

                                # Get the current state from the global environment.
                                state = self.env._get_obs()
                                # Choose an action (exploration vs exploitation) using the agent.
                                action = self.agent.select_action(state, self.epsilon)
                                # Apply the action and update the environment with the costs.
                                next_state, reward, _, _, info = self.env.step(
                                    action,
                                    external_costs=np.clip(costs_array, 0, 1e9),
                                    external_template_costs=template_costs_array
                                )

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
                                print(f"[Router Learn] Step {self.global_step_counter:2d} | Makespan: {float(np.max(costs_array)):14.2f} | Jain Index: {info.get('jain_index', 1.0):.4f} | Reward: {reward:15.2f} | Epsilon: {max(0.4, self.epsilon * 0.999):.3f} | Workers: {len(sorted_workers)}")

                                # Store the experience in the replay memory for training.
                                self.router_memory.push(state, action, next_state, reward, None)

                                # If we have enough experiences, perform a learning step to update the agent's policy.
                                if len(self.router_memory) >= self.batch_size:
                                    self.agent.learn(self.router_memory, self.batch_size)

                                # Advance to the next step.
                                self.global_step_counter += 1
                                # Clear the collected metrics for the next step.
                                self.collected_metrics.clear()
                                # Release the leader role.
                                self.step_computed = False
                                # Wake up all waiting workers so they can proceed.
                                self.lock.notify_all()
                                # Exit the while loop because the step has changed.
                                break

                            except Exception as e:
                                # If something goes wrong during computation, stop training.
                                print(f"[CRITICAL] Leader computation error: {e}")
                                self.step_computed = False
                                self.lock.notify_all()
                                self.stop_training_signal = True
                                return qdina_pb2.WorkloadSlice(stop_training=True, queries=[])
                        else:
                            # Another worker is already the leader; wait for it.
                            self.lock.wait()
                    else:
                        # Not all workers have submitted yet; wait for more.
                        self.lock.wait()

                # After exiting the loop, if the stop signal is active, tell the worker to stop.
                if self.stop_training_signal:
                    # Clear any pending workload slices to avoid sending queries.
                    self.next_workload_slices = {}
                    return qdina_pb2.WorkloadSlice(stop_training=True, queries=[])

                # Retrieve the queries assigned to this specific worker for the next step.
                sliced_queries = self.next_workload_slices.get(worker_id, [])
                return qdina_pb2.WorkloadSlice(stop_training=False, queries=sliced_queries)

        except Exception as e:
            # Catch any unexpected error and force a stop to avoid hanging workers.
            print(f"[CRITICAL] Unhandled error in SubmitMetricsAndGetWorkload: {e}")
            return qdina_pb2.WorkloadSlice(stop_training=True, queries=[])

    def _get_routed_slice_for_node(self, node_id):
        sorted_workers = sorted(self.registered_workers.keys())
        try:
            internal_id = sorted_workers.index(node_id)
        except ValueError:
            internal_id = node_id - 1
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
                    # Check if the current position matches a known prefix
                    if rest.startswith(prefix, i):
                        start = i
                        i += len(prefix)
                        # Advance until the next underscore followed by a prefix, or end of string
                        while i < len(rest):
                            if rest[i] == '_':
                                next_pos = i + 1
                                # Look ahead to see if a known prefix starts after the underscore
                                if any(rest.startswith(p, next_pos) for p in COLUMN_PREFIXES):
                                    break
                            i += 1
                        # Append the extracted column name
                        cols.append(rest[start:i])
                        found = True
                        # Skip the separating underscore
                        if i < len(rest) and rest[i] == '_':
                            i += 1
                        break
                if not found:
                    # Fallback: take the remaining substring as one column
                    cols.append(rest[i:])
                    break
            return cols

        # --- 1. Export the routing table ---
        # Write the routing table as a single CSV line mapping each template to a replica
        with open(routes_path, "w", newline="") as f:
            f.write(",".join([str(int(r)) for r in self.routing_table_state]) + "\n")

        # --- 2. Export the index configuration ---
        # Write one line per composite index, with the replica ID first, followed by its columns
        with open(config_path, "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")

            # Iterate over each replica's active indexes
            for replica_id, worker_data in self.last_known_metrics.items():
                indexes = worker_data.get('indexes', [])
                for composite in indexes:
                    # Expected format: "table_complete_col1_col2_..." (e.g., "lineitem_l_orderkey_l_shipdate")
                    parts = composite.split('_', 1)
                    if len(parts) != 2:
                        # Skip malformed entries
                        continue
                    table_full, rest = parts  # table_full is "lineitem", rest is "l_orderkey_l_shipdate"

                    # Decompose the concatenated columns into individual prefixed column names
                    cols = split_columns(rest)
                    if not cols:
                        continue

                    # Build the row: replica ID (0-based) followed by the columns
                    row = [replica_id - 1] + cols
                    writer.writerow(row)

        print(f"[Benchmark Export] Config exported successfully: {config_path}")