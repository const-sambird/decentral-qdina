import grpc
import time
import threading
import numpy as np
import torch

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
        try:
            if not self.ready_to_train:
                return qdina_pb2.WorkloadSlice(stop_training=False, queries=[])
            worker_id = request.replica_id
            
            with self.lock:
                local_step = self.global_step_counter
                
                self.collected_metrics[worker_id] = {
                    'total_cost': request.total_cost,
                    'costs': list(request.costs),
                    'storage_used': request.storage_used
                }
                
                while self.global_step_counter == local_step and not self.stop_training_signal:
                    
                    if len(self.collected_metrics) == self.env.n_replicas:
                        
                        sorted_workers = sorted(self.collected_metrics.keys())
                        
                        costs_matrix = [self.collected_metrics[w_id]['total_cost'] for w_id in sorted_workers]
                        costs_array = np.array(costs_matrix, dtype=np.float64)
                        mkspan = float(np.max(costs_array))
                        
                        all_template_costs = [self.collected_metrics[w_id]['costs'] for w_id in sorted_workers]
                        template_costs_array = np.sum(all_template_costs, axis=0)
                        
                        state = self.env._get_obs()
                        action = self.agent.select_action(state, self.epsilon)
                        
                        next_state, reward, terminated, truncated, info = self.env.step(
                            action, 
                            external_costs=costs_array, 
                            external_template_costs=template_costs_array
                        )
                        
                        jain = info.get('jain_index', 1.0)
                        
                        self.router_memory.push(state, action, next_state, reward, False)
                        if len(self.router_memory) > self.batch_size:
                            self.agent.learn(self.router_memory, batch_size=self.batch_size)
                            
                        self.routing_table_state = np.copy(next_state[:self.env.n_templates])
                        table_str = " ".join(str(int(node)) for node in self.routing_table_state)
                        print(f"[Router State] Table : [{table_str}]")
                                                
                        print(f"[Router Learn] Step {local_step:2d} | Makespan: {mkspan:14.2f} | Jain Index: {jain:.4f} | Reward: {reward:15.2f} | Epsilon: {self.epsilon:.3f}")

                        self.next_workload_slices = {w_id: self._get_routed_slice_for_node(w_id) for w_id in self.registered_workers.keys()}
                        
                        with open("local_test_metrics.csv", "a") as f:
                            f.write(f"{self.global_epoch},{local_step},{mkspan},{jain},{reward}\n")
                        
                        self.epsilon = max(0.4, self.epsilon * 0.999)

                        self.collected_metrics.clear() 
                        self.global_step_counter += 1  
                        self.lock.notify_all()         
                        break
                    else:
                        self.lock.wait()
                
                sliced_queries = self.next_workload_slices.get(worker_id, [])
                return qdina_pb2.WorkloadSlice(
                    stop_training=self.stop_training_signal,
                    queries=sliced_queries
                )
                
        except Exception as server_err:
            print(f"[CRITICAL MASTER ERROR] Crash lors du traitement du Worker {request.replica_id} : {server_err}")
            import traceback
            traceback.print_exc()
            raise server_err

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
        
    def export_benchmark_files(self, output_dir="."):
        '''
        Export the final routing configuration and associated template-column mappings 
        for benchmark evaluation.
        '''
        import csv
        import os
        
        routes_path = os.path.join(output_dir, "routes.csv")
        config_path = os.path.join(output_dir, "config.csv")
        
        final_routes = [str(int(rep_id)) for rep_id in self.routing_table_state]
        
        with open(routes_path, "w", newline="") as f:
            f.write(",".join(final_routes) + "\n")
        print(f"[Benchmark Export] File exported successfully : {routes_path}")
        
        with open(config_path, "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            
            mock_columns_by_template = {
                0: ["l_shipdate", "l_discount", "l_quantity"],
                1: ["p_partkey", "ps_partkey", "s_suppkey"],
                2: ["o_orderdate", "o_orderkey"],
                3: ["l_orderkey", "l_shipdate"]
            }
            
            for template_idx, target_replica in enumerate(self.routing_table_state):
                cols = mock_columns_by_template.get(template_idx % 4, ["l_orderkey"])
                row = [int(target_replica)] + cols
                writer.writerow(row)
                    
        print(f"[Benchmark Export] File exported successfully : {config_path}")