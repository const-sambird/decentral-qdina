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
                self.collected_metrics[worker_id] = {
                    'total_cost': request.total_cost,
                    'costs': list(request.costs),
                    'storage_used': request.storage_used,
                    'indexes': list(request.active_indexes)
                }
                
                local_step = self.global_step_counter
                
                while self.global_step_counter == local_step and not self.stop_training_signal:
                    if len(self.collected_metrics) < self.env.n_replicas:
                        self.lock.wait()
                    else:
                        sorted_workers = sorted(self.collected_metrics.keys())
                        costs_array = np.array([self.collected_metrics[w_id]['total_cost'] for w_id in sorted_workers], dtype=np.float64)
                        all_template_costs = [self.collected_metrics[w_id]['costs'] for w_id in sorted_workers]
                        template_costs_array = np.sum(all_template_costs, axis=0)
                        
                        state = self.env._get_obs()
                        action = self.agent.select_action(state, self.epsilon)
                        next_state, reward, _, _, info = self.env.step(
                            action, 
                            external_costs=np.clip(costs_array, 0, 1e9), 
                            external_template_costs=template_costs_array
                        )
                        
                        self.routing_table_state = np.copy(next_state[:self.env.n_templates])
                        self.next_workload_slices = {w_id: self._get_routed_slice_for_node(w_id) for w_id in self.registered_workers.keys()}

                        self.last_known_metrics = self.collected_metrics.copy()
                        
                        table_str = " ".join(str(int(node)) for node in self.routing_table_state)
                        print(f"[Router State] Table : [{table_str}]")
                        print(f"[Router Learn] Step {local_step:2d} | Makespan: {float(np.max(costs_array)):14.2f} | Jain Index: {info.get('jain_index', 1.0):.4f} | Reward: {reward:15.2f} | Epsilon: {max(0.4, self.epsilon * 0.999):.3f}")
                        
                        self.global_step_counter += 1
                        self.collected_metrics.clear()
                        self.lock.notify_all()

                sliced_queries = self.next_workload_slices.get(worker_id, [])
                return qdina_pb2.WorkloadSlice(stop_training=self.stop_training_signal, queries=sliced_queries)

        except Exception as server_err:
            print(f"[CRITICAL MASTER ERROR] Error {request.replica_id} : {server_err}")
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
        
    def export_benchmark_files(self, output_dir="./output/"):
        '''
        Export the final routing configuration and associated template-column mappings 
        for benchmark evaluation.
        '''
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        routes_path = os.path.join(output_dir, "routes.csv")
        config_path = os.path.join(output_dir, "config.csv")
        
        with open(routes_path, "w", newline="") as f:
            f.write(",".join([str(int(r)) for r in self.routing_table_state]) + "\n")

        with open(config_path, "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            
            data_map = {}
            
            for replica_id, worker_data in self.last_known_metrics.items():
                indexes = worker_data.get('indexes', [])
                
                for index_str in indexes:
                    parts = index_str.split('_')
                    if len(parts) >= 2:
                        table = parts[0]
                        col = "_".join(parts[1:])
                        
                        key = (replica_id, table)
                        if key not in data_map:
                            data_map[key] = set()
                        data_map[key].add(col)
            
            for (replica_id, table), cols in data_map.items():
                row = [replica_id-1] + sorted(list(cols))
                writer.writerow(row)
                    
        print(f"[Benchmark Export] Config exportée par table/réplicat : {config_path}")