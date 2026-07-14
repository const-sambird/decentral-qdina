# File: agent/network_client.py
import grpc
import time
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from protos import qdina_pb2 as qdina_pb2
from protos import qdina_pb2_grpc as qdina_pb2_grpc

from agent.environment_local import LocalIndexingEnv
from router.DQN import DQN
from agent.qnn import QuantumDQN
from common.spsa_opt import SPSAOptimiser
from common.replay_memory import ReplayMemory
from agent.database import Replica
from common.preprocessor import Preprocessor 
from common.profiling import Profiler

class QDinaNetworkClient:
    def __init__(self, replica_id: int, server_address: str, agent_mode: str, 
                 db_host: str, db_port: int, db_user: str, db_password: str, 
                 db_name: str, storage_budget: float = 10.0):
        '''
        Decentralized gRPC Client worker orchestrating local reinforcement learning indexing.
        '''
        self.replica_id = replica_id
        self.agent_mode = agent_mode.lower()
        self.storage_budget = storage_budget
        self.db_host = db_host
        self.db_port = db_port
        self.db_user = db_user
        self.db_password = db_password
        self.db_name = db_name
        
        print(f"[Worker Client {self.replica_id}] Linking gRPC channel to {server_address}...")
        self.channel = grpc.insecure_channel(server_address)
        self.stub = qdina_pb2_grpc.QDinaServiceStub(self.channel)
        
        self.n_templates = 22  
        self.candidates = []   
        self.templates_map = []
        
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.loss_fn = nn.MSELoss()
        
        self.local_memory = ReplayMemory(capacity=1000)
        self.batch_size = 16
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        
        self.env = None

    def register_to_master(self, local_hostname: str = '127.0.0.1', local_port: int = 5432):
        try:
            request = qdina_pb2.WorkerRegistration(
                replica_id=self.replica_id,
                hostname=local_hostname,
                port=local_port
            )
            response = self.stub.RegisterWorker(request)
            print(f"[Worker Client {self.replica_id}] Registration Status: {response.status} | Message: {response.message}")
            return response.status
        except grpc.RpcError as e:
            print(f"[Worker Client {self.replica_id}] Critical failure during registration step: {e.details()}")
            return False

    def _init_agent_networks(self):
        n_actions = self.env.action_space.n
        if self.agent_mode == 'classical':
            self.policy_net = DQN(n_observations=self.n_templates, n_actions=n_actions, layer_features=[64, 64])
            self.target_net = DQN(n_observations=self.n_templates, n_actions=n_actions, layer_features=[64, 64])
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
            print(f"[Worker Client {self.replica_id}] Classical DQN Policy Network built successfully.")
        elif self.agent_mode == 'quantum':
            self.policy_net = QuantumDQN(n_inputs=self.n_templates, n_qubits=5, n_actions=n_actions, qnn_type='twolocal', qnn_output='layer')
            self.optimizer = SPSAOptimiser(
                self.policy_net, 
                lr=0.1, 
                device=next(self.policy_net.parameters()).device
            )
            print(f"[Worker Client {self.replica_id}] Quantum DQN Parameterized Circuit compiled successfully.")

    def _select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.env.action_space.n - 1)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def _optimize_local_model(self):
        if len(self.local_memory) < self.batch_size:
            return
            
        transitions = self.local_memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        fixed_states = []
        for s in states:
            s_np = np.asarray(s, dtype=np.float32).flatten()
            if len(s_np) < self.n_templates:
                s_np = np.pad(s_np, (0, self.n_templates - len(s_np)), 'constant')
            else:
                s_np = s_np[:self.n_templates]
            fixed_states.append(torch.tensor(s_np, dtype=torch.float32))

        fixed_next_states = []
        for ns in next_states:
            ns_np = np.asarray(ns, dtype=np.float32).flatten()
            if len(ns_np) < self.n_templates:
                ns_np = np.pad(ns_np, (0, self.n_templates - len(ns_np)), 'constant')
            else:
                ns_np = ns_np[:self.n_templates]
            fixed_next_states.append(torch.tensor(ns_np, dtype=torch.float32))

        state_b = torch.stack(fixed_states)
        next_state_b = torch.stack(fixed_next_states)
        
        current_batch_size = state_b.size(0)
        
        rewards_clean = [r[0] if isinstance(r, (list, np.ndarray)) and len(r) > 0 else r for r in rewards]
        dones_clean = [float(d[0]) if isinstance(d, (list, np.ndarray)) and len(d) > 0 else float(d) for d in dones]

        action_b = torch.tensor(actions, dtype=torch.long).view(current_batch_size, 1)
        reward_b = torch.tensor(np.array(rewards_clean), dtype=torch.float32).view(current_batch_size, 1)
        done_b = torch.tensor(np.array(dones_clean), dtype=torch.float32).view(current_batch_size, 1)
        
        if self.agent_mode == 'classical':
            policy_outputs = self.policy_net(state_b).view(current_batch_size, -1)
            current_q_values = policy_outputs.gather(1, action_b)
            
            with torch.no_grad():
                target_outputs = self.target_net(next_state_b).view(current_batch_size, -1)
                max_next_q_values = target_outputs.max(1)[0].view(current_batch_size, 1)
                target_q_values = reward_b + (self.gamma * max_next_q_values * (1 - done_b))
            
            loss = self.loss_fn(current_q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            if random.random() < 0.1:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
        elif self.agent_mode == 'quantum':
            def closure():
                policy_outputs = self.policy_net(state_b).view(current_batch_size, -1)
                current_q = policy_outputs.gather(1, action_b)
                
                with torch.no_grad():
                    target_outputs = self.policy_net(next_state_b).view(current_batch_size, -1)
                    max_next_q = target_outputs.max(1)[0].view(current_batch_size, 1)
                    target_q = reward_b + (self.gamma * max_next_q * (1 - done_b))
                    
                loss = self.loss_fn(current_q, target_q)
                return loss
                
            self.optimizer.step(closure)

    def run_training(self):
        print(f"[Worker Client {self.replica_id}] Initiating registration protocol with Master Router...")
        registered = self.register_to_master(local_hostname=self.db_host, local_port=self.db_port)
        if not registered:
            print(f"[Worker Client {self.replica_id}] Registration failed. Proceeding with caution...")
            
        print(f"[Worker Client {self.replica_id}] Launching local environment worker loop...")
        
        current_cost_tracker = 0.0
        current_storage_usage = 0.0
        costs_per_template = [0.0] * self.n_templates
        
        # self.candidates = [
        #     ('lineitem', ['l_orderkey']),
        #     ('lineitem', ['l_partkey']),
        #     ('lineitem', ['l_suppkey']),
        #     ('lineitem', ['l_shipdate']),
        #     ('lineitem', ['l_commitdate']),
        #     ('lineitem', ['l_receiptdate']),
        #     ('lineitem', ['l_returnflag']),
            
        #     ('orders', ['o_custkey']),
        #     ('orders', ['o_orderdate']),
        #     ('orders', ['o_orderkey']),
            
        #     ('customer', ['c_nationkey']),
        #     ('customer', ['c_mktsegment']),
        #     ('supplier', ['s_nationkey']),
        #     ('supplier', ['s_suppkey']),
            
        #     ('part', ['p_partkey']),
        #     ('part', ['p_type']),
        #     ('part', ['p_size']),
        #     ('partsupp', ['ps_partkey']),
        #     ('partsupp', ['ps_suppkey']),
            
        #     ('lineitem', ['l_partkey', 'l_suppkey']),
        #     ('orders', ['o_custkey', 'o_orderdate'])
        # ]

        from common.query_loader import load_training_set_queries
        queries, templates = load_training_set_queries('./workload_output/', fraction=1.0)

        self.candidates = self._generate_candidates(queries, templates)

        self.templates_map = list(range(self.n_templates))
        
        if self.env is None:
            self.env = LocalIndexingEnv(
                replica_id=self.replica_id, hostname=self.db_host, port=self.db_port,
                user=self.db_user, password=self.db_password, db_name=self.db_name,
                candidates=self.candidates, templates=self.templates_map,
                n_templates=self.n_templates, storage_budget=self.storage_budget,
                agent_type=self.agent_mode
            )
            
        self._init_agent_networks()
        local_state, _ = self.env.reset()
        
        while True:
            try:
                metrics = qdina_pb2.LocalMetrics(
                    replica_id=self.replica_id,
                    total_cost=current_cost_tracker,
                    costs=costs_per_template,
                    storage_used=current_storage_usage,
                    active_indexes=self.env.get_active_index_names()
                )
                response = self.stub.SubmitMetricsAndGetWorkload(metrics)
                
                if response.stop_training:
                    print(f"[Worker Client {self.replica_id}] Master broadcasted stop_training signal. Synchronizing local episode boundary.")
                    local_state, _ = self.env.reset()
                    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                    current_cost_tracker = 0.0
                    current_storage_usage = 0.0
                    costs_per_template = [0.0] * self.n_templates
                    continue
                
                current_queries = list(response.queries)
                if not current_queries:
                    print(f"[Worker Client {self.replica_id}] No sub-workload queries assigned to this node for the current step.")
                    time.sleep(0.5)
                    current_queries = list(response.queries)
                if not current_queries:
                    print(f"[Worker Client {self.replica_id}] No sub-workload queries assigned to this node. Reporting idle state.")
                    current_cost_tracker = 0.0
                    costs_per_template = [0.0] * self.n_templates
                    time.sleep(0.2)
                    continue
                
                print(f"[Worker Client {self.replica_id}] Sliced workload received containing {len(current_queries)} active queries.")
                dynamic_templates_map = [hash(q_text) % self.n_templates for q_text in current_queries]
                self.env.templates = dynamic_templates_map
                                
                action = self._select_action(local_state)
                next_state, reward, terminated, truncated, info = self.env.step(action, queries=current_queries)

                if terminated:
                    print(f"[Worker Client {self.replica_id}] Local budget exceeded. Resetting environment.")
                    local_state, _ = self.env.reset()
                    current_cost_tracker = 0.0
                    current_storage_usage = 0.0
                    costs_per_template = [0.0] * self.n_templates
                    continue

                self.local_memory.push(local_state, action, next_state, reward, terminated)
                local_state = next_state
                
                self._optimize_local_model()
                
                current_cost_tracker = info.get('total_cost', 0.0)
                current_storage_usage = info.get('storage', 0.0)
                
                if 'costs' in info:
                    costs_per_template = [float(c) for c in info['costs']]
                else:
                    costs_per_template = [current_cost_tracker / self.n_templates] * self.n_templates
                    
                print(f"[Worker Client {self.replica_id}] Local Step Finished. Total Sliced Cost: {current_cost_tracker:.1f} | Storage: {current_storage_usage:.1f}MB | Epsilon: {self.epsilon:.2f}")

                if not response.stop_training:
                    self.epsilon = max(0.3, self.epsilon * 0.999)

            except grpc.RpcError as e:
                print(f"[Worker Client {self.replica_id}] Connection lost with master router. Retrying in 5 seconds... ({e.code()})")
                time.sleep(5.0)

    def _generate_candidates(self, queries: list[str], templates: list[int]) -> list[tuple[str, tuple[str, ...]]]:
        replica = Replica(
            id=self.replica_id,
            hostname=self.db_host,
            port=self.db_port,
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password
        )

        preprocessor = Preprocessor(
            profiler=Profiler(),
            database=replica,
            max_index_width=2,
            queries=queries,
            templates=templates
        )

        preprocessor.preprocess(candidate_path=None, max_candidates=None)
        
        candidates_with_table = []
        for cand in preprocessor.candidates:
            table = preprocessor.cols_to_table[cand[0]]
            candidates_with_table.append((table, cand))
        
        return candidates_with_table