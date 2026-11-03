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
                 db_name: str, budget_mode: str, storage_budget: float = 10.0,
                 router_mode: str = 'learned'):
        '''
        Decentralized gRPC Client worker orchestrating local reinforcement learning indexing.

        :param router_mode: forwarded by main_agent.py.
          - 'heuristic' -> every step, compute full-workload costs (Phase 1 of two-phase routing)
          - 'static'    -> use slice costs (cheap, routing is frozen anyway)
          - 'learned'   -> use slice costs (cheap, routing barely moves)
        '''
        self.replica_id = replica_id
        self.agent_mode = agent_mode.lower()
        self.storage_budget = storage_budget
        self.db_host = db_host
        self.db_port = db_port
        self.db_user = db_user
        self.db_password = db_password
        self.db_name = db_name
        self.budget_mode = budget_mode
        self.router_mode = router_mode.lower()

        print(f"[Worker Client {self.replica_id}] Linking gRPC channel to {server_address}...")
        self.channel = grpc.insecure_channel(server_address)
        self.stub = qdina_pb2_grpc.QDinaServiceStub(self.channel)

        self.n_templates = 22
        self.n_candidates = 0
        self.candidates = []
        self.templates_map = []

        # Real template ID lookup (query text -> template ID).
        self.query_to_template = {}

        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.loss_fn = nn.MSELoss()

        self.local_memory = ReplayMemory(capacity=50000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.episode_step_count = 0

        self.env = None

        self._step_counter = 0
        self.target_update_freq = 10
        self.param_layers = 5

        # Diagnostic counters for heuristic mode
        self._heuristic_call_count = 0
        self._heuristic_total_seconds = 0.0

    def register_to_master(self, local_hostname: str = '127.0.0.1', local_port: int = 5432):
        """Register this worker node with the master router via gRPC.

        Returns:
            bool: True if registration succeeded, False otherwise.
        """
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
        """Initialize the policy and target networks and the optimizer."""
        n_actions = self.env.action_space.n
        n_observations = 2 * self.n_templates + self.n_candidates

        if self.agent_mode == 'classical':
            self.policy_net = DQN(n_observations, n_actions, layer_features=[256, 128, 64])
            self.target_net = DQN(n_observations, n_actions, layer_features=[256, 128, 64])
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
            print(f"[Worker Client {self.replica_id}] Classical DQN Policy Network built successfully.")
        elif self.agent_mode == 'quantum':
            n_qubits = max(8, int(n_actions - 1).bit_length())

            self.policy_net = QuantumDQN(
                n_inputs=n_observations,
                n_qubits=n_qubits,
                n_actions=n_actions,
                qnn_type='twolocal',
                qnn_output='layer',
                n_shots=1024,
                torch_device='cpu',
                param_layers=self.param_layers
            )
            self.target_net = QuantumDQN(
                n_inputs=n_observations,
                n_qubits=n_qubits,
                n_actions=n_actions,
                qnn_type='twolocal',
                qnn_output='layer',
                n_shots=1024,
                torch_device='cpu',
                param_layers=self.param_layers
            )
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

            self.optimizer = SPSAOptimiser(
                self.policy_net,
                lr=0.1,
                device=next(self.policy_net.parameters()).device
            )
            print(f"[Worker Client {self.replica_id}] Quantum DQN with {n_qubits} qubits built successfully.")

    def _select_action(self, state):
        """Select an action using an epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randint(0, self.env.action_space.n - 1)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            action = q_values.argmax().item()
        return action

    def _optimize_local_model(self):
        """Perform one optimization step on the local DQN using the replay memory."""
        if len(self.local_memory) < self.batch_size:
            return

        transitions = self.local_memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        expected_size = 2 * self.n_templates + self.n_candidates

        fixed_states = []
        for s in states:
            s_np = np.asarray(s, dtype=np.float32).flatten()
            if len(s_np) < expected_size:
                s_np = np.pad(s_np, (0, expected_size - len(s_np)), 'constant')
            else:
                s_np = s_np[:expected_size]
            fixed_states.append(torch.tensor(s_np, dtype=torch.float32))

        fixed_next_states = []
        for ns in next_states:
            ns_np = np.asarray(ns, dtype=np.float32).flatten()
            if len(ns_np) < expected_size:
                ns_np = np.pad(ns_np, (0, expected_size - len(ns_np)), 'constant')
            else:
                ns_np = ns_np[:expected_size]
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

            self._step_counter += 1
            if self._step_counter % self.target_update_freq == 0:
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

    def _report_costs(self, info=None):
        """Return the per-template cost vector to report to the master.

        TWO-PHASE ROUTER (heuristic mode only):
          Phase 1 (this worker): compute cost of ALL 22 templates × ALL ~1100
                                 queries on THIS replica. This produces a dense
                                 cost vector that the router can compare against
                                 other replicas.
          Phase 2 (router):      route each template to the replica with the
                                 smallest Phase-1 cost.

        In heuristic mode we MUST always compute the full-workload cost — the
        router's diversity algorithm needs dense data to make good decisions.
        If `_full_workload_queries` is empty (e.g., `workload_output` not
        populated on this worker), we crash rather than silently report slice
        costs. Silent fallback previously made heuristic look fast while
        routing on garbage data.

        In static/learned modes we use slice costs — cheap and correct
        because:
          - static:  routing is frozen, slice cost == full cost for the
                     assigned replica.
          - learned: routing changes 1/3 of steps, so ~2/3 of logged makespans
                     are exact.
        """
        if self.router_mode == 'heuristic':
            try:
                _t0 = time.time()
                costs = self.env.estimate_full_workload_costs()
                _dt = time.time() - _t0
            except RuntimeError as e:
                print(f"[Worker Client {self.replica_id}] " + "=" * 70)
                print(f"[Worker Client {self.replica_id}] FATAL: {e}")
                print(f"[Worker Client {self.replica_id}]")
                print(f"[Worker Client {self.replica_id}] Heuristic mode is ON but the workload is EMPTY")
                print(f"[Worker Client {self.replica_id}] on this worker. The two-phase router requires")
                print(f"[Worker Client {self.replica_id}] Phase-1 (full-workload cost) to succeed.")
                print(f"[Worker Client {self.replica_id}]")
                print(f"[Worker Client {self.replica_id}] FIX on this node:")
                print(f"[Worker Client {self.replica_id}]   ls /decentral-qdina/workload_output/*.sql | wc -l   # ~1100 expected")
                print(f"[Worker Client {self.replica_id}]   ls /decentral-qdina/templates.txt                 # should exist")
                print(f"[Worker Client {self.replica_id}] " + "=" * 70)
                sys.exit(1)

            self._heuristic_call_count += 1
            self._heuristic_total_seconds += _dt

            # Print first call and every 10th to confirm Phase-1 is running
            # (costs in the ~1B range, NOT ~100M slice costs).
            if self._heuristic_call_count == 1 or self._heuristic_call_count % 10 == 0:
                avg = self._heuristic_total_seconds / self._heuristic_call_count
                print(f"[Worker Client {self.replica_id}] [HEURISTIC] Phase-1 full-workload "
                      f"cost took {_dt:.2f}s (avg {avg:.2f}s, "
                      f"{len(self.env._full_workload_queries)} queries, "
                      f"call #{self._heuristic_call_count}, sum={sum(costs):.0f})")

            return costs

        # static / learned: use slice costs (fast path)
        if info is not None:
            return info.get('costs', [0.0] * self.n_templates)
        return [0.0] * self.n_templates

    def run_training(self):
        """Main training loop for the local reinforcement learning agent."""
        print(f"[Worker Client {self.replica_id}] Initiating registration protocol with Master Router...")
        registered = self.register_to_master(local_hostname=self.db_host, local_port=self.db_port)
        if not registered:
            print(f"[Worker Client {self.replica_id}] Registration failed. Proceeding with caution...")

        print(f"[Worker Client {self.replica_id}] Launching local environment worker loop...")

        current_cost_tracker = 0.0
        current_storage_usage = 0.0
        costs_per_template = [0.0] * self.n_templates

        from common.query_loader import load_training_set_queries
        queries, templates = load_training_set_queries('./workload_output/', fraction=1.0)

        print(f"[Worker Client {self.replica_id}] Loaded {len(queries)} training queries "
              f"({len(set(templates))} unique templates) from ./workload_output/")

        self.query_to_template = {q: t for q, t in zip(queries, templates)}

        self.candidates = self._generate_candidates(queries, templates)
        self.n_candidates = len(self.candidates)

        self.templates_map = list(range(self.n_templates))

        if self.env is None:
            self.env = LocalIndexingEnv(
                replica_id=self.replica_id, hostname=self.db_host, port=self.db_port,
                user=self.db_user, password=self.db_password, db_name=self.db_name,
                candidates=self.candidates, templates=self.templates_map,
                n_templates=self.n_templates, storage_budget=self.storage_budget,
                agent_type=self.agent_mode, budget_mode=self.budget_mode
            )

        # Give the env the full workload so it can cost every template (DINA-style).
        self.env.set_full_workload(queries, templates)

        if self.router_mode == 'heuristic':
            stored = len(self.env._full_workload_queries)
            if stored == 0:
                print(f"[Worker Client {self.replica_id}] " + "=" * 70)
                print(f"[Worker Client {self.replica_id}] FATAL: heuristic mode + empty workload.")
                print(f"[Worker Client {self.replica_id}] `env.set_full_workload()` received 0 queries.")
                print(f"[Worker Client {self.replica_id}] Phase-1 (full-workload cost) cannot run.")
                print(f"[Worker Client {self.replica_id}] Check `./workload_output/*.sql` exists on this node.")
                print(f"[Worker Client {self.replica_id}] " + "=" * 70)
                sys.exit(1)
            print(f"[Worker Client {self.replica_id}] ✓ env._full_workload_queries = {stored} queries.")

        self._init_agent_networks()
        local_state, _ = self.env.reset()

        # ─────────────────────────────────────────────────────────────────────
        # TWO-PHASE ROUTER — Phase 1 bootstrap (heuristic mode only)
        #
        # The router's diversity routing needs a DENSE cost matrix at step 0.
        # If we enter the loop with `costs_per_template = [0]*22`, the router
        # sees all-zero costs on all replicas, tie-breaks everything to
        # replica 0, and never recovers.
        #
        # So BEFORE the first SubmitMetricsAndGetWorkload, we compute the
        # full-workload cost matrix on this replica (Phase 1) and send it.
        # The router will have a real (if initially similar-across-replicas)
        # matrix to route with.
        # ─────────────────────────────────────────────────────────────────────
        if self.router_mode == 'heuristic':
            print(f"[Worker Client {self.replica_id}] Phase-1 bootstrap: computing initial full-workload costs...")
            _t0 = time.time()
            costs_per_template = self.env.estimate_full_workload_costs()
            current_cost_tracker = float(sum(costs_per_template))
            _dt = time.time() - _t0
            print(f"[Worker Client {self.replica_id}] Phase-1 bootstrap done in {_dt:.2f}s: sum={current_cost_tracker:.0f}")

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
                    response, local_state, current_cost_tracker, current_storage_usage, costs_per_template = self._handle_stop_training()

                current_queries = list(response.queries)
                if not current_queries:
                    print(f"[Worker Client {self.replica_id}] Idle: no sub-workload queries assigned.")
                    if self.router_mode != 'heuristic':
                        costs_per_template = [0.0] * self.n_templates
                        current_cost_tracker = 0.0
                    time.sleep(0.2)
                    continue

                if response.epsilon is not None:
                    self.epsilon = response.epsilon

                if response.param_layers is not None:
                    self.param_layers = response.param_layers

                print(f"[Worker Client {self.replica_id}] Sliced workload received containing {len(current_queries)} active queries.")

                dynamic_templates_map = [self.query_to_template.get(q_text, 0) for q_text in current_queries]
                self.env.templates = dynamic_templates_map

                action = self._select_action(local_state)
                next_state, reward, terminated, truncated, info = self.env.step(action, queries=current_queries)

                if terminated:
                    print(f"[Worker Client {self.replica_id}] Local budget exceeded. Resetting environment.")

                    costs_per_template = [0.0] * self.n_templates
                    current_cost_tracker = 0.0
                    current_storage_usage = info.get('storage', 0.0)
                    active_indexes = self.env.get_active_index_names()

                    metrics = qdina_pb2.LocalMetrics(
                        replica_id=self.replica_id,
                        total_cost=current_cost_tracker,
                        costs=costs_per_template,
                        storage_used=current_storage_usage,
                        active_indexes=active_indexes,
                        local_reset=True
                    )
                    response = self.stub.SubmitMetricsAndGetWorkload(metrics)
                    if response.stop_training:
                        response, local_state, current_cost_tracker, current_storage_usage, costs_per_template = self._handle_stop_training()
                        continue

                    local_state, _ = self.env.reset()
                    current_cost_tracker = 0.0
                    current_storage_usage = 0.0
                    costs_per_template = [0.0] * self.n_templates
                    continue

                self.local_memory.push(local_state, action, next_state, reward, terminated)
                local_state = next_state

                self._optimize_local_model()

                current_storage_usage = info.get('storage', 0.0)

                costs_per_template = self._report_costs(info=info)
                current_cost_tracker = float(sum(costs_per_template))

                storage_str = f"{current_storage_usage / 1_000_000_000:.2f} GB"
                print(f"[Worker Client {self.replica_id}] Local Step Finished. Total Sliced Cost: {current_cost_tracker:.1f} | Reward {reward:.1f} | Storage: {storage_str} | Epsilon: {self.epsilon:.3f}")

            except grpc.RpcError as e:
                print(f"[Worker Client {self.replica_id}] Connection lost with master router. Retrying in 5 seconds... ({e.code()})")
                time.sleep(5.0)

    def _generate_candidates(self, queries: list[str], templates: list[int]) -> list[tuple[str, tuple[str, ...]]]:
        """Generate index candidates from the given workload using the Preprocessor."""
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

    def _handle_stop_training(self):
        """Handles a global episode end: resets the local environment, sends an
        acknowledgment with local_reset=True, and waits for the new workload."""
        print(f"[Worker Client {self.replica_id}] Master broadcasted stop_training signal. "
            f"Resetting local environment and acknowledging...")

        knapsack_indexes = self.env.get_knapsack_selection()
        active_indexes = [f"{table}_{'_'.join(cols)}" for table, cols in knapsack_indexes]
        print(f"[Worker {self.replica_id}] Knapsack selected {len(knapsack_indexes)} indexes within {self.storage_budget/1e9:.1f} GB budget.")

        local_state, _ = self.env.reset()
        self.episode_step_count = 0
        current_cost_tracker = 0.0
        current_storage_usage = 0.0

        # Reset costs are ignored by the master (its local_reset branch uses
        # last_valid_*), so we skip the expensive full-workload estimation
        # even in heuristic mode.
        costs_per_template = [0.0] * self.n_templates
        current_cost_tracker = 0.0

        ack_metrics = qdina_pb2.LocalMessage if False else qdina_pb2.LocalMetrics(
            replica_id=self.replica_id,
            total_cost=current_cost_tracker,
            costs=costs_per_template,
            storage_used=current_storage_usage,
            active_indexes=active_indexes,
            local_reset=True
        )

        while True:
            ack_response = self.stub.SubmitMetricsAndGetWorkload(ack_metrics)
            if not ack_response.stop_training:
                return (ack_response, local_state, current_cost_tracker,
                        current_storage_usage, costs_per_template)
            else:
                time.sleep(0.2)