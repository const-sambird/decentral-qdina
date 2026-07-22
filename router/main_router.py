import sys
import os
import time
import argparse
import grpc
from concurrent import futures
from protos import qdina_pb2_grpc

from router.network_server import QDinaServerServicer
from common.workload_manager import WorkloadManager
from common.util import parse_replicas_csv
from common.query_loader import load_training_set_queries

def prepare_workload_directory(templates_file, output_dir):
    '''
    Prepares the workload directory by splitting templates.txt into individual
    SQL files formatted as {template_id}_{instance}.sql, matching query_loader expectations.
    '''
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(templates_file):
        print(f"[Error] Seed file missing at: {templates_file}")
        sys.exit(1)
        
    with open(templates_file, 'r') as f:
        content = f.read()
        
    queries = [q.strip() + ";" for q in content.split(';') if q.strip()]
    
    print(f"[Preprocessor] Generating individual query files inside {output_dir}...")
    for idx, query_text in enumerate(queries):
        template_id = (idx % 22) + 1
        instance_id = idx // 22
        filename = f"{template_id}_{instance_id}.sql"
        with open(os.path.join(output_dir, filename), 'w') as out_f:
            out_f.write(query_text)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="decentral-qdina Central Orchestrator Router.")
    parser.add_argument('--mode', type=str, default='drift', choices=['uniform', 'drift'], help="Workload generation mode")
    parser.add_argument('--episodes', type=int, default=10, help="Number of full training episodes")
    parser.add_argument('--workload-dir', type=str, default='./workload_output', help="Directory for individual query files")
    parser.add_argument('--templates-seed', type=str, default='templates.txt', help="Seed file containing raw query SQL text strings")
    parser.add_argument('--config', type=str, default='replicas.csv', help="Cluster replicas topology configuration file")
    
    args = parser.parse_args()
    
    replicas_config = parse_replicas_csv(args.config)
    num_replicas = len(replicas_config)
    
    if num_replicas == 0:
        print(f"[Error] No active database replicas detected inside {args.config}!")
        sys.exit(1)
        
    print(f"[Master Orchestrator] Loaded cluster layout template. Expecting {num_replicas} active replicas.")
    
    # Preprocess queries into standard workspace directory structure
    if not os.path.exists(args.workload_dir) or len(os.listdir(args.workload_dir)) == 0:
        prepare_workload_directory(args.templates_seed, args.workload_dir)
        
    print(f"[Master Orchestrator] Invoking query_loader.load_training_set_queries on {args.workload_dir}...")
    # Load all queries and their template mapping for the initial workload
    initial_queries, initial_map = load_training_set_queries(args.workload_dir, fraction=1.0)
    n_templates = 22
    initial_map = [t % n_templates for t in initial_map]
    
    # Initialize the WorkloadManager with the loaded queries and mapping
    workload_mgr = WorkloadManager(initial_queries, initial_map, execution_mode=args.mode, fraction=1.0)

    # Instantiate the Server Servicer using the dynamic configuration parameters discovered
    print(f"[Master Orchestrator] Initializing QDinaServerServicer with {num_replicas} replicas and 22 templates...")
    servicer = QDinaServerServicer(n_replicas=num_replicas, n_templates=22, batch_size=64)
    servicer.execution_mode = args.mode
    servicer.current_workload_pool = initial_queries
    servicer.workload_templates_map = initial_map
    
    # Start the gRPC server infrastructure
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    qdina_pb2_grpc.add_QDinaServiceServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:50051')
    
    try:
        server.start()
        print("[Master Orchestrator] gRPC Server active and listening on port 50051...")
        
        print(f"[Master Orchestrator] Waiting for all {num_replicas} database replicas...")
        while True:
            with servicer.lock:
                current_count = len(servicer.registered_workers)
            
            if current_count >= num_replicas:
                print(f"[Master Orchestrator] All {num_replicas} replicas connected!")
                servicer.ready_to_train = True 
                break
                
            print(f"[Master Orchestrator] Registered {current_count}/{num_replicas} replicas... waiting...")
            time.sleep(2.0)
        servicer.ready_to_train = True
        print("[Master Orchestrator] All replicas connected! Launching training episodes...")

        print(f"[Master Orchestrator] Total queries in initial workload: {len(initial_queries)}")
        
        steps_per_episode = 100
        epsilon_start = 1.0
        epsilon_min = 0.30
        decay_rate = 0.9999
        
        for episode in range(args.episodes):
            print(f"\n--- [Master Orchestrator] Starting Global Episode {episode + 1}/{args.episodes} ---")
            
            # Dynamic epsilon decay for exploration-exploitation balance
            servicer.epsilon = max(epsilon_min, epsilon_start * decay_rate)
            
            with servicer.lock:
                servicer.stop_training_signal = False
                servicer.global_step_counter = 0
            
            if args.mode == 'drift' and episode > 0:
                workload_mgr.update_workload()
                current_queries = workload_mgr.workload()
                templates_map = workload_mgr.templates()
                with servicer.lock:
                    servicer.current_workload_pool = current_queries
                    servicer.workload_templates_map = templates_map
                print("[Master Orchestrator] Workload configuration updated (Drift active).")
            
            while True:
                with servicer.lock:
                    current_step = servicer.global_step_counter
                
                if current_step >= steps_per_episode:
                    break
                    
                time.sleep(0.05)

            print(f"[Master Orchestrator] Global Episode {episode + 1} Done. Broadcasting stop_training signal.")
            with servicer.lock:
                servicer.stop_training_signal = True

            # Wait for all workers to acknowledge the episode reset before proceeding
            print("[Master Orchestrator] Waiting for all workers to acknowledge episode reset...")
            while True:
                with servicer.lock:
                    if not servicer.stop_training_signal:
                        break
                time.sleep(0.1)
            print("[Master Orchestrator] All workers have reset. Proceeding to next episode.")
 
            servicer.agent.target_net.load_state_dict(servicer.agent.policy_net.state_dict())
            
            servicer.export_benchmark_files(output_dir=".")

        print("\n[Master Orchestrator] Training completed successfully!")
        print("[Master Orchestrator] Finalizing outstanding gRPC requests (grace period)...")
        server.stop(grace=3.0)
        print("[Master Orchestrator] Central Router server closed cleanly. Exiting.")

    except KeyboardInterrupt:
        print("\n[Master Orchestrator] Shutting down Central Router server safely.")
        server.stop(0)
        sys.exit(0)