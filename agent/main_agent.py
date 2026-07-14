# File: agent/main_agent.py
import sys
import argparse
import os
from agent.network_client import QDinaNetworkClient
from common.util import parse_replicas_csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="decentral-qdina Worker Node Deployment Script")
    parser.add_argument('--id', type=int, required=True, help="Unique local replica identifier ID")
    parser.add_argument('--mode', type=str, default='quantum', choices=['classical', 'quantum'], help="Execution model type")
    parser.add_argument('--server', type=str, default='localhost:50051', help="Master router gRPC server network address")
    parser.add_argument('--config', type=str, default='replicas.csv', help="Path to the replicas configuration CSV file")
    
    args = parser.parse_args()
    
    # Load settings from CSV
    replicas_settings = parse_replicas_csv(args.config)
    
    if args.id not in replicas_settings:
        print(f"[Error] Replica ID {args.id} not found in {args.config}!")
        sys.exit(1)
        
    cfg = replicas_settings[args.id]
    
    print(f"[Worker Node {args.id}] Configuration loaded from CSV successfully.")
    print(f"[Worker Node {args.id}] Targeting Local DB -> {cfg['user']}@{cfg['hostname']}:{cfg['port']}/{cfg['dbname']}")
    print(f"[Worker Node {args.id}] Mode: {args.mode.upper()} layer | Connecting to coordinator: {args.server}")
    
    # Instantiate the client using the exact parsed CSV credentials right from the start
    client = QDinaNetworkClient(
        replica_id=args.id, 
        server_address=args.server, 
        agent_mode=args.mode,
        db_host=cfg['hostname'],
        db_port=cfg['port'],
        db_user=cfg['user'],
        db_password=cfg['password'],
        db_name=cfg['dbname'],
        storage_budget=40.0
    )
    
    try:
        client.run_training()
    except KeyboardInterrupt:
        print(f"\n[Worker Node {args.id}] Safely halting client execution loops and closing channels.")
        sys.exit(0)