# File: agent/main_agent.py
import sys
import argparse
import os
from agent.network_client import QDinaNetworkClient
from common.util import parse_replicas_csv


def prepare_workload_directory(templates_file, output_dir):
    '''
    Splits `templates.txt` into per-query SQL files inside `output_dir`.

    WHY: in heuristic mode, the two-phase router requires EACH worker to
    compute the cost of the FULL workload (~1100 queries × 22 templates)
    on every step. If `workload_output/*.sql` is empty or missing, Phase-1
    cannot run and the router's diversity routing operates on garbage data.
    '''
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(templates_file):
        print(f"[Worker Preflight] WARNING: {templates_file} not found, cannot populate {output_dir}.")
        return False

    with open(templates_file, 'r') as f:
        content = f.read()

    queries = [q.strip() + ";" for q in content.split(';') if q.strip()]
    if not queries:
        print(f"[Worker Preflight] WARNING: {templates_file} is empty.")
        return False

    print(f"[Worker Preflight] Populating {output_dir} from {templates_file} ({len(queries)} queries)...")
    for idx, query_text in enumerate(queries):
        template_id = (idx % 22) + 1
        instance_id = idx // 22
        filename = f"{template_id}_{instance_id}.sql"
        with open(os.path.join(output_dir, filename), 'w') as out_f:
            out_f.write(query_text)
    return True


def count_workload_files(workload_dir):
    '''Count .sql files in `workload_dir`, return 0 if the directory doesn't exist.'''
    if not os.path.exists(workload_dir):
        return 0
    return sum(1 for f in os.listdir(workload_dir) if f.endswith('.sql'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="decentral-qdina Worker Node Deployment Script")
    parser.add_argument('--id', type=int, required=True, help="Unique local replica identifier ID")
    parser.add_argument('--mode', type=str, default='quantum', choices=['classical', 'quantum'], help="Execution model type")
    parser.add_argument('--server', type=str, default='localhost:50051', help="Master router gRPC server network address")
    parser.add_argument('--config', type=str, default='replicas.csv', help="Path to the replicas configuration CSV file")
    parser.add_argument('--budget-mode', type=str, default='enforce', choices=['enforce', 'ignore'], help="Budget enforcement mode.")
    parser.add_argument('--storage-budget', type=int, default=5_000_000_000, help="Storage budget in bytes for indexes.")
    parser.add_argument('--router-mode', type=str, default='learned',
                        choices=['learned', 'static', 'heuristic'],
                        help="MUST match the server's --router-mode. 'heuristic' activates the "
                             "two-phase router: Phase-1 (worker: full-workload cost) + "
                             "Phase-2 (router: cheapest replica per template).")
    parser.add_argument('--workload-dir', type=str, default='./workload_output', help="Directory with per-query SQL files.")
    parser.add_argument('--templates-seed', type=str, default='./templates.txt', help="Seed file with raw SQL templates.")
    args = parser.parse_args()

    # ─── Preflight: ensure workload_output exists BEFORE we start the client ───
    n_files = count_workload_files(args.workload_dir)
    print(f"[Worker Preflight] workload_output/*.sql count: {n_files}")
    print(f"[Worker Preflight] templates.txt exists: {os.path.exists(args.templates_seed)}")

    if n_files == 0:
        print(f"[Worker Preflight] {args.workload_dir} is empty. Populating from {args.templates_seed}...")
        prepare_workload_directory(args.templates_seed, args.workload_dir)
        n_files = count_workload_files(args.workload_dir)
        print(f"[Worker Preflight] After population: {n_files} files")

    if n_files == 0 and args.router_mode == 'heuristic':
        print(f"[Worker Preflight] " + "=" * 70)
        print(f"[Worker Preflight] FATAL: heuristic mode requires a populated workload.")
        print(f"[Worker Preflight]        `./workload_output/*.sql` AND `./templates.txt` are both missing.")
        print(f"[Worker Preflight]        Phase-1 (full-workload cost) cannot run without data.")
        print(f"[Worker Preflight] " + "=" * 70)
        sys.exit(1)

    # Load settings from CSV
    replicas_settings = parse_replicas_csv(args.config)
    if args.id not in replicas_settings:
        print(f"[Error] Replica ID {args.id} not found in {args.config}!")
        sys.exit(1)
    cfg = replicas_settings[args.id]

    print(f"[Worker Node {args.id}] Configuration loaded from CSV successfully.")
    print(f"[Worker Node {args.id}] Targeting Local DB -> {cfg['user']}@{cfg['hostname']}:{cfg['port']}/{cfg['dbname']}")
    print(f"[Worker Node {args.id}] Mode: {args.mode.upper()} layer | Connecting to coordinator: {args.server}")
    print(f"[Worker Node {args.id}] Router mode: {args.router_mode.upper()}  ← MUST match server")
    print(f"[Worker Node {args.id}] Workload files: {n_files} in {args.workload_dir}")

    if args.router_mode == 'heuristic':
        print(f"[Worker Node {args.id}] ⚠️  HEURISTIC mode: Phase-1 will EXPLAIN ~{n_files} queries per step.")
        print(f"[Worker Node {args.id}]    Expected delay: ~10-15s per step (this is normal).")

    client = QDinaNetworkClient(
        replica_id=args.id,
        server_address=args.server,
        agent_mode=args.mode,
        budget_mode=args.budget_mode,
        db_host=cfg['hostname'],
        db_port=cfg['port'],
        db_user=cfg['user'],
        db_password=cfg['password'],
        db_name=cfg['dbname'],
        storage_budget=args.storage_budget,
        router_mode=args.router_mode
    )

    try:
        client.run_training()
    except KeyboardInterrupt:
        print(f"\n[Worker Node {args.id}] Safely halting client execution loops and closing channels.")
        sys.exit(0)