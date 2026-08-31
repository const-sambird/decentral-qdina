#!/bin/bash
exec > >(tee -a /var/log/cloudlab_startup.log) 2>&1
set -ex

ROLE=$1                 # "router" or "worker"
WORKER_COUNT=${2:-2}    # Total replica count
SF=${3:-10}             # TPC-H Scale Factor
NODE_ID=${4:-1}         # Node ID

export DEBIAN_FRONTEND=noninteractive

# 1. Clean up lingering APT locks from early boot
sleep 5
systemctl stop unattended-upgrades.service apt-daily.service apt-daily-upgrade.service 2>/dev/null || true
pkill -9 -f unattended-upg 2>/dev/null || true
pkill -9 -f apt-get 2>/dev/null || true
rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/lib/apt/lists/lock /var/cache/apt/archives/lock 2>/dev/null || true
dpkg --configure -a 2>/dev/null || true

# 2. Add deadsnakes PPA and install dependencies
apt-get update -qq
apt-get install -y -qq software-properties-common ca-certificates dirmngr
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update -qq
apt-get install -y -qq -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" \
    git tmux netcat-openbsd curl build-essential gcc make psmisc \
    python3.12 python3.12-venv python3.12-dev

# ==============================================================================
# ROUTER SETUP (BENCHMARKER)
# ==============================================================================
if [ "$ROLE" == "router" ]; then
    BENCH_DIR="/qdina-bench"
    mkdir -p "$BENCH_DIR"
    
    # Clone repository into temporary directory
    git clone https://github.com/const-sambird/qdina-bench.git /tmp/qdina-bench
    cp -rn /tmp/qdina-bench/* "$BENCH_DIR"/ || true
    rm -rf /tmp/qdina-bench

    cd "$BENCH_DIR"
    python3.12 -m venv venv
    source venv/bin/activate
    
    if [ -f requirements.txt ]; then
        pip install --upgrade pip
        pip install -r requirements.txt
    fi

    # Prepare TPC-H data generation tool
    rm -rf "$BENCH_DIR/tpc-h"
    git clone https://github.com/gregrahn/tpch-kit.git "$BENCH_DIR/tpc-h"
    cd "$BENCH_DIR/tpc-h/dbgen"
    make MACHINE=LINUX DATABASE=POSTGRESQL -s
    chmod -R 777 "$BENCH_DIR/tpc-h"

    if [ -d "$BENCH_DIR/tpc-h/queries" ] && [ ! -d "$BENCH_DIR/queries" ]; then
        ln -sfn "$BENCH_DIR/tpc-h/queries" "$BENCH_DIR/queries"
    fi

    chmod -R 777 "$BENCH_DIR"

    # Clone decentral-qdina repository to get workload_output
    REPO_DIR="/decentral-qdina"
    rm -rf "$REPO_DIR"
    git clone https://github.com/const-sambird/decentral-qdina.git "$REPO_DIR"
    
    cd "$BENCH_DIR"
    
    # Setup tmux session
    SESSION_NAME="din-bench"
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        tmux new-session -d -s "$SESSION_NAME" -c "$BENCH_DIR" "source venv/bin/activate && bash"
        tmux send-keys -t "$SESSION_NAME" "source venv/bin/activate" C-m
    fi
    
    echo "Starting benchmark in the '$SESSION_NAME' tmux session..."
    tmux send-keys -t "$SESSION_NAME" "time python run.py -s $SF -v -c --copy-source $REPO_DIR/workload_output/ -x h all" C-m

    cat << 'MSG' > /etc/profile.d/dina_bench_welcome.sh
echo ""
echo "======================================================="
echo "  [BENCHMARKER] CENTRAL ROUTER NODE ACTIVE"
echo "  Benchmark directory: /qdina-bench"
echo "  Command to monitor: sudo tmux attach -t din-bench"
echo "======================================================="
echo ""
MSG

# ==============================================================================
# WORKER SETUP (REPLICAS)
# ==============================================================================
elif [ "$ROLE" == "worker" ]; then
    REPO_DIR="/decentral-qdina"
    
    # Clone the repository if not already present
    if [ ! -d "$REPO_DIR" ]; then
        git clone https://github.com/const-sambird/decentral-qdina.git "$REPO_DIR"
    fi
    
    # Run build_tpch_db.sh with benchmark mode to skip data generation
    if [ -f "$REPO_DIR/build_tpch_db.sh" ]; then
        chmod +x "$REPO_DIR/build_tpch_db.sh"
        bash "$REPO_DIR/build_tpch_db.sh" -s "$SF" -m benchmark > /var/log/tpch_build.log 2>&1
    fi

    # Configure PostgreSQL for network access
    if [ -f /etc/postgresql/17/main/pg_hba.conf ]; then
        sed -i '/host.*sam.*127\.0\.0\.1/d' /etc/postgresql/17/main/pg_hba.conf
        sed -i '/host.*tpchdb.*sam.*10\.10\.1\.0/d' /etc/postgresql/17/main/pg_hba.conf
        sed -i '1i host    all             sam             127.0.0.1/32            trust' /etc/postgresql/17/main/pg_hba.conf
        sed -i '$a host    tpchdb          sam             10.10.1.0/24            trust' /etc/postgresql/17/main/pg_hba.conf
        sed -i '/host.*0.0.0.0\/0/d' /etc/postgresql/17/main/pg_hba.conf
        sed -i '/host.*::\/0/d' /etc/postgresql/17/main/pg_hba.conf
        systemctl restart postgresql
    fi

    cat << MSG > /etc/profile.d/dina_bench_welcome.sh
echo ""
echo "======================================================="
echo "  [REPLICA] WORKER NODE ${NODE_ID} ACTIVE"
echo "  Database is installed and waiting for benchmarking data."
echo "======================================================="
echo ""
MSG
fi