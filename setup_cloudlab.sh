#!/bin/bash
exec > >(tee -a /var/log/cloudlab_startup.log) 2>&1
set -ex

ROLE=$1                 # "router" or "worker"
WORKER_COUNT=${2:-2}    # Total replica count defined in CloudLab
SF=${3:-10}             # TPC-H Scale Factor
BUDGET=${4:-5000000000} # Storage index budget in bytes
NODE_ID=${5:-1}         # Node replica ID (1..N for workers)

export DEBIAN_FRONTEND=noninteractive

# 1. Clean up lingering APT locks from early boot
sleep 5
systemctl stop unattended-upgrades.service apt-daily.service apt-daily-upgrade.service 2>/dev/null || true
pkill -9 -f unattended-upg 2>/dev/null || true
pkill -9 -f apt-get 2>/dev/null || true
rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/lib/apt/lists/lock /var/cache/apt/archives/lock 2>/dev/null || true
dpkg --configure -a 2>/dev/null || true

# 2. Add deadsnakes PPA and install Python 3.12
apt-get update -qq
apt-get install -y -qq software-properties-common ca-certificates dirmngr
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update -qq
apt-get install -y -qq -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" \
    git tmux netcat-openbsd curl build-essential gcc make psmisc \
    python3.12 python3.12-venv python3.12-dev

# 3. Setup Python virtual environment
REPO_DIR="/decentral-qdina"
cd "$REPO_DIR"
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 4. Dynamically generate replicas-cloudlab.csv based on WORKER_COUNT
CSV_FILE="$REPO_DIR/replicas-cloudlab.csv"
echo "id,host,port,user,password,dbname" > "$CSV_FILE"
for idx in $(seq 1 "$WORKER_COUNT"); do
    ip_host="10.10.1.$((10 + idx))"
    echo "${idx},${ip_host},5432,sam,,tpchdb" >> "$CSV_FILE"
done

if [ "$ROLE" == "router" ]; then
    BENCH_DIR="/qdina-bench"
    rm -rf "$BENCH_DIR"
    git clone https://github.com/const-sambird/qdina-bench.git "$BENCH_DIR"
    chmod -R 777 "$BENCH_DIR"
    cp "$CSV_FILE" "$BENCH_DIR/replicas.csv"

    rm -rf "$BENCH_DIR/tpc-h"
    git clone https://github.com/gregrahn/tpch-kit.git "$BENCH_DIR/tpc-h"
    cd "$BENCH_DIR/tpc-h/dbgen"
    make MACHINE=LINUX DATABASE=POSTGRESQL -s
    chmod -R 777 "$BENCH_DIR/tpc-h"

    if [ -d "$BENCH_DIR/tpc-h/queries" ] && [ ! -d "$BENCH_DIR/queries" ]; then
        ln -sfn "$BENCH_DIR/tpc-h/queries" "$BENCH_DIR/queries"
    fi

    cat << 'MSG' > /etc/profile.d/qdina_welcome.sh
echo ""
echo "======================================================="
echo "  [ROUTER] CENTRAL ROUTER NODE ACTIVE"
echo "  Command to monitor: sudo tmux attach -t qdina"
echo "======================================================="
echo ""
MSG

    cd "$REPO_DIR"
    tmux new-session -d -s qdina -c "$REPO_DIR" bash -c \
      "source venv/bin/activate && time python3 -m router.main_router --mode drift --episodes 100 --config replicas-cloudlab.csv --workload-dir ./workload_output --seed 100; exec bash"

elif [ "$ROLE" == "worker" ]; then
    if [ -f "$REPO_DIR/build_tpch_db.sh" ]; then
        chmod +x "$REPO_DIR/build_tpch_db.sh"
        bash "$REPO_DIR/build_tpch_db.sh" -s "$SF" > /var/log/tpch_build.log 2>&1
    fi

    if [ -f /etc/postgresql/17/main/pg_hba.conf ]; then
        sed -i '/host.*sam.*127\.0\.0\.1/d' /etc/postgresql/17/main/pg_hba.conf
        sed -i '/host.*tpchdb.*sam.*10\.10\.1\.0/d' /etc/postgresql/17/main/pg_hba.conf
        sed -i '1i host    all             sam             127.0.0.1/32            trust' /etc/postgresql/17/main/pg_hba.conf
        sed -i '$a host    tpchdb          sam             10.10.1.0/24            trust' /etc/postgresql/17/main/pg_hba.conf
        sed -i '/host.*0.0.0.0\/0/d' /etc/postgresql/17/main/pg_hba.conf
        sed -i '/host.*::\/0/d' /etc/postgresql/17/main/pg_hba.conf
        systemctl restart postgresql
    fi

    cat << MSG > /etc/profile.d/qdina_welcome.sh
echo ""
echo "======================================================="
echo "  [AGENT] WORKER NODE ${NODE_ID} ACTIVE"
echo "  Command to monitor: sudo tmux attach -t qdina"
echo "======================================================="
echo ""
MSG

    echo "Waiting for Central Router at 10.10.1.1:50051..."
    while ! nc -z 10.10.1.1 50051; do
        sleep 2
    done

    tmux new-session -d -s qdina -c "$REPO_DIR" bash -c \
      "source venv/bin/activate && python3 -m agent.main_agent --id ${NODE_ID} --mode classical --server 10.10.1.1:50051 --config replicas-cloudlab.csv --budget-mode ignore --storage-budget ${BUDGET}; exec bash"
fi