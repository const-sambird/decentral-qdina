#!/bin/bash
exec > >(tee -a /var/log/cloudlab_startup.log) 2>&1
set -ex
set -o pipefail

ROLE=$1                    # "router" or "worker"
WORKER_COUNT=${2:-2}       # Total replica count defined in CloudLab
SF=${3:-10}                # TPC-H Scale Factor
BUDGET=${4:-5000000000}    # Storage index budget in bytes
NODE_ID=${5:-1}            # Node replica ID (1..N for workers)
EPISODES=${6:-100}         # Number of training episodes
SEED=${7:-100}             # Random seed
ROUTER_MODE=${8:-learned}  # Routing strategy: learned | static | heuristic

export DEBIAN_FRONTEND=noninteractive

# Force IPv4 + retries
echo 'Acquire::ForceIPv4 "true";' | tee /etc/apt/apt.conf.d/99force-ipv4
echo 'Acquire::Retries "5";'       | tee /etc/apt/apt.conf.d/99retries

# 1. Clean up lingering APT locks from early boot
sleep 5
systemctl stop unattended-upgrades.service apt-daily.service apt-daily-upgrade.service 2>/dev/null || true
pkill -9 -f unattended-upg 2>/dev/null || true
pkill -9 -f apt-get 2>/dev/null || true
rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/lib/apt/lists/lock /var/cache/apt/archives/lock 2>/dev/null || true
dpkg --configure -a 2>/dev/null || true

# 2. Install base system tools from the MAIN Ubuntu repos only (no PPA)
sed -i 's|us.archive.ubuntu.com|fr.archive.ubuntu.com|g' /etc/apt/sources.list
apt-get update -qq
apt-get install -y -qq \
    git tmux netcat-openbsd curl wget ca-certificates xz-utils \
    build-essential gcc make psmisc \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncursesw5-dev tk-dev libffi-dev liblzma-dev

# 3. Install Python 3.12 via python-build-standalone
#    IMPORTANT: use the "install_only" variant (WITHOUT "-stripped") so that
#    ensurepip bundles the pip/setuptools wheels required by venv.
PYTHON_VERSION="3.12.7"
PYTHON_RELEASE="20241016"
PYTHON_INSTALL_DIR="/opt/python3.12"

if [ ! -x "${PYTHON_INSTALL_DIR}/bin/python3.12" ]; then
    echo "=== Installing Python ${PYTHON_VERSION} from python-build-standalone ==="
    ARCH=$(uname -m)   # x86_64 on CloudLab
    # NOTE: no "-stripped" suffix -> ensurepip wheels are present.
    TARBALL="cpython-${PYTHON_VERSION}+${PYTHON_RELEASE}-${ARCH}-unknown-linux-gnu-install_only.tar.gz"

    URLS=(
        "https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_RELEASE}/${TARBALL}"
        "https://github.com/indygreg/python-build-standalone/releases/download/${PYTHON_RELEASE}/${TARBALL}"
    )

    DOWNLOADED=0
    for url in "${URLS[@]}"; do
        echo "Trying ${url}..."
        if curl -fL --retry 5 --retry-delay 10 --connect-timeout 20 -o /tmp/python3.12.tar.gz "${url}"; then
            DOWNLOADED=1
            break
        fi
    done

    if [ "${DOWNLOADED}" -eq 1 ]; then
        mkdir -p "${PYTHON_INSTALL_DIR}"
        tar -xzf /tmp/python3.12.tar.gz -C "${PYTHON_INSTALL_DIR}" --strip-components=1
        rm -f /tmp/python3.12.tar.gz
        ln -sf "${PYTHON_INSTALL_DIR}/bin/python3.12" /usr/local/bin/python3.12
        ln -sf "${PYTHON_INSTALL_DIR}/bin/pip3.12"    /usr/local/bin/pip3.12
        PYTHON_BIN="/usr/local/bin/python3.12"
    else
        echo "=== All python-build-standalone mirrors failed; falling back to pyenv compilation ==="
        export PYENV_ROOT="/root/.pyenv"
        curl -fsSL https://pyenv.run | bash
        export PATH="$PYENV_ROOT/bin:$PATH"
        eval "$(pyenv init -)"
        pyenv install -s "${PYTHON_VERSION}"
        pyenv global "${PYTHON_VERSION}"
        PYTHON_BIN="${PYENV_ROOT}/versions/${PYTHON_VERSION}/bin/python3.12"

        cat > /etc/profile.d/pyenv.sh << 'EOF'
export PYENV_ROOT="/root/.pyenv"
export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
EOF
        chmod 644 /etc/profile.d/pyenv.sh
    fi
else
    echo "=== Python 3.12 already installed at ${PYTHON_INSTALL_DIR} ==="
    PYTHON_BIN="${PYTHON_INSTALL_DIR}/bin/python3.12"
fi

# Sanity check
"${PYTHON_BIN}" --version

# 4. Setup Python virtual environment
#    Fallback: if ensurepip still fails, create the venv without pip and
#    bootstrap pip manually via get-pip.py.
REPO_DIR="/decentral-qdina"
cd "$REPO_DIR"

if ! "${PYTHON_BIN}" -m venv venv; then
    echo "=== ensurepip failed inside venv; retrying with --without-pip ==="
    rm -rf venv
    "${PYTHON_BIN}" -m venv --without-pip venv
    source venv/bin/activate
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    python /tmp/get-pip.py
    rm -f /tmp/get-pip.py
else
    source venv/bin/activate
fi

pip install --upgrade pip
pip install -r requirements.txt

# ==============================================================================
# ROUTER SETUP
# ==============================================================================
if [ "$ROLE" == "router" ]; then
    CSV_FILE="$REPO_DIR/replicas-cloudlab.csv"
    echo "id,host,port,user,password,dbname" > "$CSV_FILE"
    for idx in $(seq 1 "$WORKER_COUNT"); do
        ip_host="10.10.1.$((10 + idx))"
        echo "${idx},${ip_host},5432,sam,tpchdb,," >> "$CSV_FILE"
    done

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

    if ! tmux has-session -t qdina 2>/dev/null; then
        tmux new-session -d -s qdina -c "$REPO_DIR" "source venv/bin/activate && bash"
        tmux send-keys -t qdina "source venv/bin/activate" C-m
    fi

    echo "Starting the Central Router in the 'qdina' tmux session (router_mode=${ROUTER_MODE})..."
    tmux send-keys -t qdina "time python3 -m router.main_router --mode drift --episodes ${EPISODES} --config replicas-cloudlab.csv --workload-dir ./workload_output --seed ${SEED} --router-mode ${ROUTER_MODE}" C-m

# ==============================================================================
# WORKER SETUP
# ==============================================================================
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

    cd "$REPO_DIR"

    if ! tmux has-session -t qdina 2>/dev/null; then
        tmux new-session -d -s qdina -c "$REPO_DIR" "source venv/bin/activate && bash"
        tmux send-keys -t qdina "source venv/bin/activate" C-m
    fi

    echo "Starting Agent Node ${NODE_ID} connecting to Router at 10.10.1.1:50051 (router_mode=${ROUTER_MODE})..."
    tmux send-keys -t qdina "python3 -m agent.main_agent --id ${NODE_ID} --mode classical --server 10.10.1.1:50051 --config replicas-cloudlab.csv --budget-mode ignore --storage-budget ${BUDGET} --router-mode ${ROUTER_MODE}" C-m
fi