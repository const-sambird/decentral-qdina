#!/bin/bash
exec > >(tee -a /var/log/cloudlab_startup.log) 2>&1
set -ex
set -o pipefail

ROLE=$1
WORKER_COUNT=${2:-2}
SF=${3:-10}
BUDGET=${4:-5000000000}
NODE_ID=${5:-1}
EPISODES=${6:-100}
SEED=${7:-100}
ROUTER_MODE=${8:-learned}

export DEBIAN_FRONTEND=noninteractive
export HOME="${HOME:-/root}"
export USER="${USER:-root}"

retry() {
    local max_attempts=$1
    local delay=$2
    shift 2
    local attempt=1
    while true; do
        if "$@"; then
            return 0
        fi
        if [ "$attempt" -ge "$max_attempts" ]; then
            echo "=== Command failed after ${max_attempts} attempts: $* ==="
            return 1
        fi
        echo "=== Attempt ${attempt}/${max_attempts} failed; retrying in ${delay}s: $* ==="
        attempt=$((attempt + 1))
        sleep "$delay"
    done
}

git_clone_retry() {
    local url=$1
    local dest=$2
    local attempt=1
    while true; do
        rm -rf "$dest"
        if git clone "$url" "$dest"; then
            return 0
        fi
        if [ "$attempt" -ge 5 ]; then
            echo "=== git clone failed after 5 attempts: ${url} ==="
            return 1
        fi
        echo "=== git clone attempt ${attempt}/5 failed; retrying in 10s ==="
        attempt=$((attempt + 1))
        sleep 10
    done
}

echo 'Acquire::ForceIPv4 "true";' | tee /etc/apt/apt.conf.d/99force-ipv4
echo 'Acquire::Retries "5";'       | tee /etc/apt/apt.conf.d/99retries

# 1. Nettoyage locks APT
sleep 5
systemctl stop unattended-upgrades.service apt-daily.service apt-daily-upgrade.service 2>/dev/null || true
pkill -9 -f unattended-upg 2>/dev/null || true
pkill -9 -f apt-get 2>/dev/null || true
rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/lib/apt/lists/lock /var/cache/apt/archives/lock 2>/dev/null || true
dpkg --configure -a 2>/dev/null || true

# 2. Outils système (dépôts Ubuntu uniquement, avec retries)
if [ -f /etc/apt/sources.list ]; then
    sed -i 's|us.archive.ubuntu.com|fr.archive.ubuntu.com|g' /etc/apt/sources.list || true
fi
retry 5 10 apt-get update -qq
retry 5 10 apt-get install -y -qq \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" \
    git tmux netcat-openbsd curl wget ca-certificates xz-utils \
    build-essential gcc make psmisc \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libncursesw5-dev tk-dev libffi-dev liblzma-dev

# 3. Python 3.12 (toujours invoqué par son chemin réel, JAMAIS via un symlink)
PYTHON_VERSION="3.12.7"
PYTHON_RELEASE="20241016"
PYTHON_INSTALL_DIR="/opt/python3.12"
PYTHON_BIN="${PYTHON_INSTALL_DIR}/bin/python3.12"

python_is_healthy() {
    local bin=$1
    [ -x "$bin" ] && "$bin" -c "import os, ssl, sqlite3, zlib, ctypes" 2>/dev/null
}

if python_is_healthy "$PYTHON_BIN"; then
    echo "=== Existing Python 3.12 at ${PYTHON_INSTALL_DIR} is healthy ==="
else
    echo "=== No healthy Python 3.12 found, (re)installing ==="
    rm -rf "$PYTHON_INSTALL_DIR"
    rm -f /usr/local/bin/python3.12 /usr/local/bin/pip3.12

    ARCH=$(uname -m)
    TARBALL="cpython-${PYTHON_VERSION}+${PYTHON_RELEASE}-${ARCH}-unknown-linux-gnu-install_only.tar.gz"
    TARBALL_ENC="cpython-${PYTHON_VERSION}%2B${PYTHON_RELEASE}-${ARCH}-unknown-linux-gnu-install_only.tar.gz"

    URLS=(
        "https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_RELEASE}/${TARBALL_ENC}"
        "https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_RELEASE}/${TARBALL}"
        "https://github.com/indygreg/python-build-standalone/releases/download/${PYTHON_RELEASE}/${TARBALL_ENC}"
    )

    DOWNLOADED=0
    for url in "${URLS[@]}"; do
        echo "Trying ${url}..."
        rm -f /tmp/python3.12.tar.gz
        if curl -4 -fL --retry 5 --retry-delay 5 --connect-timeout 20 -o /tmp/python3.12.tar.gz "${url}"; then
            DOWNLOADED=1
            break
        fi
    done

    if [ "${DOWNLOADED}" -eq 1 ]; then
        mkdir -p "$PYTHON_INSTALL_DIR"
        if tar -xzf /tmp/python3.12.tar.gz -C "$PYTHON_INSTALL_DIR" --strip-components=1; then
            rm -f /tmp/python3.12.tar.gz
            if ! python_is_healthy "$PYTHON_BIN"; then
                echo "=== python-build-standalone install failed health check; falling back to pyenv ==="
                rm -rf "$PYTHON_INSTALL_DIR"
                DOWNLOADED=0
            else
                # Wrapper script (PAS de symlink!) pour préserver argv[0] et sys.prefix=/opt/python3.12
                cat > /usr/local/bin/python3.12 << 'EOF'
#!/bin/sh
exec /opt/python3.12/bin/python3.12 "$@"
EOF
                chmod +x /usr/local/bin/python3.12
                cat > /usr/local/bin/pip3.12 << 'EOF'
#!/bin/sh
exec /opt/python3.12/bin/pip3.12 "$@"
EOF
                chmod +x /usr/local/bin/pip3.12
                # On garde PYTHON_BIN="/opt/python3.12/bin/python3.12" pour que venv écrive le bon home dans pyvenv.cfg
            fi
        else
            rm -f /tmp/python3.12.tar.gz
            rm -rf "$PYTHON_INSTALL_DIR"
            DOWNLOADED=0
        fi
    fi

    if [ "${DOWNLOADED}" -eq 0 ] || ! python_is_healthy "$PYTHON_BIN"; then
        echo "=== Falling back to pyenv (compilation, ~10 min) ==="
        export PYENV_ROOT="/root/.pyenv"
        if [ ! -x "$PYENV_ROOT/bin/pyenv" ]; then
            rm -rf "$PYENV_ROOT"
            curl -4 -fsSL --retry 5 --retry-delay 5 https://pyenv.run | bash
        fi
        export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
        "$PYENV_ROOT/bin/pyenv" install -s "${PYTHON_VERSION}"
        "$PYENV_ROOT/bin/pyenv" global "${PYTHON_VERSION}"
        PYTHON_BIN="${PYENV_ROOT}/versions/${PYTHON_VERSION}/bin/python3.12"

        cat > /etc/profile.d/pyenv.sh << 'EOF'
export PYENV_ROOT="/root/.pyenv"
export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"
EOF
        chmod 644 /etc/profile.d/pyenv.sh
    fi
fi

# Sanity check ultime sur le binaire réel
"$PYTHON_BIN" -c "import os, sys, ssl; print('Python 3.12 OK:', sys.version, sys.prefix)"

# 4. Venv (avec fallback --without-pip + get-pip.py)
REPO_DIR="/decentral-qdina"
cd "$REPO_DIR"
rm -rf venv

if ! "$PYTHON_BIN" -m venv venv; then
    echo "=== ensurepip failed; using --without-pip + get-pip.py fallback ==="
    rm -rf venv
    "$PYTHON_BIN" -m venv --without-pip venv
    curl -4 -fsSL --retry 5 --retry-delay 5 https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    "$REPO_DIR/venv/bin/python" /tmp/get-pip.py
    rm -f /tmp/get-pip.py
fi

VENV_PYTHON="$REPO_DIR/venv/bin/python"
VENV_PIP="$REPO_DIR/venv/bin/pip"

# Vérification venv
"$VENV_PYTHON" -c "import os, sys, ssl; print('Venv OK:', sys.executable, sys.prefix)"
retry 5 10 "$VENV_PIP" install --retries 5 --timeout 60 --upgrade pip
retry 5 10 "$VENV_PIP" install --retries 5 --timeout 60 -r requirements.txt

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
    git_clone_retry https://github.com/const-sambird/qdina-bench.git "$BENCH_DIR"
    chmod -R 777 "$BENCH_DIR"
    cp "$CSV_FILE" "$BENCH_DIR/replicas.csv"

    git_clone_retry https://github.com/gregrahn/tpch-kit.git "$BENCH_DIR/tpc-h"
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
    tmux kill-session -t qdina 2>/dev/null || true
    tmux new-session -d -s qdina -c "$REPO_DIR" bash
    sleep 1
    tmux send-keys -t qdina "cd ${REPO_DIR} && source venv/bin/activate && time ${VENV_PYTHON} -m router.main_router --mode drift --episodes ${EPISODES} --config replicas-cloudlab.csv --workload-dir ./workload_output --seed ${SEED} --router-mode ${ROUTER_MODE}" C-m

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
        until sudo -u postgres pg_isready -q; do
            sleep 1
        done
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
    while ! nc -z -w 2 10.10.1.1 50051; do
        sleep 2
    done

    cd "$REPO_DIR"
    tmux kill-session -t qdina 2>/dev/null || true
    tmux new-session -d -s qdina -c "$REPO_DIR" bash
    sleep 1
    tmux send-keys -t qdina "cd ${REPO_DIR} && source venv/bin/activate && ${VENV_PYTHON} -m agent.main_agent --id ${NODE_ID} --mode classical --server 10.10.1.1:50051 --config replicas-cloudlab.csv --budget-mode ignore --storage-budget ${BUDGET} --router-mode ${ROUTER_MODE}" C-m
fi
