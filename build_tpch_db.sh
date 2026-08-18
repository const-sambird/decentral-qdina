#!/bin/bash
set -euo pipefail

SCALE_FACTOR=""
MODE="standard"

while getopts "s:m:" opt; do
    case "${opt}" in
        s)
            SCALE_FACTOR="${OPTARG}"
            ;;
        m)
            MODE="${OPTARG}"
            ;;
        *)
            echo "Usage: sudo bash build_tpch_db.sh -s <SCALE_FACTOR> [-m <MODE>]"
            echo "Example: sudo bash build_tpch_db.sh -s 25 -m benchmark"
            exit 1
            ;;
    esac
done

if [ -z "${SCALE_FACTOR}" ]; then
    echo "Error: Scale factor (-s) is required."
    echo "Usage: sudo bash build_tpch_db.sh -s <SCALE_FACTOR> [-m <MODE>]"
    echo "Example: sudo bash build_tpch_db.sh -s 25 -m benchmark"
    exit 1
fi

THREADS=$(nproc)
TARGET_USER=${SUDO_USER:-$USER}

echo "=== [0/8] Purging existing PostgreSQL processes and residual memory ==="
systemctl stop postgresql 2>/dev/null || true
systemctl stop "postgresql@*" 2>/dev/null || true

if id "postgres" >/dev/null 2>&1; then
    pkill -9 -u postgres 2>/dev/null || true
    for shm_id in $(ipcs -m | awk '$3=="postgres" {print $2}'); do
        ipcrm -m "$shm_id" 2>/dev/null || true
    done
    for sem_id in $(ipcs -s | awk '$3=="postgres" {print $2}'); do
        ipcrm -s "$sem_id" 2>/dev/null || true
    done
fi

echo "=== [1/8] Checking and mounting /data partition ==="
mkdir -p /data

if ! mountpoint -q /data; then
    sed -i '/\/data/d' /etc/fstab
    
    # 1. Utilisation prioritaire de l'outil CloudLab pour monter n'importe quel disque additionnel
    if [ -x /usr/local/etc/emulab/mkextrafs.pl ]; then
        /usr/local/etc/emulab/mkextrafs.pl -f /data || true
    fi

    # 2. Si mkextrafs n'a pas suffi, on cherche un NVMe (Spécifique à certains clusters comme Utah)
    if ! mountpoint -q /data; then
        # Le '|| true' empêche le script de crasher si grep ne trouve rien
        NVME_DEV=$(lsblk -dpno NAME | grep nvme | head -n 1 || true)

        if [ -n "$NVME_DEV" ]; then
            NVME_PART="${NVME_DEV}p4"
            if [ -b "$NVME_PART" ]; then
                mkfs.ext4 -F "$NVME_PART"
                echo "$NVME_PART /data ext4 defaults 0 0" >> /etc/fstab
                systemctl daemon-reload
                mount /data
            fi
        else
            echo "Warning: No NVMe device detected. Falling back to default root storage."
        fi
    fi
fi

AVAIL_GB=$(df -BG /data | awk 'NR==2 {print $4}' | tr -d 'G')
if [ "$AVAIL_GB" -lt 50 ]; then
    echo "Warning: /data has only ${AVAIL_GB}GB available. Generating a large Scale Factor might fail."
fi

chown -R "${TARGET_USER}:" /data
chmod 755 /data
cd /data

echo "=== [2/8] Installing PostgreSQL 17 and HypoPG ==="
rm -f /etc/apt/sources.list.d/pgdg.list /etc/apt/sources.list.d/pgdg.sources /etc/apt/keyrings/postgresql.gpg

apt update -qq
apt install -y -qq curl ca-certificates gnupg git build-essential gcc make postgresql-common psmisc

if [ -f /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh ]; then
    /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y
else
    install -d /etc/apt/keyrings
    curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor --yes -o /etc/apt/keyrings/postgresql.gpg
    echo "deb [signed-by=/etc/apt/keyrings/postgresql.gpg] http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list
fi

apt update -qq
apt install -y -qq postgresql-17 postgresql-contrib-17 postgresql-server-dev-17

rm -rf /data/hypopg
git clone https://github.com/HypoPG/hypopg.git /data/hypopg
cd /data/hypopg
git checkout 1.4.1 -q
make -s
make -s install

apt-get update -qq && apt-get install -y -qq locales
locale-gen en_US.UTF-8
update-locale LANG=en_US.UTF-8

echo "=== [3/8] Configuring PostgreSQL storage and network access ==="
systemctl stop postgresql 2>/dev/null || true
if id "postgres" >/dev/null 2>&1; then
    pkill -9 -u postgres 2>/dev/null || true
fi

mkdir -p /data/postgresql/17/main /var/run/postgresql
rm -rf /data/postgresql/17/main/* /var/lib/postgresql/17/main/* 2>/dev/null || true
chown -R postgres:postgres /data/postgresql /var/run/postgresql
chmod 700 /data/postgresql/17/main
chmod 2777 /var/run/postgresql

# Initialisation du cluster propre sur /data
sudo -u postgres /usr/lib/postgresql/17/bin/initdb -D /data/postgresql/17/main --locale=en_US.UTF-8 --encoding=UTF8

PG_CONF="/etc/postgresql/17/main/postgresql.conf"
PG_HBA="/etc/postgresql/17/main/pg_hba.conf"

sed -i "s|data_directory = .*|data_directory = '/data/postgresql/17/main'|g" "${PG_CONF}"
sed -i '/host.*sam.*127\.0\.0\.1/d' "${PG_HBA}"
sed -i '/host.*tpchdb.*sam.*10\.10\.1\.0/d' "${PG_HBA}"
sed -i '1i host    all             sam             127.0.0.1/32            trust' "${PG_HBA}"
sed -i '$a host    tpchdb          sam             10.10.1.0/24            trust' "${PG_HBA}"
sed -i "s/#listen_addresses = 'localhost'/listen_addresses = '*'/g" "${PG_CONF}"
sed -i "s/listen_addresses = 'localhost'/listen_addresses = '*'/g" "${PG_CONF}"

systemctl restart postgresql

until sudo -u postgres pg_isready -q; do
    sleep 1
done

sudo -u postgres psql -c "ALTER SYSTEM SET shared_buffers = '16GB';"
sudo -u postgres psql -c "ALTER SYSTEM SET maintenance_work_mem = '16GB';"
sudo -u postgres psql -c "ALTER SYSTEM SET max_wal_size = '40GB';"
sudo -u postgres psql -c "ALTER SYSTEM SET checkpoint_completion_target = 0.9;"
sudo -u postgres psql -c "ALTER SYSTEM SET wal_level = minimal;"
sudo -u postgres psql -c "ALTER SYSTEM SET max_wal_senders = 0;"
sudo -u postgres psql -c "ALTER SYSTEM SET synchronous_commit = off;"
systemctl restart postgresql

until sudo -u postgres pg_isready -q; do
    sleep 1
done

sudo -u postgres psql -c "DO \$\$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'sam') THEN CREATE USER sam SUPERUSER; ELSE ALTER USER sam SUPERUSER; END IF; END \$\$;"
sudo -u postgres psql -c "DROP DATABASE IF EXISTS tpchdb;"
sudo -u postgres psql -c "CREATE DATABASE tpchdb OWNER sam ENCODING 'UTF8' LC_COLLATE 'en_US.UTF-8' LC_CTYPE 'en_US.UTF-8';"
sudo -u postgres psql -d tpchdb -c "CREATE EXTENSION IF NOT EXISTS hypopg;"


if [ "$MODE" != "benchmark" ]; then
    echo "=== [4/8] Compiling dbgen and generating data (Scale Factor: ${SCALE_FACTOR}) ==="
    rm -rf /data/pg-tpch-dbgen
    git clone https://github.com/joaomcosta/pg-tpch-dbgen.git /data/pg-tpch-dbgen
    cd /data/pg-tpch-dbgen/dbgen

    cp makefile.suite Makefile
    sed -i 's/^CC[[:space:]]*=.*/CC      = gcc/' Makefile
    sed -i 's/^DATABASE[[:space:]]*=.*/DATABASE= POSTGRESQL/' Makefile
    sed -i 's/^MACHINE[[:space:]]*=.*/MACHINE = LINUX/' Makefile
    sed -i 's/^WORKLOAD[[:space:]]*=.*/WORKLOAD = TPCH/' Makefile
    make dbgen -s

    for i in $(seq 1 "${THREADS}"); do
        ./dbgen -s "${SCALE_FACTOR}" -C "${THREADS}" -S "$i" -f &
    done
    wait

    for f in *.tbl*; do
        sed -i 's/|$//' "$f" &
    done
    wait

    echo "=== [5/8] Creating schema (dss.ddl) and importing data ==="
    psql -h 127.0.0.1 -U sam -d tpchdb -f dss.ddl

    psql -h 127.0.0.1 -U sam -d tpchdb -c "\copy region FROM 'region.tbl' WITH (FORMAT csv, DELIMITER '|');"
    rm -f region.tbl
    psql -h 127.0.0.1 -U sam -d tpchdb -c "\copy nation FROM 'nation.tbl' WITH (FORMAT csv, DELIMITER '|');"
    rm -f nation.tbl

    for tbl in supplier customer part partsupp orders lineitem; do
        for f in ${tbl}.tbl.*; do
            psql -h 127.0.0.1 -U sam -d tpchdb -c "\copy ${tbl} FROM '$f' WITH (FORMAT csv, DELIMITER '|');"
            rm -f "$f"
        done
    done

    echo "=== [6/8] Creating primary keys, foreign keys, indexes and running ANALYZE ==="
    psql -h 127.0.0.1 -U sam -d tpchdb -f ../dss/tpch-pkeys.sql
    psql -h 127.0.0.1 -U sam -d tpchdb -f ../dss/tpch-fkey.sql
    psql -h 127.0.0.1 -U sam -d tpchdb -f ../dss/tpch-index.sql
    psql -h 127.0.0.1 -U sam -d tpchdb -c "ANALYZE;"
else
    echo "=== [4/8] Benchmark Mode: Skipping dbgen compilation and data generation ==="
    echo "=== [5/8] Benchmark Mode: Skipping schema creation and data import ==="
    echo "=== [6/8] Benchmark Mode: Skipping constraints and index creation ==="
fi

echo "=== [7/8] Cleaning up temporary files and resetting server configuration ==="
rm -rf /data/hypopg /data/pg-tpch-dbgen
apt clean
rm -rf /tmp/* /var/tmp/*

sudo -u postgres psql -c "ALTER USER sam NOSUPERUSER;"
sudo -u postgres psql -c "ALTER SYSTEM RESET ALL;"
systemctl restart postgresql
until sudo -u postgres pg_isready -q; do
    sleep 1
done

echo "=== [8/8] Database table sizes ==="
psql -h 127.0.0.1 -U sam -d tpchdb -c "
SELECT 
    relname AS table_name,
    pg_size_pretty(pg_total_relation_size(relid)) AS total_size,
    pg_size_pretty(pg_relation_size(relid)) AS data_size,
    pg_size_pretty(pg_indexes_size(relid)) AS index_size
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC;
"

echo "=========================================================="
echo " Setup complete! tpchdb is ready for SF${SCALE_FACTOR} (Mode: ${MODE})."
echo "=========================================================="