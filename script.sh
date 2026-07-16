#!/bin/bash

REPO_DIR="decentral-qdina"

if [ ! -d "$REPO_DIR" ]; then
    sudo sed -i '1i host    all             sam             127.0.0.1/32            trust' /etc/postgresql/17/main/pg_hba.conf && sudo systemctl restart postgresql
    sudo sed -i '$a host    tpchdb          sam             10.10.1.0/24            trust' /etc/postgresql/17/main/pg_hba.conf    
    sudo sed -i '/host.*0.0.0.0\/0/d' /etc/postgresql/17/main/pg_hba.conf
    sudo sed -i '/host.*::\/0/d' /etc/postgresql/17/main/pg_hba.conf
    sudo systemctl restart postgresql
    echo "Cloning repository..."
    git clone https://github.com/const-sambird/decentral-qdina.git
    cd "$REPO_DIR" || exit
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else 
    echo "Repository already exists. Updating..."
    cd "$REPO_DIR" || exit
    git pull
    source venv/bin/activate
fi

if ! tmux has-session -t qdina 2>/dev/null; then
    tmux new-session -d -s qdina "source venv/bin/activate && bash"
    tmux send-keys -t qdina "source venv/bin/activate" C-m
fi

if [ "$1" == "--node" ] && [ "$2" == "router" ]; then
    echo "Starting the Central Router in the 'qdina' tmux session..."
    tmux send-keys -t qdina "time python3 -m router.main_router --mode drift --episodes 100 --config replicas-cloudlab.csv --workload-dir ./workload_output" C-m
elif [ "$1" == "--node" ] && [ "$2" -ge 1 ] && [ "$2" -le 6 ] 2>/dev/null; then
    ROUTER_IP="10.10.1.1"
    echo "Starting Agent Node $2 connecting to Router at ${ROUTER_IP}:50051..."
    echo "python3 -m agent.main_agent --id $2 --mode quantum --server ${ROUTER_IP}:50051 --config replicas-cloudlab.csv"
    tmux send-keys -t qdina "python3 -m agent.main_agent --id $2 --mode quantum --server ${ROUTER_IP}:50051 --config replicas-cloudlab.csv" C-m
else
    echo "Usage for Router : ./start.sh --node router"
    echo "Usage for Agent  : ./start.sh --node <1-6> --router-ip <IP_ADDRESS>"
    exit 1
fi

echo "The process is running in the background."
echo "Use 'tmux attach -t qdina' to view logs."