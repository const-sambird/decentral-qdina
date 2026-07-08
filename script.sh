#!/bin/bash

REPO_DIR="decentral-qdina"

if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning repository..."
    git clone https://github.com/const-sambird/decentral-qdina.git
    cd "$REPO_DIR" || exit
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    python3 common/qgen.py
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
    if [ "$3" == "--router" ] && [ -n "$4" ]; then
        ROUTER_IP="$4"
        echo "Starting Agent Node $2 connecting to Router at ${ROUTER_IP}:50051..."
        echo "python3 -m agent.main_agent --id $2 --mode quantum --server ${ROUTER_IP}:50051 --config replicas-cloudlab.csv --workload-dir ./workload_output"
        tmux send-keys -t qdina "python3 -m agent.main_agent --id $2 --mode quantum --server ${ROUTER_IP}:50051 --config replicas-cloudlab.csv --workload-dir ./workload_output" C-m
    else
        echo "Error: Missing router IP address for the worker agent."
        exit 1
    fi
else
    echo "Usage for Router : ./start.sh --node router"
    echo "Usage for Agent  : ./start.sh --node <1-6> --router-ip <IP_ADDRESS>"
    exit 1
fi

echo "The process is running in the background."
echo "Use 'tmux attach -t qdina' to view logs."