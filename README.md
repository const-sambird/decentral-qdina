# decentral-qdina

## Project Structure

```text
decentral-qdina/
│
├── common/                             # Shared files and pure utility functions
│   ├── __init__.py
│   ├── replay_memory.py                # Replay buffer management for reinforcement learning
│   ├── spsa_opt.py                     # Local quantum circuit optimizer
│   ├── qiskit_spsa.py                  # Official Qiskit SPSA alternative
│   ├── util.py                         # Utility functions (SQL cleaning, extractions)
│   ├── workload_manager.py             # TPC-H query batch manager
│   ├── tpch_generator.py               # TPC-H query generator
│   ├── tpcds_generator.py              # TPC-DS query generator
│   ├── qgen.py                         # Quantum circuit and configuration generator
│   └── workload_manager.py
│
├── router/                             # Central Router global logic (Master)
│   ├── __init__.py
│   ├── DQN.py                          # Classical linear AI model for query routing
│   ├── router_agent.py                 # Router DQN training loop (Extracted from learner.py)
│   ├── environment_global.py           # Global state management and Jain Fairness Index computation
│   ├── main_router.py                  # Main router entry point
│   └── network_server.py               # Network server (FastAPI/Sockets) to orchestrate workers
│
├── agent/                              # Local quantum indexing logic (Worker)
│   ├── __init__.py
│   ├── qnn.py                          # Quantum circuit definition (QuantumDQN)
│   ├── encoding.py                     # State encoding (Angle, Basis, etc.) for the QNN
│   ├── database.py                     # Database connection and PostgreSQL management
│   ├── cost_estimator.py               # HypoPG cost estimation for virtual index setups
│   ├── environment_local.py            # Indexing environment and original DINA reward function
│   ├── qia_environment.py              # Alternative quantum indexing environment variant
│   ├── main_agent.py                   # Main agent entry point
│   └── network_client.py               # Network client to receive workload and return cost metrics
│
├── data/                               # Configuration files and SQL data
│   ├── templates/                      # Structured SQL query template directories
│   │   ├── tpc-ds/                     # TPC-DS raw benchmark query templates
│   │   └── tpc-h/                      # TPC-H raw benchmark query templates
│   └── templates.txt                   # Combined legacy TPC-H query templates
│
├── protos/
|   ├── qdina.proto
|   ├── qdina_pb2_grpc.py
│   └── qdina_pb2.py
|
├── tpc-h/
│   └── dbgen/
|
├── replicas.csv
├── requirements.txt                    # Complete project dependencies
└── README.md                           # General project documentation
```
