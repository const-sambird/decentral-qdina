# -*- coding: utf-8 -*-
"""CloudLab profile for decentralized qDINA Benchmark experiments."""

import geni.portal as portal
import geni.rspec.pg as pg
import geni.rspec.igext as ig

pc = portal.Context()

# Default values for experiments
DEFAULT_INDEX_CONFIGURATION = '0,p_name'
DEFAULT_ROUTEING_TABLE = '0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0'
# Connection format: port, dbname, user, (empty password), (empty)
DEFAULT_REPLICA_STRING = '5432,tpchdb,sam,,'

pc.defineParameter('nodes', 'Number of database replicas', portal.ParameterType.INTEGER, 2)

pc.defineParameter(
    "scaleFactor",
    "TPC-H Database Scale Factor (GB)",
    portal.ParameterType.INTEGER,
    10,
    legalValues=[
        (1, "1 GB (Scale Factor 1)"),
        (10, "10 GB (Scale Factor 10)"),
        (30, "30 GB (Scale Factor 30)"),
        (100, "100 GB (Scale Factor 100)"),
        (300, "300 GB (Scale Factor 300)"),
        (1000, "1000 GB / 1 TB (Scale Factor 1000)"),
    ],
    longDescription="Select the TPC-H database scale factor."
)

pc.defineParameter('config', 'Index configuration', portal.ParameterType.STRING, DEFAULT_INDEX_CONFIGURATION, longDescription='Separate each entry with a single space character.')
pc.defineParameter('routes', 'Routeing table', portal.ParameterType.STRING, DEFAULT_ROUTEING_TABLE)
pc.defineParameter('partial', 'Training partition templates', portal.ParameterType.STRING, '', longDescription='If less than the full set of templates was used to train, report that runtime separately')
pc.defineParameter('replica_str', 'Replica connection string (after ID and Host)', portal.ParameterType.STRING, DEFAULT_REPLICA_STRING)

params = pc.bindParameters()

if params.nodes < 1:
    pc.reportError(portal.ParameterError('Invalid number of database replicas! must have at least 1'))

parsed_config = params.config.replace(' ', '\n')

pc.verifyParameters()

rspec = pg.Request()

# Profile description
tour = ig.Tour()
tour.Description(ig.Tour.TEXT, "Profile for decentralized qDINA Benchmark. Sets up the network topology and PostgreSQL configurations for N nodes on c6620. Receives dynamic configuration parameters from the API.")
rspec.addTour(tour)

CLEAN_UBUNTU_IMAGE = "urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU22-64-STD"

lan = rspec.Link("qdina_private_lan")
lan.link_multiplexing = True
lan.best_effort = True

# Central router node
router = rspec.RawPC("router")
router.hardware_type = "c6620" # Strict constraint for your project
router.disk_image = CLEAN_UBUNTU_IMAGE

iface_router = router.addInterface("if_router")
iface_router.addAddress(pg.IPv4Address("10.10.1.1", "255.255.255.0"))
lan.addInterface(iface_router)

# Building the replicas.csv file with private network IPs
replicas_csv_content = ""

for i in range(1, params.nodes + 1):
    worker_ip = "10.10.1.{0}".format(10 + i)
    if i == params.nodes:
        # No newline for the very last entry
        replicas_csv_content += "{0},{1},{2}".format(i, worker_ip, params.replica_str)
    else:
        # Keep newline for others
        replicas_csv_content += "{0},{1},{2}\\n".format(i, worker_ip, params.replica_str)

# Commands to inject API parameters into physical files before cloning the repo
# Using echo -ne prevents Bash from forcing a default newline at the end of the file
setup_files_cmd = (
    "sudo mkdir -p /qdina-bench && "
    "echo -ne '{0}' | sudo tee /qdina-bench/replicas.csv > /dev/null && "
    "echo -ne '{1}' | sudo tee /qdina-bench/routes.csv > /dev/null && "
    "echo -ne '{2}' | sudo tee /qdina-bench/config.csv > /dev/null".format(replicas_csv_content, params.routes, parsed_config)
)

if params.partial != '':
    setup_files_cmd += " && echo -ne '{0}' | sudo tee /qdina-bench/partial.csv > /dev/null".format(params.partial)

# Execution of the benchmark installation script (Dynamic injection of the scaleFactor)
router_cmd = "{0} && sudo rm -rf /decentral-qdina && sudo git clone https://github.com/const-sambird/decentral-qdina.git /decentral-qdina && sudo bash /decentral-qdina/setup_benchmark.sh router {1} {2} 0".format(setup_files_cmd, params.nodes, params.scaleFactor)

router.addService(pg.Execute(shell="bash", command=router_cmd))

# Worker nodes (PostgreSQL replicas)
for i in range(1, params.nodes + 1):
    worker_name = "replica_{0}".format(i)
    worker = rspec.RawPC(worker_name)
    worker.hardware_type = "c6620" # Strict constraint for your project
    worker.disk_image = CLEAN_UBUNTU_IMAGE

    iface_worker = worker.addInterface("if_{0}".format(worker_name))
    worker_ip = "10.10.1.{0}".format(10 + i)
    iface_worker.addAddress(pg.IPv4Address(worker_ip, "255.255.255.0"))
    lan.addInterface(iface_worker)

    # Execution of the benchmark installation script (Dynamic injection of the scaleFactor)
    worker_cmd = "sudo rm -rf /decentral-qdina && sudo git clone https://github.com/const-sambird/decentral-qdina.git /decentral-qdina && sudo bash /decentral-qdina/setup_benchmark.sh worker {0} {1} {2}".format(params.nodes, params.scaleFactor, i)
    
    worker.addService(pg.Execute(shell="bash", command=worker_cmd))

pc.printRequestRSpec(rspec)