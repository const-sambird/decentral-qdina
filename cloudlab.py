# -*- coding: utf-8 -*-
"""CloudLab profile for automated decentralized qDINA experiments."""

import geni.portal as portal
import geni.rspec.pg as pg
import geni.rspec.igext as ig

pc = portal.Context()

# Add native CloudLab node selector (left completely blank by default so CloudLab picks available nodes)
pc.defineParameter(
    "hwType",
    "Hardware Node Type",
    portal.ParameterType.NODETYPE,
    "",
    longDescription="Leave blank to let CloudLab automatically assign available nodes from the cluster."
)

pc.defineParameter(
    "workerCount",
    "Number of Database Replicas",
    portal.ParameterType.INTEGER,
    4,
    longDescription="Number of worker database replica nodes to deploy (1 to N)."
)

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
    longDescription="Select the TPC-H database scale factor to generate on NVMe /data storage."
)

pc.defineParameter(
    "indexBudgetPercent",
    "Index Storage Budget Percentage",
    portal.ParameterType.INTEGER,
    50,
    longDescription="Enter the percentage of the database size allocated to the agent index storage budget (can exceed 100%)."
)

pc.defineParameter(
    "episodes",
    "Number of Training Episodes",
    portal.ParameterType.INTEGER,
    100,
    longDescription="Total number of episodes for the router training loop."
)

pc.defineParameter(
    "seed",
    "Random Seed",
    portal.ParameterType.INTEGER,
    100,
    longDescription="Random seed for reproducibility."
)

pc.defineParameter(
    "routerMode",
    "Router Strategy",
    portal.ParameterType.STRING,
    "learned",
    legalValues=[
        ("learned",   "Learned (DQN) - default"),
        ("static",    "Static (no routing changes) - ablation"),
        ("heuristic", "Heuristic (DiversityClusterDB) - ablation"),
    ],
    longDescription="Select the routing strategy: "
                    "'learned' uses the DQN router (default), "
                    "'static' disables routing entirely (ablation study), "
                    "'heuristic' uses the DiversityClusterDB heuristic (ablation study)."
)

params = pc.bindParameters()

if params.workerCount < 1:
    pc.reportError(portal.ParameterError("workerCount must be at least 1."))

rspec = pg.Request()

# Profile description and instructions
tour = ig.Tour()
tour.Description(ig.Tour.TEXT, "Profile for automated decentralized qDINA experiments. It provisions a central router and multiple PostgreSQL replica workers. CloudLab manages node allocation dynamically based on availability.")
tour.Instructions(ig.Tour.TEXT, "Wait for the startup scripts to finish. You can monitor the progress by connecting to the nodes via SSH and running: sudo tail -f /var/log/cloudlab_startup.log . Once finished, access the agent loops via: sudo tmux attach -t qdina")
rspec.addTour(tour)

STORAGE_BUDGET = int((params.scaleFactor * 10**9) * (params.indexBudgetPercent / 100.0))
CLEAN_UBUNTU_IMAGE = "urn:publicid:IDN+emulab.net+image+emulab-ops//UBUNTU22-64-STD"

lan = rspec.Link("qdina_private_lan")
lan.link_multiplexing = True
lan.best_effort = True

# Central router node (Hardware type is assigned only if explicitly specified)
router = rspec.RawPC("router")
if params.hwType != "":
    router.hardware_type = params.hwType
router.disk_image = CLEAN_UBUNTU_IMAGE

iface_router = router.addInterface("if_router")
iface_router.addAddress(pg.IPv4Address("10.10.1.1", "255.255.255.0"))
lan.addInterface(iface_router)

# Arguments: role, worker_count, scale_factor, storage_budget, node_id, episodes, seed, router_mode
router_cmd = "sudo rm -rf /decentral-qdina && sudo git clone https://github.com/const-sambird/decentral-qdina.git /decentral-qdina && sudo bash /decentral-qdina/setup_cloudlab.sh router {0} {1} {2} 0 {3} {4} {5}".format(
    params.workerCount, params.scaleFactor, STORAGE_BUDGET, params.episodes, params.seed, params.routerMode
)
router.addService(pg.Execute(shell="sh", command=router_cmd))

# Worker nodes (PostgreSQL replicas and agents)
for i in range(1, params.workerCount + 1):
    worker_name = "replica_{0}".format(i)
    worker = rspec.RawPC(worker_name)
    if params.hwType != "":
        worker.hardware_type = params.hwType
    worker.disk_image = CLEAN_UBUNTU_IMAGE

    iface_worker = worker.addInterface("if_{0}".format(worker_name))
    worker_ip = "10.10.1.{0}".format(10 + i)
    iface_worker.addAddress(pg.IPv4Address(worker_ip, "255.255.255.0"))
    lan.addInterface(iface_worker)

    # Arguments: role, worker_count, scale_factor, storage_budget, node_id
    worker_cmd = "sudo rm -rf /decentral-qdina && sudo git clone https://github.com/const-sambird/decentral-qdina.git /decentral-qdina && sudo bash /decentral-qdina/setup_cloudlab.sh worker {0} {1} {2} {3}".format(
        params.workerCount, params.scaleFactor, STORAGE_BUDGET, i
    )
    worker.addService(pg.Execute(shell="sh", command=worker_cmd))

pc.printRequestRSpec(rspec)