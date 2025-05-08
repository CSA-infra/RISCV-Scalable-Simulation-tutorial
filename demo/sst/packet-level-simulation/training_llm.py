import sys
import argparse
import sst
import os
import math

from sst.merlin.base import *
from sst.merlin.endpoint import *
from sst.merlin.interface import *
from sst.merlin.topology import *
from sst.ember import *


params = {
    # in GB/s
    "link_bw" : 128,

    "input_buf_size" : "4MB",
    "output_buf_size" : "4MB",

    "flit_size" : "256B",

    "link_lat" : "10ns",
    "input_latency" : "10ns",
    "output_latency" : "10ns",
    "host_link_latency" : "100ns",

    "num_vns" : 2,
    "width": 2,

    "xbar_arb" : "merlin.xbar_arb_lru"
}

llm_config_default= os.path.join(os.path.dirname(os.path.realpath(__file__)), "small_config.json")

parser = argparse.ArgumentParser(
        prog=f'sst {__file__} --',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--tp", type=int, help="Tensor Parallelism level", default=8)
parser.add_argument("--pp", type=int, help="Data Parallelism level", default=1)
parser.add_argument("--dp", type=int, help="Pipeline Parallelism level", default=1)
parser.add_argument("--batch_size", type=int, help="Number of sequence processed in parallel", default=32)
parser.add_argument("--sequence_len", type=int, help="Number of token per sequence", default=8192)
parser.add_argument("--n_step", type=int, help="Number of steps", default=128)
parser.add_argument("--llm_config", type=str, help="Configuration file of the Large Language Model", default=llm_config_default)
parser.add_argument("--verbose", type=int, help="Set verbosity", default=0)
parser.add_argument("--log", type=str, help="Enable motif logger", action='store', nargs='?', const="logger")
parser.add_argument("--stats", type=str, help="write statistics, argument changes the filename", nargs="?", const="-")
parser.add_argument("--topology", type=str, help="Network topology", default="single",
                    choices=["single", "dragonfly", "fattree"] )
args = parser.parse_args()

assert os.path.exists(args.llm_config), "LLM config file does not exist!"

num_ranks = args.tp * args.pp * args.dp
topology = args.topology.lower()

if args.stats:
    enableStats = True
    sst.setStatisticLoadLevel(10)

    fname = args.stats
    if fname.endswith(".csv"):
        sst.setStatisticOutput("sst.statOutputCSV",
                               {   "filepath" : fname,
                                "separator" : ";"
                                } )
    else:
        sst.setStatisticOutput("sst.statOutputConsole")
else:
    enableStats = False


# Network topology definition start
PlatformDefinition.setCurrentPlatform("firefly-defaults")

### set up the endpoint
networkif = ReorderLinkControl()
networkif.link_bw = str(params["link_bw"]) + "GB/s"
networkif.input_buf_size = params["input_buf_size"]
networkif.output_buf_size = params["output_buf_size"]

assert (num_ranks == args.tp or num_ranks == args.pp or num_ranks == args.dp or
       (args.dp > 1 and args.pp > 1 and args.tp > 1)), "Only 1D and 3D parallelism are supported"

ep = EmberMPIJob(0,num_ranks)

ep.ember.verbose = 0

ep.network_interface = networkif
ep.addMotif("Init")

if args.tp > 1 and args.pp == 1 and args.dp == 1:
    ep.addMotif(f"LLMTensorParallelism batch_size={args.batch_size} sequence_len={args.sequence_len} n_step={args.n_step} llm_config={args.llm_config} verbose={args.verbose}")

if args.pp > 1 and args.tp == 1 and args.dp == 1:
    ep.addMotif(f"LLMPipelineParallelism batch_size={args.batch_size} sequence_len={args.sequence_len} n_step={args.n_step} llm_config={args.llm_config} verbose={args.verbose}")

if args.dp > 1 and args.tp == 1 and args.pp == 1:
    ep.addMotif(f"LLMDataParallelism batch_size={args.batch_size} sequence_len={args.sequence_len} n_step={args.n_step} llm_config={args.llm_config} verbose={args.verbose}")

if args.dp > 1 and args.tp > 1 and args.pp > 1:
    ep.addMotif(f"LLM3DParallelism tp={args.tp} pp={args.pp} dp={args.dp} batch_size={args.batch_size} sequence_len={args.sequence_len} n_step={args.n_step} llm_config={args.llm_config} verbose={args.verbose}")

ep.addMotif("Fini")
ep.nic.nic2host_lat= params["host_link_latency"]

if args.log:
    ep.enableMotifLog(args.log)

if topology == "single":
    topo = topoSingle()
    topo.num_ports = num_ranks
    rank_per_router = num_ranks

elif topology == "dragonfly":
    rank_per_router = args.tp
    topo = topoDragonFly()
    topo.hosts_per_router = rank_per_router

    topo.routers_per_group = args.dp
    topo.num_groups = args.pp

    topo.intergroup_links = params["width"]
    topo.algorithm = "minimal"

elif topology == "fattree":
    rank_per_router = args.tp
    topo = topoFatTree()
    fattree_shape = f"1,1:{args.pp},{args.pp}:{args.dp},{args.dp}:{rank_per_router}"
    topo.host_link_latency = params["host_link_latency"]
    topo.shape = fattree_shape
else:
    print(topology, " unknown!")
    sys.exit()

# Set up the routers
router = hr_router()
router.link_bw =  str(params["link_bw"]) + "GB/s"
router.flit_size = params["flit_size"]
router.xbar_bw = str(params["link_bw"]*rank_per_router) + "GB/s"
router.input_latency = params["input_latency"]
router.output_latency = params["output_latency"]
router.input_buf_size = params["input_buf_size"]
router.output_buf_size = params["output_buf_size"]
router.num_vns = params["num_vns"]
router.xbar_arb = params["xbar_arb"]


### Setup the topology
topo.link_latency = params["link_lat"]
topo.router = router

system = System()
system.setTopology(topo)
system.allocateNodes(ep,"linear")

system.build()
