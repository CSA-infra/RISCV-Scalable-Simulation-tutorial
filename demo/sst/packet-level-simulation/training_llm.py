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
    "link_bw" : "25GB/s",

    "input_buf_size" : "14KB",
    "output_buf_size" : "14KB",

    "flit_size" : "256B",

    "xbar_bw" : "50GB/s",

    "link_lat" : "10ns",
    "input_latency" : "10ns",
    "output_latency" : "10ns",
    "host_link_latency" : "100ns",

    "num_vns" : 1,
    "width": 2,

    "xbar_arb" : "merlin.xbar_arb_lru"
}

llm_config_default= os.path.join(os.path.dirname(os.path.realpath(__file__)), "llm_config.json")

parser = argparse.ArgumentParser(
        prog=f'sst {__file__} --',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--tp", type=int, help="Tensor Parallelism level", default=2)
parser.add_argument("--pp", type=int, help="Data Parallelism level", default=2)
parser.add_argument("--dp", type=int, help="Pipeline Parallelism level", default=2)
parser.add_argument("--batch_size", type=int, help="Number of sequence processed in parallel", default=32)
parser.add_argument("--sequence_len", type=int, help="Number of token per sequence", default=8192)
parser.add_argument("--n_step", type=int, help="Number of steps", default=128)
parser.add_argument("--llm_config", type=str, help="Configuration file of the Large Language Model", default=llm_config_default)
parser.add_argument("--verbose", type=int, help="Set verbosity", default=0)
parser.add_argument("--stats", type=str, help="write statistics, argument changes the filename", nargs="?", const="-")
args = parser.parse_args()

assert os.path.exists(args.llm_config), "LLM config file does not exist!"

num_ranks = args.tp * args.pp * args.dp

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
networkif.link_bw = params["link_bw"]
networkif.input_buf_size = params["input_buf_size"]
networkif.output_buf_size = params["output_buf_size"]

ep = EmberMPIJob(0,num_ranks)
ep.network_interface = networkif
ep.addMotif("Init")
#ep.addMotif(f"LLMTensorParallelism batch_size={args.batch_size} sequence_len={args.sequence_len} n_step={args.n_step} llm_config={args.llm_config} verbose={args.verbose}")
#ep.addMotif(f"LLMPipelineParallelism batch_size={args.batch_size} sequence_len={args.sequence_len} n_step={args.n_step} llm_config={args.llm_config} verbose={args.verbose}")
ep.addMotif(f"LLMDataParallelism batch_size={args.batch_size} sequence_len={args.sequence_len} n_step={args.n_step} llm_config={args.llm_config} verbose={args.verbose}")
#ep.addMotif(f"LLM3DParallelism tp={args.tp} pp={args.pp} dp={args.dp} batch_size={args.batch_size} sequence_len={args.sequence_len} n_step={args.n_step} llm_config={args.llm_config}")
ep.addMotif("Fini")
ep.nic.nic2host_lat= params["host_link_latency"]

# Set up the routers
router = hr_router()
router.link_bw = params["link_bw"]
router.flit_size = params["flit_size"]
router.xbar_bw = params["xbar_bw"]
router.input_latency = params["input_latency"]
router.output_latency = params["output_latency"]
router.input_buf_size = params["input_buf_size"]
router.output_buf_size = params["output_buf_size"]
router.num_vns = params["num_vns"]
router.xbar_arb = params["xbar_arb"]

topo = topoSingle()
topo.num_ports = num_ranks


### Setup the topology
topo.link_latency = params["link_lat"]
topo.router = router
#topo.bundleEndpoints = False

system = System()
system.setTopology(topo)
system.allocateNodes(ep,"linear")

system.build()
