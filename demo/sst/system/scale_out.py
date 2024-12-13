import sst
import os
from sst.merlin import *

os_verbosity = 0

enableStats = False
sst.setStatisticLoadLevel(4)
sst.setStatisticOutput("sst.statOutputConsole")

num_threads_per_cpu = 2
num_cpu_per_node = 1
num_node = 4
app_args = "64 64 4"

cpu_clock = "3GHz"

coherence_protocol="MESI"
cache_line_size = 64

l2cache_size = 1 * 1024**2 # 1MiB
page_size = 4096
memsize = 2 * 1024**3 # 2GiB
physMemSize = str(memsize) + " B"


full_exe_name = "../software/riscv64/mha_MPI_OMP"
#full_exe_name = "../software/riscv64/hello_MPI_OMP"

exe_name= full_exe_name.split("/")[-1]

network_topology = "simple"
network_topology = "torus"


rdma_nic_num_posted_recv=128
rdma_nic_comp_q_size=256

tlbParams = {
        "hitLatency": 3,
        "num_tlb_entries_per_thread": 64,
        "tlb_set_size": 4,
        "minVirtAddr" : 0x1000,
        "maxVirtAddr" : memsize
        }

networkParams = {
        "packetSize" : "2048B",
        "link_bw" : "50GB/s",
        "xbar_bw" : "50GB/s",

        "link_lat" : "10ns",
        "input_latency" : "10ns",
        "output_latency" : "10ns",

        "flit_size" : "256B",
        "input_buf_size" : "14KB",
        "output_buf_size" : "14KB",
}

rdmaLinkParams = {
        "link_bw" :"50GB/s",
        "input_buf_size" : "14KB",
        "output_buf_size" : "14KB"
        }




if network_topology == "torus":
    networkParams |= {
            "num_dims" : 2,
            "torus.width" : "1x1",
            "torus.shape" : "2x2",
            "torus.local_ports" : 1
            }

else: # simple
    networkParams |= {
            "router_radix" : num_node
            }


nodeRtrParams = {
        "xbar_bw" : "512GB/s",
        "link_bw" : "400GB/s",
        "input_buf_size" : "40KB",
        "output_buf_size" : "40KB",
        "flit_size" : "72B",
        "id" : "0",
        "topology" : "merlin.singlerouter"
        }

memCtrlParams = {
        "clock" : "1.6GHz",
        "backend.mem_size" : physMemSize,
        "backing" : "malloc",
        "initBacking" : 1,
        "addr_range_start" : 0x0,
        "addr_range_end" : memsize - 1,
        "backendConvertor.request_width" : cache_line_size
        }

# DRAM bandwidth = memCtrl.clock * cache_line_size * max_requests_per_cycle = 204.8 GB/s
memCtrlParams = {
        "clock" : "1.6GHz",
        "backend.mem_size" : physMemSize,
        "backing" : "malloc",
        "initBacking" : 1,
        "addr_range_start" : 0x0,
        "addr_range_end" : memsize - 1,
        "backendConvertor.request_width" : cache_line_size
        }

memBackendParams = {
        "mem_size" : physMemSize,
        "access_time" : "25ns",
        "max_requests_per_cycle" : 2,
        "request_width" : cache_line_size
        }

memNICParams = {
        "min_packet_size" : "72B",
        "network_bw" : "400GB/s",
        "network_input_buffer_size" : "4KiB",
        "network_output_buffer_size" : "4KiB"
        }

# OS related params
osParams = {
        "dbgLevel" : os_verbosity,
        "dbgMask" : 8,
        "cores" : num_cpu_per_node,
        "hardwareThreadCount" : num_threads_per_cpu,
        "page_size"  : page_size,
        "physMemSize" : physMemSize,
        "useMMU" : True,
        }

osl1cacheParams = {
        "access_latency_cycles" : 1,
        "cache_frequency" : cpu_clock,
        "replacement_policy" : "lru",
        "coherence_protocol" : coherence_protocol,
        "associativity" : 8,
        "cache_line_size" : cache_line_size,
        "cache_size" : "32 KiB",
        "L1" : "1",
        }

mmuParams = {
        "num_cores": num_cpu_per_node,
        "num_threads": num_threads_per_cpu,
        "page_size": page_size,
        "useNicTlb": True,
        }


vanadis_cpu_type = "vanadis.VanadisCPU"
cpuParams = {
        "clock" : cpu_clock,
        "hardware_threads": num_threads_per_cpu,
        "physical_fp_registers" : 168 * num_threads_per_cpu,
        "physical_integer_registers" : 180 * num_threads_per_cpu,
        "integer_arith_units" : 2,
        "integer_arith_cycles" : 2,
        "integer_div_units" : 1,
        "integer_div_cycles" : 20,
        "fp_arith_cycles" : 3,
        "fp_arith_units" : 2,
        "fp_div_units" : 2,
        "fp_div_cycles" : 20,
        "branch_units" : 1,
        "branch_unit_cycles" : 2,
        "reorder_slots" : 128,
        "decodes_per_cycle" : 4,
        "issues_per_cycle" :  4,
        "retires_per_cycle" : 4,
        }

branchPredParams = {
        "branch_entries" : 64
        }

decoderParams = {
        "loader_mode" : 1,
        "uop_cache_entries" : 1536,
        "predecode_cache_entries" : 4
        }

lsqParams = {
        "max_stores" : 16,
        "max_loads" : 32,
        }



l1dcacheParams = {
        "access_latency_cycles" : 1,
        "cache_frequency" : cpu_clock,
        "replacement_policy" : "lru",
        "coherence_protocol" : coherence_protocol,
        "associativity" : 8,
        "cache_line_size" : cache_line_size,
        "cache_size" : "64 KiB",
        "prefetcher" : "cassini.NextBlockPrefetcher",
        "prefetcher.reach" : 2,
        "L1" : "1",
        }

l1icacheParams = {
        "access_latency_cycles" : 1,
        "cache_frequency" : cpu_clock,
        "replacement_policy" : "lru",
        "coherence_protocol" : coherence_protocol,
        "associativity" : 8,
        "cache_line_size" : cache_line_size,
        "cache_size" : "32 KiB",
        "prefetcher" : "cassini.NextBlockPrefetcher",
        "prefetcher.reach" : 1,
        "L1" : "1",
        }

l2cacheParams = {
        "access_latency_cycles" : 8,
        "max_requests_per_cycle" : 8,
        "cache_frequency" : cpu_clock,
        "replacement_policy" : "lru",
        "coherence_protocol" : coherence_protocol,
        "associativity" : 16,
        "cache_line_size" : cache_line_size,
        "cache_size" : str(l2cache_size) + 'B',
        "mshr_latency_cycles": 3,
        }

busParams = {
        "bus_frequency" : cpu_clock,
        }

dirCtrlParams = {
        "coherence_protocol" : coherence_protocol,
        "entry_cache_size" : l2cache_size*num_cpu_per_node/cache_line_size,
        "cache_line_size" : cache_line_size,
        "addr_range_start" : 0x0,
        "addr_range_end" : memsize - 1
        }


rdmaNiCParams = {
        "clock" : cpu_clock,
        "useDmaCache": "true",
        "maxPendingCmds" : rdma_nic_num_posted_recv,
        "maxMemReqs" : rdma_nic_comp_q_size,
        "maxCmdQSize" : rdma_nic_num_posted_recv,
        "cache_line_size"    : cache_line_size,
        'baseAddr': memsize,
        'cmdQSize' : 64,
        }


rdmaCacheParams = {
        "access_latency_cycles" : 2,
        "max_requests_per_cycle" : 1,
        "mshr_num_entries": 64,
        "cache_frequency" : cpu_clock,
        "replacement_policy" : "lru",
        "coherence_protocol" : coherence_protocol,
        "associativity" : 8,
        "cache_line_size" : cache_line_size,
        "cache_size" : "32 KiB",
        "L1" : "1",
        }


app_params = {}
if app_args != "":
    app_args_list = app_args.split(" ")
    # We have a plus 1 because the executable name is arg0
    app_args_count = len( app_args_list ) + 1

    app_params["argc"] = app_args_count

    arg_start = 1
    for next_arg in app_args_list:
        app_params["arg" + str(arg_start)] = next_arg
        arg_start = arg_start + 1
else:
    app_params["argc"] = 1

class CPU_Builder:
    def __init__(self):
        pass

    def build( self, nodeId, cpuId ):

        prefix = 'node' + str(nodeId) + '.cpu' + str( cpuId )
        cpu = sst.Component(prefix, vanadis_cpu_type)
        cpu.addParams( cpuParams )
        cpu.addParam( "core_id", cpuId )
        cpu.addParam( "node_id", nodeId )
        if enableStats:
            cpu.enableAllStatistics()

        # CPU.decoder
        for n in range(num_threads_per_cpu):
            decode     = cpu.setSubComponent( "decoder"+str(n), "vanadis.VanadisRISCV64Decoder" )
            decode.addParams( decoderParams )

            if enableStats:
                decode.enableAllStatistics()

            # CPU.decoder.osHandler
            os_hdlr     = decode.setSubComponent( "os_handler", "vanadis.VanadisRISCV64OSHandler" )

            # CPU.decocer.branch_pred
            branch_pred = decode.setSubComponent( "branch_unit", "vanadis.VanadisBasicBranchUnit" )
            branch_pred.addParams( branchPredParams )

            if enableStats:
                branch_pred.enableAllStatistics()


        # CPU.lsq
        cpu_lsq = cpu.setSubComponent( "lsq", "vanadis.VanadisBasicLoadStoreQueue" )
        cpu_lsq.addParams(lsqParams)
        if enableStats:
            cpu_lsq.enableAllStatistics()


        icache_if = cpu.setSubComponent( "mem_interface_inst", "memHierarchy.standardInterface" )
        icache_if.addParam("coreId",cpuId)

        dcache_if = cpu_lsq.setSubComponent( "memory_interface", "memHierarchy.standardInterface" )
        dcache_if.addParam("coreId",cpuId)

        # L1 D-Cache
        l1cache = sst.Component(prefix + ".l1dcache", "memHierarchy.Cache")
        l1cache.addParams( l1dcacheParams )
        if enableStats:
            l1cache.enableAllStatistics()

        l1dcache_2_cpu     = l1cache.setSubComponent("cpulink", "memHierarchy.MemLink")
        l1dcache_2_l2cache = l1cache.setSubComponent("memlink", "memHierarchy.MemLink")

        # L1 I-Cache
        l1icache = sst.Component(prefix + ".l1icache", "memHierarchy.Cache")
        l1icache.addParams(l1icacheParams)
        if enableStats:
            l1icache.enableAllStatistics()

        # Bus
        cache_bus = sst.Component(prefix + ".bus", "memHierarchy.Bus")
        cache_bus.addParams(busParams)
        if enableStats:
            cache_bus.enableAllStatistics()

        # L2 D-Cache
        l2cache = sst.Component(prefix + ".l2cache", "memHierarchy.Cache")
        l2cache.addParams(l2cacheParams)
        if enableStats:
            l2cache.enableAllStatistics()

        l2cache_2_cpu = l2cache.setSubComponent("cpulink", "memHierarchy.MemLink")

        # CPU D-TLB
        dtlbWrapper = sst.Component(prefix+".dtlb", "mmu.tlb_wrapper")
        dtlb = dtlbWrapper.setSubComponent("tlb", "mmu.simpleTLB" );
        dtlb.addParam("num_hardware_threads", num_threads_per_cpu)
        dtlb.addParams(tlbParams)

        # CPU I-TLB
        itlbWrapper = sst.Component(prefix+".itlb", "mmu.tlb_wrapper")
        itlbWrapper.addParam("exe",True)
        itlb = itlbWrapper.setSubComponent("tlb", "mmu.simpleTLB" );
        itlb.addParam("num_hardware_threads", num_threads_per_cpu)
        itlb.addParams(tlbParams)

        # CPU (data) -> D-TLB
        link = sst.Link(prefix+".link_cpu_dtlb")
        link.connect( (dcache_if, "port", "25ps"), (dtlbWrapper, "cpu_if", "25ps") )

        # CPU (instruction) -> I-TLB
        link = sst.Link(prefix+".link_cpu_itlb")
        link.connect( (icache_if, "port", "25ps"), (itlbWrapper, "cpu_if", "25ps") )

        l1icache_2_cpu     = l1icache.setSubComponent("cpulink", "memHierarchy.MemLink")
        l1icache_2_l2cache = l1icache.setSubComponent("memlink", "memHierarchy.MemLink")

        # D-TLB -> D-L1
        link = sst.Link(prefix+".link_l1cache")
        link.connect( (dtlbWrapper, "cache_if", "25ps"), (l1dcache_2_cpu, "port", "25ps") )

        # I-TLB -> I-L1
        link = sst.Link(prefix+".link_l1icache")
        link.connect( (itlbWrapper, "cache_if", "25ps"), (l1icache_2_cpu, "port", "25ps") )

        # L1 I-Cache to bus
        link = sst.Link(prefix + ".link_l1dcache_l2cache")
        link.connect( (l1dcache_2_l2cache, "port", "25ps"), (cache_bus, "high_network_0", "25ps") )

        # L1 D-Cache to bus
        link = sst.Link(prefix + ".link_l1icache_l2cache")
        link.connect( (l1icache_2_l2cache, "port", "25ps"), (cache_bus, "high_network_1", "25ps") )

        # BUS to L2 cache
        link = sst.Link(prefix+".link_bus_l2cache")
        link.connect( (cache_bus, "low_network_0", "25ps"), (l2cache_2_cpu, "port", "25ps") )

        return cpu, l2cache, dtlb, itlb


def addParamsPrefix(prefix,params):
    #print( prefix )
    ret = {}
    for key, value in params.items():
        #print( key, value )
        ret[ prefix + "." + key] = value

    #print( ret )
    return ret



class OS_Builder:
    def __init__(self):
        pass

    def build( self, numNodes, nodeId):

        self.prefix = 'node' + str(nodeId)

        self.nodeOS = sst.Component(self.prefix + ".os", "vanadis.VanadisNodeOS")
        self.nodeOS.addParam("node_id", nodeId)
        self.nodeOS.addParams(osParams)
        if enableStats:
            self.nodeOS.enableAllStatistics()

        processList = (
                ( 1, {
                    "env_count" : 7,
                    "env0" : "OMP_NUM_THREADS={}".format(num_cpu_per_node*num_threads_per_cpu),
                    "env1" : "PMI_SIZE={}".format(num_node),
                    "env2" : "PMI_RANK={}".format(nodeId),
                    "env3" : "RDMA_NIC_NUM_POSTED_RECV={}".format(rdma_nic_num_posted_recv),
                    "env4" : "RDMA_NIC_COMP_Q_SIZE={}".format(rdma_nic_comp_q_size),
                    "env5" : "TZ=UTC",
                    "env6" : "MV2_ENABLE_AFFINITY=0",
                    "exe"  : full_exe_name,
                    "arg0" : exe_name,
                    } ),
                )

        processList[0][1].update(app_params)

        num=0
        for i,process in processList:
            for y in range(i):
                self.nodeOS.addParams( addParamsPrefix( "process" + str(num), process ) )
                num+=1

        self.mmu = self.nodeOS.setSubComponent( "mmu", "mmu.simpleMMU" )

        self.mmu.addParams(mmuParams)

        mem_if = self.nodeOS.setSubComponent( "mem_interface", "memHierarchy.standardInterface" )

        l1cache = sst.Component(self.prefix + ".node_os.l1cache", "memHierarchy.Cache")
        l1cache.addParams(osl1cacheParams)

        l1cache_2_cpu = l1cache.setSubComponent("cpulink", "memHierarchy.MemLink")

        link = sst.Link(self.prefix + ".link_os_l1cache")
        link.connect( (mem_if, "port", "25ps"), (l1cache_2_cpu, "port", "25ps") )

        return l1cache

    def connectCPU( self, core, cpu ):
        link = sst.Link(self.prefix + ".link_core" + str(core) + "_os")
        link.connect( (cpu, "os_link", "5ns"), (self.nodeOS, "core" + str(core), "5ns") )

    def connectTlb( self, core, name, tlblink ):
        linkName = self.prefix + ".link_mmu_core" + str(core) + "_" + name
        link = sst.Link( linkName )
        link.connect( (self.mmu, "core"+str(core)+ "." +name, "25ps"), (tlblink, "mmu", "25ps") )

    def connectNicTlb( self, name, niclink ):
        linkName = self.prefix + ".link_mmu_" + name
        link = sst.Link( linkName )
        link.connect( (self.mmu, name, "25ps"), (niclink, "mmu", "25ps") )




class rdmaNic_Builder:
    def __init__(self,numNodes):
        self.numNodes = numNodes

    def build( self, nodeId ):

        prefix = 'node' + str(nodeId)
        nic = sst.Component( prefix + ".nic", "rdmaNic.nic")
        nic.addParams(rdmaNiCParams)
        nic.addParam( 'nicId', nodeId )
        nic.addParam( 'pesPerNode', 1 )
        nic.addParam( 'numNodes', self.numNodes )
        if enableStats :
            nic.enableAllStatistics()


        # NIC DMA interface
        dmaIf = nic.setSubComponent("dma", "memHierarchy.standardInterface")

        # NIC MMIO interface
        mmioIf = nic.setSubComponent("mmio", "memHierarchy.standardInterface")

        # NIC DMA Cache
        dmaCache = sst.Component(prefix + ".nicDmaCache", "memHierarchy.Cache")
        dmaCache.addParams(rdmaCacheParams)

        # NIC DMA TLB
        tlbWrapper = sst.Component(prefix+".nicDmaTlb", "mmu.tlb_wrapper")
        tlb = tlbWrapper.setSubComponent("tlb", "mmu.simpleTLB" );
        tlb.addParam("num_hardware_threads", num_cpu_per_node*num_threads_per_cpu)
        tlb.addParams(tlbParams)

        # Cache to CPU interface
        dmaCacheToCpu = dmaCache.setSubComponent("cpulink", "memHierarchy.MemLink")

        # NIC DMA -> TLB
        link = sst.Link(prefix+".link_cpu_dtlb")
        link.connect( (dmaIf, "port", "25ps"), (tlbWrapper, "cpu_if", "25ps") )

        # NIC DMA TLB -> cache
        link = sst.Link(prefix+".link_cpu_l1dcache")
        link.connect( (tlbWrapper, "cache_if", "25ps"), (dmaCacheToCpu, "port", "25ps") )

        # NIC internode interface
        netLink = nic.setSubComponent( "rtrLink", "merlin.linkcontrol" )
        netLink.addParams(rdmaLinkParams)

        return mmioIf, dmaCache, tlb, (netLink, "rtr_port", '10ns')

class memory_Builder:
    def __init__(self):
        pass

    def build( self, nodeId, numPorts,  group  ):

        self.prefix = 'node' + str(nodeId)
        self.numPorts = numPorts + 1

        self.chiprtr = sst.Component(self.prefix + ".chiprtr", "merlin.hr_router")
        self.chiprtr.addParam("num_ports", self.numPorts)
        self.chiprtr.addParams(nodeRtrParams)
        self.chiprtr.setSubComponent("topology","merlin.singlerouter")

        if enableStats:
            self.chiprtr.enableAllStatistics()

        dirctrl = sst.Component(self.prefix + ".dirctrl", "memHierarchy.DirectoryController")
        dirctrl.addParams(dirCtrlParams)
        dirtoMemLink = dirctrl.setSubComponent("memlink", "memHierarchy.MemLink")
        self.connect( "Dirctrl", self.numPorts - 1, dirctrl, group, linkType="cpulink" )
        if enableStats:
            dirctrl.enableAllStatistics()

        memctrl = sst.Component(self.prefix + ".memory", "memHierarchy.MemController")
        memctrl.addParams(memCtrlParams)
        if enableStats:
            memctrl.enableAllStatistics()

        memToDir = memctrl.setSubComponent("cpulink", "memHierarchy.MemLink")

        memory = memctrl.setSubComponent("backend", "memHierarchy.simpleMem")
        memory.addParams(memBackendParams)

        link = sst.Link(self.prefix + ".link_dir_mem")
        link.connect( (dirtoMemLink, "port", "25ps"), (memToDir, "port", "25ps") )

    def connect( self, name, port, comp, group=None, linkType="memlink"  ):

        assert group
        assert port < self.numPorts

        memNIC = comp.setSubComponent(linkType, "memHierarchy.MemNIC")
        memNIC.addParam("group", group)
        memNIC.addParams(memNICParams)

        link = sst.Link(self.prefix + ".link_rtr" + str(port) )
        link.connect( (self.chiprtr, "port" + str(port), "25ps"), (memNIC, "port", "25ps") )


class Endpoint():
    def __init__(self,numNodes):
        self.numNodes = numNodes

    def prepParams(self):
        pass

    def build(self, nodeId, extraKeys ):

        prefix = 'node' + str(nodeId);

        cpuBuilder = CPU_Builder()
        memBuilder = memory_Builder()
        osBuilder = OS_Builder()

        numPorts = 3 + num_cpu_per_node
        port = 0
        memBuilder.build(nodeId, numPorts, group=2 )

        # build the Vanadis OS, it returns
        osCache = osBuilder.build( self.numNodes, nodeId)

        # connect OS L1 to Memory
        memBuilder.connect( "OS_L1", port, osCache, group=1 )
        port += 1;

        # build the Vanadis CPU block, this returns
        # cpu, L2 cache, DTLB ITLB
        for i in range(num_cpu_per_node):
            cpu, L2, dtlb, itlb = cpuBuilder.build(nodeId, i)

            osBuilder.connectCPU( i, cpu )
            osBuilder.connectTlb( i, "dtlb", dtlb )
            osBuilder.connectTlb( i, "itlb", itlb )

            # connect CPU L2 to Memory
            memBuilder.connect( "CPU_L2", port, L2, group=1 )
            port += 1;

        nicBuilder = rdmaNic_Builder(self.numNodes)
        # build the Rdma NIC, this returns
        # MMIO link, DMA cache, DMA TLB
        mmioIf, dmaCache, dmaTlb, netLink = nicBuilder.build(nodeId)

        osBuilder.connectNicTlb( "nicTlb", dmaTlb )

        # connect the NIC MMIO to Memory
        #memBuilder.connect( "NIC_MMIO", port, mmioIf, 3, source="1", dest="2" )
        memBuilder.connect( "NIC_MMIO", port, mmioIf, group=2 )
        port += 1;

        # connect the NIC DMA Cache to Memory
        #memBuilder.connect( "NIC_DMA", port, dmaCache, 1, dest="2" )
        memBuilder.connect( "NIC_DMA", port, dmaCache, group=1 )
        port += 1;
        return netLink

ep = Endpoint( num_node )

def setNode( nodeId ):
    return ep;

for p in networkParams:
    sst.merlin._params[p] = networkParams[p]

if network_topology == "torus":
    topo = topoTorus()
else:
    topo = topoSimple()

topo.prepParams()
topo.setEndPointFunc( setNode )
topo.build()
