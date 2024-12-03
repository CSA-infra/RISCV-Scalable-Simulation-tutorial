import sst
import os
from sst.merlin import *


sst.setStatisticLoadLevel(4)
sst.setStatisticOutput("sst.statOutputConsole")

num_node = 2
os_verbosity = 0
coherence_protocol="MESI"
cpu_clock = "3GHz"
physMemSize = "2GiB"
full_exe_name = "../software/riscv64/mha_MPI_OMP"
app_args = "32 32 2"
# MUSL libc uses this in localtime, if we don't set TZ=UTC we
# can get different results on different systems
app_env = ("TZ=UTC " + "RDMA_NIC_NUM_POSTED_RECV=128 RDMA_NIC_COMP_Q_SIZE=256 OMP_NUM_THREADS=1").split()

tlbParams = {
        "hitLatency": 10,
        "num_hardware_threads": 1,
        "num_tlb_entries_per_thread": 64,
        "tlb_set_size": 4,
        }

networkParams = {
        "packetSize" : "2048B",
        "link_bw" : "16GB/s",
        "xbar_bw" : "16GB/s",

        "link_lat" : "10ns",
        "input_latency" : "10ns",
        "output_latency" : "10ns",

        "flit_size" : "8B",
        "input_buf_size" : "14KB",
        "output_buf_size" : "14KB",

        "num_dims" : 2,
        "torus.width" : "1x1",
        "torus.shape" : "2x1",
        "torus.local_ports" : 1
        }



verbosity = 1
lsq_ld_entries = 16
lsq_st_entries = os.getenv("VANADIS_LSQ_ST_ENTRIES", 8)

rob_slots = os.getenv("VANADIS_ROB_SLOTS", 64)
retires_per_cycle = os.getenv("VANADIS_RETIRES_PER_CYCLE", 4)
issues_per_cycle = os.getenv("VANADIS_ISSUES_PER_CYCLE", 4)
decodes_per_cycle = os.getenv("VANADIS_DECODES_PER_CYCLE", 4)

integer_arith_cycles = int(os.getenv("VANADIS_INTEGER_ARITH_CYCLES", 2))
integer_arith_units = int(os.getenv("VANADIS_INTEGER_ARITH_UNITS", 2))
fp_arith_cycles = int(os.getenv("VANADIS_FP_ARITH_CYCLES", 8))
fp_arith_units = int(os.getenv("VANADIS_FP_ARITH_UNITS", 2))
branch_arith_cycles = int(os.getenv("VANADIS_BRANCH_ARITH_CYCLES", 2))

vanadis_cpu_type = "vanadis.VanadisCPU"
#vanadis_cpu_type = "vanadis.VanadisCPU"

if (verbosity > 0):
    print( "Verbosity: " + str(verbosity) + " -> loading Vanadis CPU type: " + vanadis_cpu_type )

class vanadis_Builder:
    def __init__(self):
        pass

    def build( self, nodeId, cpuId ):

        prefix = 'node' + str(nodeId) + '.cpu' + str( cpuId )
        cpu = sst.Component(prefix, vanadis_cpu_type)
        cpu.addParams({
            "clock" : cpu_clock,
            "verbose" : verbosity,
            "physical_fp_registers" : 168,
            "physical_int_registers" : 180,
            "integer_arith_cycles" : integer_arith_cycles,
            "integer_arith_units" : integer_arith_units,
            "fp_arith_cycles" : fp_arith_cycles,
            "fp_arith_units" : fp_arith_units,
            "branch_unit_cycles" : branch_arith_cycles,
            "print_int_reg" : 1,
            "reorder_slots" : rob_slots,
            "decodes_per_cycle" : decodes_per_cycle,
            "issues_per_cycle" :  issues_per_cycle,
            "retires_per_cycle" : retires_per_cycle,
            "pause_when_retire_address" : os.getenv("VANADIS_HALT_AT_ADDRESS", 0)
            })
        cpu.enableAllStatistics()

        if app_args != "":
            app_args_list = app_args.split(" ")
            # We have a plus 1 because the executable name is arg0
            app_args_count = len( app_args_list ) + 1
            cpu.addParams({ "app.argc" : app_args_count })
            if (verbosity > 0):
                print( "Identified " + str(app_args_count) + " application arguments, adding to input parameters." )
            arg_start = 1
            for next_arg in app_args_list:
                if (verbosity > 0):
                    print( "arg" + str(arg_start) + " = " + next_arg )
                cpu.addParams({ "app.arg" + str(arg_start) : next_arg })
                arg_start = arg_start + 1
        else:
            if (verbosity > 0):
                print( "No application arguments found, continuing with argc=0" )

        decode = cpu.setSubComponent( "decoder0", "vanadis.VanadisRISCV64Decoder" )

        decode.addParams({
            "uop_cache_entries" : 1536,
            "predecode_cache_entries" : 4
            })
        decode.enableAllStatistics()

        os_hdlr = decode.setSubComponent( "os_handler", "vanadis.VanadisRISCV64OSHandler" )
        os_hdlr.addParams({
            "verbose" : os_verbosity,
            "brk_zero_memory" : "yes"
            })

        branch_pred = decode.setSubComponent( "branch_unit", "vanadis.VanadisBasicBranchUnit")
        branch_pred.addParams({
            "branch_entries" : 32
            })
        branch_pred.enableAllStatistics()

        icache_if = cpu.setSubComponent( "mem_interface_inst", "memHierarchy.standardInterface" )
        icache_if.addParam("coreId",cpuId)

        cpu_lsq = cpu.setSubComponent( "lsq", "vanadis.VanadisBasicLoadStoreQueue" )
        cpu_lsq.addParams({
            "verbose" : verbosity,
            "address_mask" : 0xFFFFFFFF,
            "max_loads" : lsq_ld_entries,
            "max_stores" : lsq_st_entries
            })
        cpu_lsq.enableAllStatistics()

        dcache_if = cpu_lsq.setSubComponent( "memory_interface", "memHierarchy.standardInterface" )
        dcache_if.addParam("coreId",cpuId)

        # L1 D-Cache
        l1cache = sst.Component(prefix + ".l1dcache", "memHierarchy.Cache")
        l1cache.addParams({
            "access_latency_cycles" : "2",
            "cache_frequency" : cpu_clock,
            "replacement_policy" : "lru",
            "coherence_protocol" : coherence_protocol,
            "associativity" : "8",
            "cache_line_size" : "64",
            "cache_size" : "32 KB",
            "L1" : "1",
            })

        l1dcache_2_cpu     = l1cache.setSubComponent("cpulink", "memHierarchy.MemLink")
        l1dcache_2_l2cache = l1cache.setSubComponent("memlink", "memHierarchy.MemLink")

        # L1 I-Cache
        l1icache = sst.Component(prefix + ".l1icache", "memHierarchy.Cache")
        l1icache.addParams({
            "access_latency_cycles" : "2",
            "cache_frequency" : cpu_clock,
            "replacement_policy" : "lru",
            "coherence_protocol" : coherence_protocol,
            "associativity" : "8",
            "cache_line_size" : "64",
            "cache_size" : "32 KB",
            #"prefetcher" : "cassini.NextBlockPrefetcher",
            #"prefetcher.reach" : 1,
            "L1" : "1",
            })

        # Bus
        cache_bus = sst.Component(prefix + ".bus", "memHierarchy.Bus")
        cache_bus.addParams({
            "bus_frequency" : cpu_clock,
            })

        # L2 D-Cache
        l2cache = sst.Component(prefix + ".l2cache", "memHierarchy.Cache")
        l2cache.addParams({
            "access_latency_cycles" : "14",
            "cache_frequency" : cpu_clock,
            "replacement_policy" : "lru",
            "coherence_protocol" : coherence_protocol,
            "associativity" : "16",
            "cache_line_size" : "64",
            "mshr_latency_cycles" : 3,
            "cache_size" : "1MB",
            })

        l2cache_2_cpu = l2cache.setSubComponent("cpulink", "memHierarchy.MemLink")

        # CPU D-TLB
        dtlbWrapper = sst.Component(prefix+".dtlb", "mmu.tlb_wrapper")
        dtlb = dtlbWrapper.setSubComponent("tlb", "mmu.simpleTLB" );
        dtlb.addParams(tlbParams)

        # CPU I-TLB
        itlbWrapper = sst.Component(prefix+".itlb", "mmu.tlb_wrapper")
        itlbWrapper.addParam("exe",True)
        itlb = itlbWrapper.setSubComponent("tlb", "mmu.simpleTLB" );
        itlb.addParams(tlbParams)

        # CPU (data) -> D-TLB
        link = sst.Link(prefix+".link_cpu_dtlb")
        link.connect( (dcache_if, "port", "1ns"), (dtlbWrapper, "cpu_if", "1ns") )

        # CPU (instruction) -> I-TLB
        link = sst.Link(prefix+".link_cpu_itlb")
        link.connect( (icache_if, "port", "1ns"), (itlbWrapper, "cpu_if", "1ns") )

        l1icache_2_cpu     = l1icache.setSubComponent("cpulink", "memHierarchy.MemLink")
        l1icache_2_l2cache = l1icache.setSubComponent("memlink", "memHierarchy.MemLink")

        # D-TLB -> D-L1
        link = sst.Link(prefix+".link_l1cache")
        link.connect( (dtlbWrapper, "cache_if", "1ns"), (l1dcache_2_cpu, "port", "1ns") )

        # I-TLB -> I-L1
        link = sst.Link(prefix+".link_l1icache")
        link.connect( (itlbWrapper, "cache_if", "1ns"), (l1icache_2_cpu, "port", "1ns") )

        # L1 I-Cache to bus
        link = sst.Link(prefix + ".link_l1dcache_l2cache")
        link.connect( (l1dcache_2_l2cache, "port", "1ns"), (cache_bus, "high_network_0", "1ns") )

        # L1 D-Cache to bus
        link = sst.Link(prefix + ".link_l1icache_l2cache")
        link.connect( (l1icache_2_l2cache, "port", "1ns"), (cache_bus, "high_network_1", "1ns") )

        # BUS to L2 cache
        link = sst.Link(prefix+".link_bus_l2cache")
        link.connect( (cache_bus, "low_network_0", "1ns"), (l2cache_2_cpu, "port", "1ns") )

        return cpu, l2cache, dtlb, itlb


class vanadisOS_Builder:
    def __init__(self):
        pass

    def build( self, numNodes, nodeId, cpuId ):

        self.prefix = 'node' + str(nodeId)

        self.nodeOS = sst.Component(self.prefix + ".os", "vanadis.VanadisNodeOS")
        self.nodeOS.addParams({
            "node_id": nodeId,
            "dbgLevel" : os_verbosity,
            "dbgMask" : -1,
            "cores" : 1,
            "hardwareThreadCount" : 1,
            "page_size"  : 4096,
            "physMemSize" : physMemSize,
            "process0.exe" : full_exe_name,
            "useMMU" : True,
            })

        cnt = 0
        for value in app_args:
            key= "process0.arg" + str(cnt);
            self.nodeOS.addParam( key, value )
            cnt += 1

        self.nodeOS.addParam( "process0.argc", cnt )

        cnt = 0
        for value in app_env:
            key= "process0.env" + str(cnt);
            self.nodeOS.addParam( key, value )
            cnt += 1


        # for mvapich runtime
        self.nodeOS.addParam(  "process0.env" + str(cnt), "PMI_SIZE=" + str(numNodes) )
        cnt += 1

        self.nodeOS.addParam(  "process0.env" + str(cnt), "PMI_RANK=" + str(nodeId) )
        cnt += 1

        self.nodeOS.addParam( "process0.env_count", cnt )

        self.mmu = self.nodeOS.setSubComponent( "mmu", "mmu.simpleMMU" )

        self.mmu.addParams({
            "num_cores": 1,
            "num_threads": 1,
            "page_size": 4096,
            "useNicTlb": True,
            })

        mem_if = self.nodeOS.setSubComponent( "mem_interface", "memHierarchy.standardInterface" )
        mem_if.addParam("coreId",cpuId)

        l1cache = sst.Component(self.prefix + ".node_os.l1cache", "memHierarchy.Cache")
        l1cache.addParams({
            "access_latency_cycles" : "2",
            "cache_frequency" : cpu_clock,
            "replacement_policy" : "lru",
            "coherence_protocol" : coherence_protocol,
            "associativity" : "8",
            "cache_line_size" : "64",
            "cache_size" : "32 KB",
            "L1" : "1",
            })

        l1cache_2_cpu = l1cache.setSubComponent("cpulink", "memHierarchy.MemLink")

        link = sst.Link(self.prefix + ".link_os_l1cache")
        link.connect( (mem_if, "port", "1ns"), (l1cache_2_cpu, "port", "1ns") )

        return l1cache

    def connectCPU( self, core, cpu ):
        link = sst.Link(self.prefix + ".link_core" + str(core) + "_os")
        link.connect( (cpu, "os_link", "5ns"), (self.nodeOS, "core0", "5ns") )

    def connectTlb( self, core, name, tlblink ):
        linkName = self.prefix + ".link_mmu_core" + str(core) + "_" + name
        link = sst.Link( linkName )
        link.connect( (self.mmu, "core"+str(core)+ "." +name, "1ns"), (tlblink, "mmu", "1ns") )

    def connectNicTlb( self, name, niclink ):
        linkName = self.prefix + ".link_mmu_" + name
        link = sst.Link( linkName )
        link.connect( (self.mmu, name, "1ns"), (niclink, "mmu", "1ns") )




class rdmaNic_Builder:
    def __init__(self,numNodes):
        self.numNodes = numNodes

    def build( self, nodeId ):

        prefix = 'node' + str(nodeId)
        nic = sst.Component( prefix + ".nic", "rdmaNic.nic")
        nic.addParams({
            "clock" : "1GHz",
            "useDmaCache": "true",
            "maxPendingCmds" : 128,
            "maxMemReqs" : 256,
            "maxCmdQSize" : 128,
            "cache_line_size"    : 64,
            'baseAddr': 0x80000000,
            'cmdQSize' : 64,
            })
        nic.addParam( 'nicId', nodeId )
        nic.addParam( 'pesPerNode', 1 )
        nic.addParam( 'numNodes', self.numNodes )


        # NIC DMA interface
        dmaIf = nic.setSubComponent("dma", "memHierarchy.standardInterface")

        # NIC MMIO interface
        mmioIf = nic.setSubComponent("mmio", "memHierarchy.standardInterface")

        # NIC DMA Cache
        dmaCache = sst.Component(prefix + ".nicDmaCache", "memHierarchy.Cache")
        dmaCache.addParams({
            "access_latency_cycles" : "1",
            "access_latency_cycles" : "2",
            "cache_frequency" : cpu_clock,
            "replacement_policy" : "lru",
            "coherence_protocol" : coherence_protocol,
            "associativity" : "8",
            "cache_line_size" : "64",
            "cache_size" : "32KB",
            "L1" : "1",
            })

        # NIC DMA TLB
        tlbWrapper = sst.Component(prefix+".nicDmaTlb", "mmu.tlb_wrapper")
        tlb = tlbWrapper.setSubComponent("tlb", "mmu.simpleTLB" );
        tlb.addParams(tlbParams)

        # Cache to CPU interface
        dmaCacheToCpu = dmaCache.setSubComponent("cpulink", "memHierarchy.MemLink")

        # NIC DMA -> TLB
        link = sst.Link(prefix+".link_cpu_dtlb")
        link.connect( (dmaIf, "port", "1ns"), (tlbWrapper, "cpu_if", "1ns") )

        # NIC DMA TLB -> cache
        link = sst.Link(prefix+".link_cpu_l1dcache")
        link.connect( (tlbWrapper, "cache_if", "1ns"), (dmaCacheToCpu, "port", "1ns") )

        # NIC internode interface
        netLink = nic.setSubComponent( "rtrLink", "merlin.linkcontrol" )
        netLink.addParam("link_bw","16GB/s")
        netLink.addParam("input_buf_size","14KB")
        netLink.addParam("output_buf_size","14KB")

        return mmioIf, dmaCache, tlb, (netLink, "rtr_port", '10ns')

class memory_Builder:
    def __init__(self):
        pass

    def build( self, nodeId, numPorts,  group  ):

        self.prefix = 'node' + str(nodeId)
        self.numPorts = numPorts + 1

        self.chiprtr = sst.Component(self.prefix + ".chiprtr", "merlin.hr_router")
        self.chiprtr.addParams({
            "xbar_bw" : "50GB/s",
            "link_bw" : "25GB/s",
            "input_buf_size" : "40KB",
            "output_buf_size" : "40KB",
            "num_ports" : str(self.numPorts + 1),
            "flit_size" : "72B",
            "id" : "0",
            "topology" : "merlin.singlerouter"
            })

        self.chiprtr.setSubComponent("topology","merlin.singlerouter")

        dirctrl = sst.Component(self.prefix + ".dirctrl", "memHierarchy.DirectoryController")
        dirctrl.addParams({
            "coherence_protocol" : coherence_protocol,
            "entry_cache_size" : "1024",
            "addr_range_start" : "0x0",
            "addr_range_end" : "0x7fffffff",
            })
        dirtoMemLink = dirctrl.setSubComponent("memlink", "memHierarchy.MemLink")
        self.connect( "Dirctrl", numPorts, dirctrl, group, linkType="cpulink" )

        memctrl = sst.Component(self.prefix + ".memory", "memHierarchy.MemController")
        memctrl.addParams({
            "clock" : cpu_clock,
            "backend.mem_size" : physMemSize,
            "backing" : "malloc",
            "initBacking" : 1,
            "addr_range_start" : "0x0",
            "addr_range_end" : "0x7fffffff",
            })

        memToDir = memctrl.setSubComponent("cpulink", "memHierarchy.MemLink")

        memory = memctrl.setSubComponent("backend", "memHierarchy.simpleMem")
        memory.addParams({
            "mem_size" : physMemSize,
            "access_time" : "1 ns",
            })

        link = sst.Link(self.prefix + ".link_dir_mem")
        link.connect( (dirtoMemLink, "port", "1ns"), (memToDir, "port", "1ns") )

    def connect( self, name, port, comp, group=None, linkType="memlink"  ):

        assert group
        assert port < self.numPorts

        config="{} MemNIC config: groupt={} ".format(name,group)

        memNIC = comp.setSubComponent(linkType, "memHierarchy.MemNIC")
        memNIC.addParams({
            "group" : group,
            "network_bw" : "25GB/s",
            })

        link = sst.Link(self.prefix + ".link_rtr" + str(port) )
        link.connect( (self.chiprtr, "port" + str(port), "1ns"), (memNIC, "port", "1ns") )


class Endpoint():
    def __init__(self,numNodes):
        self.numNodes = numNodes

    def prepParams(self):
        pass

    def build(self, nodeId, extraKeys ):

        prefix = 'node' + str(nodeId);

        cpuBuilder = vanadis_Builder()
        memBuilder = memory_Builder()
        osBuilder = vanadisOS_Builder()
        nicBuilder = rdmaNic_Builder(self.numNodes)

        numPorts = 4
        port = 0
        memBuilder.build(nodeId, numPorts, group=2 )

        # build the Vanadis OS, it returns
        osCache = osBuilder.build( self.numNodes, nodeId, 0 )

        # connect OS L1 to Memory
        #memBuilder.connect( "OS_L1", port, osCache, 1, dest="2" )
        memBuilder.connect( "OS_L1", port, osCache, group=1 )
        port += 1;

        # build the Vanadis CPU block, this returns
        # cpu, L2 cache, DTLB ITLB
        cpu, L2, dtlb, itlb = cpuBuilder.build(nodeId,0)

        osBuilder.connectCPU( 0, cpu )
        osBuilder.connectTlb( 0, "dtlb", dtlb )
        osBuilder.connectTlb( 0, "itlb", itlb )

        # connect CPU L2 to Memory
        #memBuilder.connect( "CPU_L2", port, L2, 1, dest="2,3" )
        memBuilder.connect( "CPU_L2", port, L2, group=1 )
        port += 1;

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

topo = topoTorus()
topo.prepParams()
topo.setEndPointFunc( setNode )
topo.build()

