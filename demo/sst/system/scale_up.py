import sst
import os
import sys
from sst.merlin import *


enableStats = True
sst.setStatisticLoadLevel(10)
if len(sys.argv) > 1:
    sst.setStatisticOutput("sst.statOutputCSV",
                       {   "filepath" : sys.argv[1],
                        "separator" : ";"
                        } )
else:
    sst.setStatisticOutput("sst.statOutputConsole")

num_threads_per_cpu = 2
num_cpu_per_node = 2
app_args = "128 128 2"

cpu_clock = "3GHz"

coherence_protocol="MESI"
cache_line_size = 64

l2cache_size = 1 * 1024**2 # 1MiB
page_size = 4096
memsize = 4 * 1024**3 # 4GiB
physMemSize = str(memsize) + " B"


full_exe_name = "../software/riscv64/mha_OMP_16"
exe_name= full_exe_name.split("/")[-1]

tlbParams = {
        "hitLatency": 3,
        "num_tlb_entries_per_thread": 64,
        "tlb_set_size": 4,
        "minVirtAddr" : 0x1000,
        "maxVirtAddr" : memsize
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
        "useNicTlb":  False,
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
        link.connect( (l1dcache_2_l2cache, "port", "1ns"), (cache_bus, "high_network_0", "1ns") )

        # L1 D-Cache to bus
        link = sst.Link(prefix + ".link_l1icache_l2cache")
        link.connect( (l1icache_2_l2cache, "port", "1ns"), (cache_bus, "high_network_1", "1ns") )

        # BUS to L2 cache
        link = sst.Link(prefix+".link_bus_l2cache")
        link.connect( (cache_bus, "low_network_0", "1ns"), (l2cache_2_cpu, "port", "1ns") )

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
                    "env_count" : 3,
                    "env0" : "OMP_NUM_THREADS={}".format(num_cpu_per_node*num_threads_per_cpu),
                    "env1" : "TZ=UTC",
                    "env2" : "MV2_ENABLE_AFFINITY=0",
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
        self.connect( "Dirctrl", self.numPorts -1 , dirctrl, group, linkType="cpulink" )
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


class node_Builder():
    def __init__(self):
        pass

    def prepParams(self):
        pass

    def build(self, nodeId, extraKeys ):

        prefix = 'node' + str(nodeId);

        cpuBuilder = CPU_Builder()
        memBuilder = memory_Builder()
        osBuilder = OS_Builder()

        numPorts = 1  + num_cpu_per_node
        port = 0
        memBuilder.build(nodeId, numPorts, group=2 )

        # build the Vanadis OS, it returns
        osCache = osBuilder.build( 1, nodeId)

        # connect OS L1 to Memory
        #memBuilder.connect( "OS_L1", port, osCache, 1, dest="2" )
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
            #memBuilder.connect( "CPU_L2", port, L2, 1, dest="2,3" )
            memBuilder.connect( "CPU_L2", port, L2, group=1 )
            port += 1;

nodeBuilder = node_Builder()

nodeBuilder.build(0,{})
