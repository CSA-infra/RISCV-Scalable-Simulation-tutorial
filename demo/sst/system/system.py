import os
import sst

verbosity = 1
debug_level=10
debug=1



# Tell SST what statistics handling we want
enableStats = False
sst.setStatisticLoadLevel(4)
sst.setStatisticOutput("sst.statOutputConsole")

app_args = "32 32 2"
full_exe_name = "../software/riscv64/mha_MPI_OMP"
exe_name= full_exe_name.split("/")[-1]

cpu_clock = "3GHz"
num_node = 2
num_cpu_per_node = 2
num_threads_per_cpu = 1

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


protocol="MESI"
page_size = 4096
cache_line_size = 64

l2_cache_size = "1MiB"
entry_cache_size = 16384 * num_cpu_per_node

memory_per_node = "4GiB"
addr_range_end = 4*1024*1024*1024 - 1

networkParams = {
    "packetSize" : "2048B",
    "link_bw" : "16GB/s",
    "xbar_bw" : "16GB/s",
    "link_lat" : "10ns",
    "input_latency" : "10ns",
    "output_latency" : "10ns",
    "flitSize" : "8B",
    "input_buf_size" : "14KB",
    "output_buf_size" : "14KB",
}

# OS related params
osParams = {
    "processDebugLevel" : 0,
    "dbgLevel" : verbosity,
    "dbgMask" : 8,
    "cores" : num_cpu_per_node,
    "hardwareThreadCount" : num_threads_per_cpu,
    "page_size"  : page_size,
    "physMemSize" : memory_per_node,
    "useMMU" : True,
}

osl1cacheParams = {
    "access_latency_cycles" : 1,
    "cache_frequency" : cpu_clock,
    "replacement_policy" : "lru",
    "coherence_protocol" : protocol,
    "associativity" : "8",
    "cache_line_size" : cache_line_size,
    "cache_size" : "32 KiB",
    "L1" : "1",
    "debug" : debug,
    "debug_level" : debug_level,
}

mmuParams = {
    "debug_level": 0,
    "num_cores": num_cpu_per_node,
    "num_threads": num_threads_per_cpu,
    "page_size": page_size,
}

NodeRtrParams ={
      "xbar_bw" : "256GB/s",
      "link_bw" : "256GB/s",
      "input_buf_size" : "2KB",
      "num_ports" : str(num_cpu_per_node+2 + 2 if num_node > 1 else 0),
      "flit_size" : "72B",
      "output_buf_size" : "2KB",
      "id" : "0",
      "topology" : "merlin.singlerouter"
}

dirCtrlParams = {
      "coherence_protocol" : protocol,
      "entry_cache_size" : entry_cache_size,
      "debug" : debug,
      "debug_level" : debug_level,
      "addr_range_start" : 0x0,
      "addr_range_end" : addr_range_end
}

dirNicParams = {
      "network_bw" : "512GB/s",
      "group" : 2,
}

memCtrlParams = {
      "clock" : cpu_clock,
      "backend.mem_size" : memory_per_node,
      "addr_range_start": 0x0,
      "addr_range_end": addr_range_end,
      "debug_level" : debug_level,
      "debug" : debug
}

memParams = {
      "mem_size" : memory_per_node,
      "access_time" : "1 ns"
}

# CPU related params
tlbParams = {
    "debug_level": 0,
    "hitLatency": 1,
    "num_hardware_threads": num_threads_per_cpu,
    "num_tlb_entries_per_thread": 64,
    "tlb_set_size": 4,
}

decoderParams = {
    "loader_mode" : "1",
    "uop_cache_entries" : 1536,
    "predecode_cache_entries" : 4
}

branchPredParams = {
    "branch_entries" : 32
}

cpuParams = {
    "clock" : cpu_clock,
    "verbose" : verbosity,
    "dbg_mask" : debug_level,
    "hardware_threads": num_threads_per_cpu,
    "physical_fp_registers" : 168 * num_threads_per_cpu,
    "physical_integer_registers" : 180 * num_threads_per_cpu,
    "integer_arith_cycles" : 2,
    "integer_arith_units" : 2,
    "fp_arith_cycles" : 4,
    "fp_arith_units" : 2,
    "branch_unit_cycles" : 2,
    "reorder_slots" : 64,
    "decodes_per_cycle" : 4,
    "issues_per_cycle" :  4,
    "retires_per_cycle" : 4,
}

lsqParams = {
    "verbose" : verbosity,
    "address_mask" : 0xFFFFFFFF,
    "max_stores" : 8,
    "max_loads" : 16,
}

l1dcacheParams = {
    "access_latency_cycles" : "2",
    "cache_frequency" : cpu_clock,
    "replacement_policy" : "lru",
    "coherence_protocol" : protocol,
    "associativity" : "8",
    "cache_line_size" : cache_line_size,
    "cache_size" : "64 KiB",
    "L1" : "1",
    "debug" : debug,
    "debug_level" : debug_level,
}

l1icacheParams = {
    "access_latency_cycles" : "2",
    "cache_frequency" : cpu_clock,
    "replacement_policy" : "lru",
    "coherence_protocol" : protocol,
    "associativity" : "8",
    "cache_line_size" : cache_line_size,
    "cache_size" : "32 KiB",
    "prefetcher" : "cassini.NextBlockPrefetcher",
    "prefetcher.reach" : 1,
    "L1" : "1",
    "debug" : debug,
    "debug_level" : debug_level,
}

l2cacheParams = {
    "access_latency_cycles" : "14",
    "cache_frequency" : cpu_clock,
    "replacement_policy" : "lru",
    "coherence_protocol" : protocol,
    "associativity" : "16",
    "cache_line_size" : cache_line_size,
    "cache_size" : l2_cache_size,
    "mshr_latency_cycles": 3,
    "debug" : debug,
    "debug_level" : debug_level,
}
busParams = {
    "bus_frequency" : cpu_clock,
}

l2memLinkParams = {
    "group" : 1,
    "network_bw" : "512GB/s"
}


NiCParams = {
    "clock" : "1GHz",
    "debug_level": debug_level,
    "useDmaCache": "true",
    "debug_mask": -1,
    "maxPendingCmds" : 128,
    "maxMemReqs" : 256,
    "maxCmdQSize" : 128,
    "cache_line_size"    : cache_line_size,
    "baseAddr": addr_range_end + 1,
    "cmdQSize" : 64,
}

dmaCacheParams = {
    "access_latency_cycles" : "2",
    "cache_frequency" : cpu_clock,
    "replacement_policy" : "lru",
    "coherence_protocol" : protocol,
    "associativity" : "8",
    "cache_line_size" : cache_line_size,
    "cache_size" : "32KiB",
    "debug_level" : debug_level,
}

def addParamsPrefix(prefix,params):
    #print( prefix )
    ret = {}
    for key, value in params.items():
        #print( key, value )
        ret[ prefix + "." + key] = value

    #print( ret )
    return ret

class CPU_Builder:
    def __init__(self):
        pass

    # CPU
    def build( self, prefix, nodeId, cpuId ):

        # CPU
        cpu = sst.Component(prefix, "vanadis.VanadisCPU")
        cpu.addParams( cpuParams )
        cpu.addParam( "core_id", cpuId )
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

        # CPU.lsq mem interface which connects to D-cache
        cpuDcacheIf = cpu_lsq.setSubComponent( "memory_interface", "memHierarchy.standardInterface" )

        # CPU.mem interface for I-cache
        cpuIcacheIf = cpu.setSubComponent( "mem_interface_inst", "memHierarchy.standardInterface" )

        # L1 D-cache
        cpu_l1dcache = sst.Component(prefix + ".l1dcache", "memHierarchy.Cache")
        cpu_l1dcache.addParams( l1dcacheParams )
        if enableStats:
            cpu_l1dcache.enableAllStatistics()

        # L1 I-cache to cpu interface
        l1dcache_2_cpu     = cpu_l1dcache.setSubComponent("cpulink", "memHierarchy.MemLink")
        # L1 I-cache to L2 interface
        l1dcache_2_l2cache = cpu_l1dcache.setSubComponent("memlink", "memHierarchy.MemLink")

        # L2 I-cache
        cpu_l1icache = sst.Component( prefix + ".l1icache", "memHierarchy.Cache")
        cpu_l1icache.addParams( l1icacheParams )
        if enableStats:
            cpu_l1icache.enableAllStatistics()

        # L1 I-iache to cpu interface
        l1icache_2_cpu     = cpu_l1icache.setSubComponent("cpulink", "memHierarchy.MemLink")
        # L1 I-cache to L2 interface
        l1icache_2_l2cache = cpu_l1icache.setSubComponent("memlink", "memHierarchy.MemLink")

        # L2 cache
        cpu_l2cache = sst.Component(prefix+".l2cache", "memHierarchy.Cache")
        cpu_l2cache.addParams( l2cacheParams )
        if enableStats:
            cpu_l2cache.enableStats()

        # L2 cache cpu interface
        l2cache_2_l1caches = cpu_l2cache.setSubComponent("cpulink", "memHierarchy.MemLink")

        # L2 cache mem interface
        l2cache_2_mem = cpu_l2cache.setSubComponent("memlink", "memHierarchy.MemNIC")
        l2cache_2_mem.addParams( l2memLinkParams )
        if enableStats:
            l2cache_2_mem.enableAllStatistics()

        # L1 to L2 buss
        cache_bus = sst.Component(prefix+".bus", "memHierarchy.Bus")
        cache_bus.addParams(busParams)
        if enableStats:
            cache_bus.enableStats()


        # CPU data TLB
        dtlbWrapper = sst.Component(prefix+".dtlb", "mmu.tlb_wrapper")
        dtlb = dtlbWrapper.setSubComponent("tlb", "mmu." + "simpleTLB" );
        dtlb.addParams(tlbParams)

        # CPU instruction TLB
        itlbWrapper = sst.Component(prefix+".itlb", "mmu.tlb_wrapper")
        itlbWrapper.addParam("exe",True)
        itlb = itlbWrapper.setSubComponent("tlb", "mmu." + "simpleTLB" );
        itlb.addParams(tlbParams)

        # CPU (data) -> TLB -> Cache
        link_cpu_dtlb_link = sst.Link(prefix+".link_cpu_dtlb_link")
        link_cpu_dtlb_link.connect( (cpuDcacheIf, "port", "1ns"), (dtlbWrapper, "cpu_if", "1ns") )
        link_cpu_dtlb_link.setNoCut()

        # data TLB -> data L1
        link_cpu_l1dcache_link = sst.Link(prefix+".link_cpu_l1dcache_link")
        link_cpu_l1dcache_link.connect( (dtlbWrapper, "cache_if", "1ns"), (l1dcache_2_cpu, "port", "1ns") )
        link_cpu_l1dcache_link.setNoCut()

        # CPU (instruction) -> TLB -> Cache
        link_cpu_itlb_link = sst.Link(prefix+".link_cpu_itlb_link")
        link_cpu_itlb_link.connect( (cpuIcacheIf, "port", "1ns"), (itlbWrapper, "cpu_if", "1ns") )
        link_cpu_itlb_link.setNoCut()

        # instruction TLB -> instruction L1
        link_cpu_l1icache_link = sst.Link(prefix+".link_cpu_l1icache_link")
        link_cpu_l1icache_link.connect( (itlbWrapper, "cache_if", "1ns"), (l1icache_2_cpu, "port", "1ns") )
        link_cpu_l1icache_link.setNoCut();

        # data L1 -> bus
        link_l1dcache_l2cache_link = sst.Link(prefix+".link_l1dcache_l2cache_link")
        link_l1dcache_l2cache_link.connect( (l1dcache_2_l2cache, "port", "1ns"), (cache_bus, "high_network_0", "1ns") )
        link_l1dcache_l2cache_link.setNoCut()

        # instruction L1 -> bus
        link_l1icache_l2cache_link = sst.Link(prefix+".link_l1icache_l2cache_link")
        link_l1icache_l2cache_link.connect( (l1icache_2_l2cache, "port", "1ns"), (cache_bus, "high_network_1", "1ns") )
        link_l1icache_l2cache_link.setNoCut()

        # BUS to L2 cache
        link_bus_l2cache_link = sst.Link(prefix+".link_bus_l2cache_link")
        link_bus_l2cache_link.connect( (cache_bus, "low_network_0", "1ns"), (l2cache_2_l1caches, "port", "1ns") )
        link_bus_l2cache_link.setNoCut()

        return (cpu, "os_link", "5ns"), (l2cache_2_mem, "port", "1ns") , (dtlb, "mmu", "1ns"), (itlb, "mmu", "1ns")



class Node_Builder:
    def __init__(self):
        pass

    def build(self, nodeId, dummy):
        node_prefix = "Node" + str(nodeId)

        # node OS
        node_os = sst.Component(node_prefix + ".os", "vanadis.VanadisNodeOS")
        node_os.addParams(osParams)
        if enableStats:
            node_os.enableAllStatistics()


        processList = (
            ( 1, {
                "env_count" : 5,
                "env0" : "OMP_NUM_THREADS={}".format(num_cpu_per_node*num_threads_per_cpu),
                "env1" : "PMI_SIZE={}".format(num_node),
                "env2" : "PMI_RANK={}".format(nodeId),
                "env3" : "RDMA_NIC_NUM_POSTED_RECV={}".format(128),
                "env4" : "RDMA_NIC_COMP_Q_SIZE={}".format(256),
                "exe" : full_exe_name,
                "arg0" : exe_name,
            } ),
        )

        processList[0][1].update(app_params)

        num=0
        for i,process in processList:
            for y in range(i):
                node_os.addParams( addParamsPrefix( "process" + str(num), process ) )
                num+=1

        # node OS MMU
        node_os_mmu = node_os.setSubComponent( "mmu", "mmu." + "simpleMMU" )
        node_os_mmu.addParams(mmuParams)

        # node OS memory interface to L1 data cache
        node_os_mem_if = node_os.setSubComponent( "mem_interface", "memHierarchy.standardInterface" )

        # node OS l1 data cache
        os_cache = sst.Component(node_prefix + ".os.cache", "memHierarchy.Cache")
        os_cache.addParams(osl1cacheParams)
        os_cache_2_cpu = os_cache.setSubComponent("cpulink", "memHierarchy.MemLink")
        os_cache_2_mem = os_cache.setSubComponent("memlink", "memHierarchy.MemNIC")
        os_cache_2_mem.addParams( l2memLinkParams )

        # node router
        node_rtr_port_count = 0
        node_rtr = sst.Component(node_prefix + ".node_rtr", "merlin.hr_router")
        node_rtr.addParams(NodeRtrParams)
        node_rtr.setSubComponent("topology","merlin.singlerouter")
        if enableStats:
            node_rtr.enableAllStatistics()

        # node directory controller
        dirctrl = sst.Component(node_prefix + ".dirctrl", "memHierarchy.DirectoryController")
        dirctrl.addParams(dirCtrlParams)
        if enableStats:
            dirctrl.enableStats()

        # node directory controller port to memory
        dirtoM = dirctrl.setSubComponent("memlink", "memHierarchy.MemLink")
        # node directory controller port to cpu
        dirNIC = dirctrl.setSubComponent("cpulink", "memHierarchy.MemNIC")
        dirNIC.addParams(dirNicParams)

        # node memory controller
        memctrl = sst.Component(node_prefix + ".memory", "memHierarchy.MemController")
        memctrl.addParams( memCtrlParams )

        # node memory controller port to directory controller
        memToDir = memctrl.setSubComponent("cpulink", "memHierarchy.MemLink")

        # node memory controller backend
        memory = memctrl.setSubComponent("backend", "memHierarchy.simpleMem")
        memory.addParams(memParams)

        # Directory controller to memory router
        link_dir_2_rtr = sst.Link(node_prefix + ".link_dir_2_rtr")
        link_dir_2_rtr.connect( (node_rtr, "port"+str(node_rtr_port_count), "1ns"), (dirNIC, "port", "1ns") )
        node_rtr_port_count = node_rtr_port_count + 1
        link_dir_2_rtr.setNoCut()

        # Directory controller to memory controller
        link_dir_2_mem = sst.Link(node_prefix + ".link_dir_2_mem")
        link_dir_2_mem.connect( (dirtoM, "port", "1ns"), (memToDir, "port", "1ns") )
        link_dir_2_mem.setNoCut()

        # ostlb -> os l1 cache
        link_os_cache_link = sst.Link(node_prefix + ".link_os_cache_link")
        link_os_cache_link.connect( (node_os_mem_if, "port", "1ns"), (os_cache_2_cpu, "port", "1ns") )
        link_os_cache_link.setNoCut()

        os_cache_2_rtr = sst.Link(node_prefix + ".os_cache_2_rtr")
        os_cache_2_rtr.connect( (os_cache_2_mem, "port", "1ns"), (node_rtr, "port"+str(node_rtr_port_count), "1ns") )
        node_rtr_port_count = node_rtr_port_count + 1
        os_cache_2_rtr.setNoCut()

        cpuBuilder = CPU_Builder()

        # build all CPUs
        for cpu in range(num_cpu_per_node):

            prefix_cpu= node_prefix + ".cpu" + str(cpu)
            os_hdlr, l2cache, dtlb, itlb = cpuBuilder.build(prefix_cpu, nodeId, cpu)

            # MMU -> dtlb
            link_mmu_dtlb_link = sst.Link(prefix_cpu + ".link_mmu_dtlb_link")
            link_mmu_dtlb_link.connect( (node_os_mmu, "core"+ str(cpu) +".dtlb", "1ns"), dtlb )

            # MMU -> itlb
            link_mmu_itlb_link = sst.Link(prefix_cpu + ".link_mmu_itlb_link")
            link_mmu_itlb_link.connect( (node_os_mmu, "core"+ str(cpu) +".itlb", "1ns"), itlb )

            # CPU os handler -> node OS
            link_core_os_link = sst.Link(prefix_cpu + ".link_core_os_link")
            link_core_os_link.connect( os_hdlr, (node_os, "core" + str(cpu), "5ns") )

            # connect cpu L2 to router
            link_l2cache_2_rtr = sst.Link(prefix_cpu + ".link_l2cache_2_rtr")
            link_l2cache_2_rtr.connect( l2cache, (node_rtr, "port" + str(node_rtr_port_count), "1ns") )
            node_rtr_port_count = node_rtr_port_count + 1

        if num_node > 1:
            nic = sst.Component( node_prefix + ".nic", "rdmaNic.nic")
            nic.addParams(NiCParams)
            nic.addParam( 'nicId', nodeId )
            nic.addParam( 'pesPerNode', 1 )
            nic.addParam( 'numNodes', num_node )

            if enableStats:
                nic.enableAllStatistics()


            # NIC DMA interface
            dmaIf = nic.setSubComponent("dma", "memHierarchy.standardInterface")

            # NIC MMIO interface
            mmioIf = nic.setSubComponent("mmio", "memHierarchy.standardInterface")

            mmioNIC = mmioIf.setSubComponent("memlink", "memHierarchy.MemNIC")
            mmioNIC.addParams({
                "group" : 2,
                "network_bw" : "128GB/s",
            })

            mmioLink = sst.Link(node_prefix + ".link_mmio_rtr")
            mmioLink.connect( (node_rtr, "port" + str(node_rtr_port_count), "1ns"), (mmioNIC, "port", "1ns") )
            node_rtr_port_count = node_rtr_port_count + 1

            # NIC DMA Cache
            dmaCache = sst.Component(node_prefix + ".nicDmaCache", "memHierarchy.Cache")
            dmaCache.addParams(dmaCacheParams)
            if enableStats:
                dmaCache.enableStats()

            dmaNIC = dmaCache.setSubComponent("memlink", "memHierarchy.MemNIC")
            dmaNIC.addParams({
                "group" : 1,
                "network_bw" : "128GB/s",
            })

            dmaLink = sst.Link(node_prefix + ".link_dma_rtr")
            dmaLink.connect( (node_rtr, "port" + str(node_rtr_port_count), "1ns"), (dmaNIC, "port", "1ns") )
            node_rtr_port_count = node_rtr_port_count + 1


            # NIC DMA TLB
            tlbWrapper = sst.Component(node_prefix+".nicDmaTlb", "mmu.tlb_wrapper")
            tlb = tlbWrapper.setSubComponent("tlb", "mmu.simpleTLB" );
            tlb.addParams(tlbParams)

            tlbLink = sst.Link( node_prefix + ".link_mmu_nicDmaTlb" )
            tlbLink.connect( (node_os_mmu, "nicTlb", "1ns"), (tlb, "mmu", "1ns") )

            # Cache to CPU interface
            dmaCacheToCpu = dmaCache.setSubComponent("cpulink", "memHierarchy.MemLink")

            # NIC DMA -> TLB
            link = sst.Link(node_prefix+".link_cpu_dtlb")
            link.connect( (dmaIf, "port", "1ns"), (tlbWrapper, "cpu_if", "1ns") )

            # NIC DMA TLB -> cache
            link = sst.Link(node_prefix+".link_cpu_l1dcache")
            link.connect( (tlbWrapper, "cache_if", "1ns"), (dmaCacheToCpu, "port", "1ns") )

            # NIC internode interface
            netLink = nic.setSubComponent( "rtrLink", "merlin.linkcontrol" )
            netLink.addParam("link_bw","16GB/s")
            netLink.addParam("input_buf_size","14KB")
            netLink.addParam("output_buf_size","14KB")

            return (netLink, "rtr_port", '10ns')


nodeBuilder = Node_Builder()
def setNode( nodeId ):
    return nodeBuilder;

if num_node > 1 :
    from sst.merlin import *

    sst.merlin._params["link_lat"] = networkParams['link_lat']
    sst.merlin._params["link_bw"] = networkParams['link_bw']
    sst.merlin._params["xbar_bw"] = networkParams['xbar_bw']
    sst.merlin._params["flit_size"] = networkParams['flitSize']
    sst.merlin._params["input_latency"] = networkParams['input_latency']
    sst.merlin._params["output_latency"] = networkParams['output_latency']
    sst.merlin._params["input_buf_size"] = networkParams['input_buf_size']
    sst.merlin._params["output_buf_size"] = networkParams['output_buf_size']
    sst.merlin._params["router_radix"] = num_node


    topo = topoSimple()
    topo.prepParams()

    topo.setEndPointFunc( setNode )

    topo.build()

else :
    node = nodeBuilder.build(0, {})
