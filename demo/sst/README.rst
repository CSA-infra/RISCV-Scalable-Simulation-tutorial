Scale-out system simulation with SST
====================================

**How to perform a scale-out system simulation with cycle-approximate accuracy?**
The goal of the second part of this tutorial is to introduce the Structural Simulation
Toolkit (SST) framework which allows to simulate a scale-out system with a
cycle-approximate accuracy.

Environment Setup
-----------------

To run the SST experiments you need to install SST. Please refer to `Installation instructions`_.


Context
-------


System under exploration
~~~~~~~~~~~~~~~~~~~~~~~~
.. _cpu figure:

.. figure:: images/sst/cpu.svg
   :width: 400
   :align: center

   Microarchitecture of a cpu core.


The system under exploration is made up of multi-threaded RISC-V CPU cores. As illustrated
in Figure :numref:`cpu figure`, a CPU core is attached to an L1 data cache and an L1
instruction cache. The two caches are interconnect to a second level of cache (L2 cache)
with a memory bus. The core itself is composed of one decoder for each thread, one branch
unit and one dispatch unit, one register file for floating point numbers and another one
for integers, a load store unit (or load store queue), multiple ALU and multiple FPU. The
core is attached to each cache through a TLB and a memory interface. TLBs are managed by
the operating system.


.. _node figure:

.. figure:: images/sst/node.svg
   :width: 600
   :align: center

   Microarchitecture of a compute node.

As shown in Figure :numref:`node figure`, the RISC-V cores are integrated into a compute
node. The number of cores per node is configurable from the script. The set of L2 caches
are federated with a directory which maintains coherency in the node. The L2s and the
directory are interconnected through a NoC. The directory is attached to a DRAM
controller. In addition, a node integrates a component that emulates an operating systems.
The latter manages the virtual memory and is attached to every CPU core to provide the
minimal service required to run applications.

.. _system figure:

.. figure:: images/sst/system.svg
   :width: 800
   :align: center
   :alt: Scale-out system microarchitecture

   Microarchitecture of a multi-node system.

Multi-node can be interconnect with a network to build a scale-out system, as illustrated
in Figure :numref:`system figure`. Each node has an independent operating system and a
private memory space. To allow communication between node, we can use
Message Passing Interface (MPI). To do that, each node integrates a NIC in addition. The
latter is interconnected to the NoC.

The inter-node network is built with pymerlin (a python script provided in SST-elements).
Thanks to that script we can defined different topologies easily (e.g., single router, fat
tree, dragonfly, torus, mesh, torus, etc).


Every components or sub-components are configurable, for instance you can configure the
latency of the ALU or the capacity of each cache. You can find more information on the
parameters and their impact on the simulated system using **sst-info** command.

.. list-table:: How to find the available parameters
   :widths: 25 50
   :header-rows: 1

   * - Command
     - Description
   * - sst-info vanadis
     - Parameters of the cpu core and the operating system
   * - sst-info mmu
     - Parameters of the TBL and MMU
   * - sst-info sst-info memHierarchy
     - Parameters of the cache, directory controller, DRAM, memory bus
   * - sst-info merlin
     - Parameters of the NoC and internode network components
   * - sst-info sst-info rdmaNic
     - Parameters of the NIC


Workload under evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~


The workload under evaluation is inspired by a Multi-head attention, one of the
calculation layers of transformers :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.

.. _OMP_MHA figure:

.. figure:: images/sst/mha.svg
   :width: 600
   :align: center
   :alt: Multi-head attention block

   Illustration of the workload run on a single-node system.

As shown in Figure :numref:`OMP_MHA figure`, the application multiplies an *Embeddings*
matrix of Seq\ :sub:`len`\ x D\ :sub:`model` \ elements with 3 matrices of
D\ :sub:`model` x D\ :sub:`model` weights, producing 3 matrices of  Seq\ :sub:`len`\ x D\ :sub:`model` \ elements,
called Keys, Queries and Values. In fact, the weight matrices are divided into *heads*.
Each head of Queries matrix are multiplied with the corresponding transposed head of Keys
matrix, producing *QK* matrix. The latter is then scaled. Then the *softmax* of each row of
the scaled *QK* is computed. Afterward, the result of the *softmax* is multiplied with
Values matrix, producing *QKV* matrix. Finally, *QKV* is summed with the *Embeddings*
matrix.


.. _mha_OMP: ../../demo/sst/software/mha_OMP.c

The corresponding code is implemented in **C** `mha_OMP`_, and is parallelized with **OpenMP**.


Matrix-Matrix multiplication is the heaviest workload in this application. To minimize the
data movement, a tiled GEMM is implemented. *TILE_SIZE* macro defines the dimension of the
tiles.

.. code-block:: C
   :linenos:
   :emphasize-lines: 1

   const int bsize = TILE_SIZE;
   int ii0, ii1, ii2;
   int i0, i1, i2;
   int h;
   int start_head, stop_head;
   data_t pp;
   #pragma omp parallel for shared (dst, src1, src2) private(h,i0,i1,i2,ii0,ii1,ii2,pp) collapse(2)
   for (h=0; h < heads; h++) {
      for (ii0 = 0; ii0<m; ii0+=bsize) {
         for (ii1 = h*n; ii1<((h+1)*n); ii1+=bsize) {
            for(ii2 = 0; ii2<k; ii2+=bsize) {
               for (i0 = ii0; i0 < MIN(ii0+bsize,m); i0++) {
                  for (i1 = ii1; i1 < MIN(ii1+bsize,((h+1)*n)); i1++) {
                     pp = 0;
                     for (i2 = ii2; i2 < MIN(ii2+bsize,k); i2++) {
                        pp += src1[(i0+h*m)*(stride_1)+i2] * src2[i2*stride_2+i1];
                     }
                     dst[i0*(stride_0)+i1]+= pp;
                  }
               }
            }
         }
      }
   }

.. _OMP_MPI_MHA figure:

.. figure:: images/sst/mha_mpi.svg
   :width: 600
   :align: center
   :alt: Multi-head attention block

   Illustration of the workload run on a multi-node system.

.. _mha_mpi_OMP: ../../demo/sst/software/mha_MPI_OMP.c

As the *heads* can be processed independently until the addition, the workload can be
easily parallelized on a distributed memory system. As illustrated in Figure :numref:`OMP_MPI_MHA figure`,
running the workload on a multi-node system requires only a few extra steps.
The corresponding application is implemented with MPI to handle the communication between
the nodes and OpenMP to parallelize the kernels within a node. The code is written in
**C** as well `mha_mpi_OMP`_


.. code-block:: C
   :emphasize-lines: 24-25, 38-40, 64
   :linenos:

   MPI_Init(&argc, &argv);
   MPI_Comm_size(WORLD, &n_ranks);
   MPI_Comm_rank(WORLD, &rank);
   MPI_Datatype col, col_type;

   /*
    ...
    */

   MPI_Type_vector(dmodel, dmodel/n_ranks, dmodel, mpi_data_type, &col);
   MPI_Type_commit(&col);
   MPI_Type_create_resized(col, 0, dmodel/n_ranks*sizeof(data_t), &col_type);
   MPI_Type_commit(&col_type);

   /*
    ...
    */

   if(rank == root) {
      init_random_tensor(embeddings, data_type, dmodel*S);
      init_random_tensor(ATTNw, data_type, dmodel*dmodel);
   }

   MPI_Bcast(embeddings, dmodel*S, mpi_data_type, root, WORLD);
   MPI_Bcast(ATTNw, dmodel*dmodel, mpi_data_type, root, WORLD);

   if(rank == root) {
      Qw = calloc(dmodel*dmodel, sizeof(data_t));
      init_random_tensor(Qw, data_type, dmodel*dmodel);

      Kw = calloc(dmodel*dmodel, sizeof(data_t));
      init_random_tensor(Kw, data_type, dmodel*dmodel);

      Vw = calloc(dmodel*dmodel, sizeof(data_t));
      init_random_tensor(Vw, data_type, dmodel*dmodel);
   }

   MPI_Scatter(Qw, 1, col_type, Qw_heads, dmodel*dmodel/n_ranks, mpi_data_type, root, WORLD);
   MPI_Scatter(Kw, 1, col_type, Kw_heads, dmodel*dmodel/n_ranks, mpi_data_type, root, WORLD);
   MPI_Scatter(Vw, 1, col_type, Vw_heads, dmodel*dmodel/n_ranks, mpi_data_type, root, WORLD);

   /*
    ...
    */

   /* MHA */

   gemm(Q, embeddings, Qw_heads, data_type, 1, S, dmodel/n_ranks, dmodel, dmodel/n_ranks, dmodel, dmodel/n_ranks);
   gemm(K, embeddings, Kw_heads, data_type, 1, S, dmodel/n_ranks, dmodel, dmodel/n_ranks, dmodel, dmodel/n_ranks);
   gemm(V, embeddings, Vw_heads, data_type, 1, S, dmodel/n_ranks, dmodel, dmodel/n_ranks, dmodel, dmodel/n_ranks);

   gemm_t(KQ, Q, K, data_type, h/n_ranks, S, S, dmodel/h, S, dmodel/n_ranks, dmodel/n_ranks);

   scale(KQ, KQ, ((void*)&scale_f), data_type, h/n_ranks*S, S);

   softmax(softmax_out, KQ, data_type, h/n_ranks*S, S);

   gemm(QKV, softmax_out, V, data_type, h/n_ranks, S, dmodel/h, S, dmodel/n_ranks, S, dmodel/n_ranks);

   gemm(ATTNout, QKV, ATTNw, data_type, 1, S, dmodel, dmodel/n_ranks, dmodel, dmodel/n_ranks, dmodel);

   add(&ATTNout[S/n_ranks*rank*dmodel], &ATTNout[S/n_ranks*rank*dmodel], &embeddings[S/n_ranks*rank*dmodel], data_type, S/n_ranks, dmodel);

   MPI_Allreduce(ATTNout, embeddings, S*dmodel, mpi_data_type, MPI_SUM, WORLD);

   /*
    ...
    */

   MPI_Finalize();




Firstly, the *Embeddings* matrix needs to be locally stored in every memory space. To do that we
use a broadcast. Every node produces different heads, hence only the required weights are
stored in each memory domain (**scatter**). Consequently, less computation are required.
After, the final addition, we need to gather the heads by executing a **MPI ALL REDUCE**,
after that all the nodes have the *Output* result.



DEMO
----

For the demo, we will explore two systems. The first is a single-node system, the second
is a scale-out system.

Scale-up system
~~~~~~~~~~~~~~~
.. _scale_up: ../../demo/sst/instruction-level-simulation/scale_up.py

The python script `scale_up`_ build the system for the scale up system. You can explore
the script to understand how a system is built with SST.

You can run a simulation by executing the following command in a terminal from
*demo/sst/ssytem* folder:

.. code:: bash

   sst scale_up.py -- --stats

You can also store the statistics in a csv file by passing a file name:

.. code:: bash

   sst scale_up.py -- --stats stats.csv

You can configure the number of threads, the number of CPU, the dimensions of the
workload (Seq\ :sub:`len`\, D\ :sub:`model` \, heads), and the binary version from the command line:

.. code:: bash

   sst scale_up.py -- --num_cpu_per_node 2 --num_threads_per_cpu 2 --app_args "64 128 8"
   --exe "../software/riscv64/mha_OMP_8"

First experiment: Impact of tiling dimension on performance
###########################################################

For the first experiment, we will evaluate the impact of the GEMM tiles dimension on the
**simulated performance**. 4 binaries are provided in *software* folder.
You can run a simulation with each binary. To explain the performance difference, you can
use the generated statistics.

.. hint:: Store the stats in a CSV file
   Storing the statistics in a csv file makes analysis easier. You can open the file in
   Excel to filter the stats by component or type.



Second experiment: Scaling evaluation
#####################################

For the second experiment, we will observe the scaling of the **simulated system**
(*i.e.*, performance of the application) and of the simulation (*i.e.*, performance of
SST).

.. admonition:: Pick the correct binary

   Make sure to use the most efficient binary based on the first experiment


.. admonition:: Run the simulations in parallel

   You can run the simulations with 4 threads (--num-threads=4)

.. admonition:: Measuring simulation time

   You can measure the simulation time by enabling --print-timing-info option.

   i.e `sst --print-timing-info scale_up.py ...`


For the performance of the simulated system, you can fill the table below:


+--------------------------+----------------------------------+----------------------------------+----------------------------------+
|                          | 1 CPU                            | 2 CPU                            | 4 CPU                            |
+--------------------------+----------+-----------+-----------+----------+-----------+-----------+----------+-----------+-----------+
|                          | 1 thread | 2 threads | 4 threads | 1 thread | 2 threads | 4 threads | 1 thread | 2 threads | 4 threads |
+==========================+==========+===========+===========+==========+===========+===========+==========+===========+===========+
| Simulated time (ms)      |          |           |           |          |           |           |          |           |           |
+--------------------------+----------+-----------+-----------+----------+-----------+-----------+----------+-----------+-----------+



For the performance of the simulated system, make sure to simulated a system with an intense
activity (*e.g.*, 4 CPU 2 threads). You can fill the table below:

.. admonition:: Calculating the simulation speed in MIPS

   You can get the number of Million of instructions simulated per second by summing the
   number of instructions executed by all the cores, then divided by the simulation time multiplied by one million.



+--------------------------------+----------+-----------+-----------+----------+
| Number of simulation threads   | 1        | 2         | 4         | 8        |
+================================+==========+===========+===========+==========+
| Simulation time (s)            |          |           |           |          |
+--------------------------------+----------+-----------+-----------+----------+
| Million of instr. per second   |          |           |           |          |
+--------------------------------+----------+-----------+-----------+----------+

Scale-out system
~~~~~~~~~~~~~~~~
.. _scale_out: ../../demo/sst/instruction-level-simulation/scale_out.py

The python script `scale_out`_ build the system for the scale out system. You can explore
the script to understand how a system is built with SST.

You can run a simulation by executing the following command in a terminal from
*demo/sst/ssytem* folder:

.. code:: bash

   sst scale_out.py

You can also run the simulation in parallel with MPI:

.. code:: bash

   mpirun -np 4 sst scale_out.py


By default, the inter-node network instantiates a simple topology (single router).
You can configure the number of node in the system from the command line by setting num_node_per_router argument:

.. code:: bash

   sst scale_out.py -- --num_node_per_router=4


First experiment: Changing the inter-node network topology
##########################################################

.. literalinclude:: ../demo/sst/instruction-level-simulation/scale_out.py
   :language: python
   :linenos:
   :lineno-start: 35
   :start-at: Network topology definition start
   :end-at: Network topology definition end


You can change the network topology by editing the python script from line 35 to 48.


To use a **torus** topology, you need to comment the line 38 and uncomment the lines 40 to
42. *torus_width* defines the number of link between two routers. *torus_shape* defines
the shape of the network: the size of the array defines the number of dimensions (i.e. 2
elements means a 2D torus, 3 elements a 3D torus) and each element defines the number of
router per dimension. The number of instantiated nodes is equal to the total number of
routers times *num_node_per_router*.


To use a **fat tree** topology, you need to comment the line 38 and uncomment the lines
44 to 46. *fattree_shape* defines the shape of the network.


Second experiment: Scaling evaluation
#####################################


For the last experiment, we will observe the scaling of the **simulated system**
(*i.e.*, performance of the application) and of the simulation (*i.e.*, performance of
SST).

The objective is to observe the scaling of the simulated system to define the expectation
for scaling of the simulation. Ideally, we would like to observe the simulation time
decreasing with the simulated time.

+--------------------------------+----------+-----------+-----------+----------+
| Number of node & MPI ranks     | 1        | 2         | 4         | 8        |
+================================+==========+===========+===========+==========+
| Simulated time (ms)            |          |           |           |          |
+--------------------------------+----------+-----------+-----------+----------+
| Simulation time (s)            |          |           |           |          |
+--------------------------------+----------+-----------+-----------+----------+

References
----------

.. bibliography::
