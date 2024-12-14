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
_______

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

As shown in Figure :numref:`node figure`, the RISC-V cores are integrated into a compute node. The number of cores per node is
configurable from the script. The set of L2 caches are federated with a directory which
maintains coherency in the node. The L2s and the directory are interconnected through a
NoC (in that case, a crossbar router). The directory is attached to a DRAM controller. In
addition, a node integrates a component that emulates an operating systems. The latter
manages the virtual memory and is attached to every CPU core to provide the minimal
service required to run applications.

.. _system figure:

.. figure:: images/sst/system.svg
   :width: 800
   :align: center
   :alt: Scale-out system microarchitecture

   Microarchitecture of a multi-node system.

Multi-node can be interconnect with a network to build a scale-out system, as illustrated
in Figure :numref:`system figure`. In that case
each node has a private operating system and memory space. To allow communication between
node, we can use Message Passing Interface (MPI). To do that, each node integrates a NIC
in addition. The latter is interconnected to the NoC.

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
matrix, producing *QK* matrix. The latter is then scaled. Then the *soft max* of each row of
the scaled *QK* is computed. Afterward, the result of the *softmax* is multiplied with
Values matrix, producing *QKV* matrix. Finally, *QKV* is summed with the *Embeddings*
matrix.


.. _mha_OMP: ../../demo/sst/software/mha_OMP.c

The corresponding code is implemented in **C** `mha_OMP`_, and is parallelized with **OpenMP**.



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
Firstly, the *Embeddings* matrix needs to be locally stored in every memory space. To do that we
use a broadcast. Every node produces different heads, hence only the required weights are
stored in each memory domain. Consequently, less computation are required. After, the
final addition, we need to gather the heads by executing a **MPI ALL REDUCE**, after that
all the nodes have the *Output* result.



DEMO
____

For the demo, we will explore two systems. The first is a single-node system, the second
is a scale-out system.

Scale-up system
~~~~~~~~~~~~~~~
.. _scale_up: ../../demo/sst/system/scale_up.py

The python script `scale_up`_ build the system for the scale up system. You can explore
the script to understand how a system is built with SST.

You can run a simulation by executing the following command in a terminal from
*demo/sst/ssytem* folder:

.. code:: bash

   sst scale_up.py




Scale-out system
~~~~~~~~~~~~~~~~


.. bibliography::
