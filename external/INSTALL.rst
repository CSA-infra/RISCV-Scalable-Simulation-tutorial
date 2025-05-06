.. _Installation instructions:

Installation instructions for scale-out system simulation
==========================================================

To run the demo on your side you need to install at least SST.
The rv64 binaries are already compiled. However, if you want to compile new applications you must install the mpi compiler as described below.

Run the following command to download the required sub-modules:

.. code:: bash

   git submodule init --update

Install instructions for SST
----------------------------

You must install **SST-core** SST. To do this, run the following commands in a terminal:

.. code:: bash

   cd sst/sst-core
   export SST_CORE_HOME=$(pwd)/install
   ./autogen.sh
   mkdir build
   cd build
   ../configure --prefix=$SST_CORE_HOME
   make -j all
   make install
   export PATH=$SST_CORE_HOME/bin:$PATH
   cd ../../../

Then, you can install **SST-elements** as follow:

.. code:: bash

   cd sst/sst-elements
   git apply ../sst-elements.patch
   export SST_ELEMENTS_HOME=$(pwd)/install
   ./autogen.sh
   mkdir build
   cd build
   ../configure --prefix=$SST_ELEMENTS_HOME --with-sst-core=$SST_CORE_HOME
   make -j all
   make install
   cd ../../../

Install instructions for rv64 mpi compiler
------------------------------------------
The first step is to install **riscv64-unknown-linux-musl-gcc**. To do this, run the following commands in a terminal:

.. code:: bash

   cd riscv-gnu-toolchain
   export RV64_GNU_INSTALL=$(pwd)/install
   CFLAGS="-O3 -fPIC" CXXFLAGS="-O3 -fPIC" ./configure --prefix=$RV64_GNU_INSTALL --disable-multilib --with-languages=c,c++
   make -j8 musl

Then, you must build the RDMA library

.. code:: bash

   cd sst/libRDMA
   make

Finally, you can build and install **mpicc** as follow:

.. code:: bash

   export RDMA_NIC_DIR=$(realpath ./sst/sst-elements/src/sst/elements/rdmaNic)
   export RDMA_LIB_DIR=$(realpath ./sst/libRDMA/riscv64/)

   tar xzvf mvapich2-2.3.7-1.tar.gz
   ulimit -n 4096
   patch --directory=mvapich2-2.3.7-1/ -p1 < mvapich2-2.3.7-1.patch

   cd mvapich2-2.3.7-1/
   ./autogen.sh

   mkdir install
   mkdir build

   export MVAPICH2_INSTALL_DIR=$(pwd)/install

   cd build

   ../configure                                                                        \
         --prefix=${MVAPICH2_INSTALL_DIR}                                              \
         --enable-fortran=no                                                           \
         --with-device=ch3:rdma                                                        \
         --enable-romio=no                                                             \
         --enable-hybrid=no                                                            \
         --enable-shared=no                                                            \
         --enable-static=yes                                                           \
         --with-pmi=vanadis                                                            \
         --with-pm=none                                                                \
         --enable-threads=single                                                       \
         --enable-rsh=yes                                                              \
         --host=riscv64-unknown-linux-musl                                             \
         CC=${RV64_GNU_INSTALL}/bin/riscv64-unknown-linux-musl-gcc                     \
         CFLAGS="-I${RDMA_NIC_DIR}/tests/app/rdma/include -I${RDMA_NIC_DIR} -fPIC"     \
         CXX=${RV64_GNU_INSTALL}/bin/riscv64-unknown-linux-musl-g++                    \
         CXXFLAGS="-I${RDMA_NIC_DIR}/tests/app/rdma/include -I${RDMA_NIC_DIR} -fPIC"   \
         LDFLAGS="-L${RDMA_LIB_DIR}"                                                   \
         LIBS=-lrdma

   make -j8 install
