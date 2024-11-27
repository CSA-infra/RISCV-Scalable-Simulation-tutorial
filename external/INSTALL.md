# Install external tools

## How to install SST

[Installation instructions](sst/INSTALL.md)

## How to install riscv compiler

Install dependencies
```bash
$ sudo apt-get install autoconf automake autotools-dev curl python3 python3-pip libmpc-dev libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf libtool patchutils bc zlib1g-dev libexpat-dev ninja-build git cmake libglib2.0-dev libslirp-dev
```

Install **riscv64-unknown-linux-musl-gcc**
```bash
cd riscv-gnu-toolchain
export RV64_GNU_INSTALL=$(pwd)/install
CFLAGS="-O3 -fPIC" CXXFLAGS="-O3 -fPIC" ./configure --prefix=$RV64_GNU_INSTALL --disable-multilib --with-languages=c,c++
make -j8 musl
```

### Build MVAPICH with SST RDMA NIC channel device

First build librdma from rdmaNic
```bash
cd sst/libRDMA
make
```


```bash
export RDMA_NIC_DIR=$(realpath ./sst/sst-elements/src/sst/elements/rdmaNic)
export RDMA_LIB_DIR=$(realpath ./sst/libRDMA/riscv64/)

tar xzvf mvapich2-2.3.7-1.tar.gz
ulimit -n 4096
patch --directory=mvapich2-2.3.7-1/ -p1 < mvapich2-2.3.7-1.patch

cd mvapich2-2.3.7-1/
./autogen.sh

mkdir install
mkdir build
cd build

../configure                                                                            \
                --prefix=$(pwd)/../install                                              \
                --enable-fortran=no                                                     \
                --with-device=ch3:rdma                                                  \
                --enable-romio=no                                                       \
                --enable-hybrid=no                                                      \
                --enable-shared=no                                                      \
                --enable-static=yes                                                     \
                --with-pmi=vanadis                                                      \
                --with-pm=none                                                          \
                --enable-threads=multiple                                               \
                --enable-rsh=yes                                                        \
                --host=riscv64-unknown-linux-musl                                       \
                CC=${RV64_GNU_INSTALL}/bin/riscv64-unknown-linux-musl-gcc               \
                CFLAGS="-I${RDMA_NIC_DIR}/tests/app/rdma/include -I${RDMA_NIC_DIR}"     \
                CXX=${RV64_GNU_INSTALL}/bin/riscv64-unknown-linux-musl-g++              \
                CXXFLAGS="-I${RDMA_NIC_DIR}/tests/app/rdma/include -I${RDMA_NIC_DIR}"   \
                LDFLAGS="-L${RDMA_LIB_DIR}"                                             \
                LIBS=-lrdma

make -j8 install
```
