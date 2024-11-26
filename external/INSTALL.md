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
