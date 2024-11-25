# Install external tools

## How to install SST

[Installation instructions](sst/INSTALL.md)

## How to install riscv-gnu-toolchain
### Dependencies
Install dependencies
```bash
$ sudo apt-get install autoconf automake autotools-dev curl python3 python3-pip libmpc-dev libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf libtool patchutils bc zlib1g-dev libexpat-dev ninja-build git cmake libglib2.0-dev libslirp-dev
```

### Install
```bash
cd sst/riscv-gnu-toolchain
mkdir install
export GCC_INSTALL_DIR=$(pwd)/install
CFLAGS="-O3 -fPIC" CXXFLAGS="-O3 -fPIC" ./configure --prefix=$GCC_INSTALL_DIR --disable-multilib --with-languages=c,c++
make -j all
make install
```
