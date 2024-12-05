# Install SST

## Dependencies
Install dependencies
```bash
sudo apt install openmpi-bin openmpi-common libtool libtool-bin autoconf python3 python3-dev automake build-essential git
```


## How to install sst-core on a debian system
```bash
cd sst-core
export SST_CORE_HOME=$(pwd)/install
./autogen.sh
mkdir build
cd build
../configure --prefix=$SST_CORE_HOME
make -j all
make install
cd ../../
```

## How to install sst-elements on a debian system
```bash
cd sst-elements
git apply ../sst-elements.patch
export SST_ELEMENTS_HOME=$(pwd)/install
./autogen.sh
mkdir build
cd build
../configure --prefix=$SST_ELEMENTS_HOME --with-sst-core=$SST_CORE_HOME
make -j all
make install
cd ../../
```
