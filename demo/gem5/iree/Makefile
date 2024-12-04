# Copyright (c) 2024 imec v.z.w.
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

IREE_GIT_TAG=iree-3.0.0
TOOLCHAIN_ROOT=..
TOOLCHAIN_URL=https://toolchains.bootlin.com/downloads/releases/toolchains/riscv64-lp64d/tarballs/riscv64-lp64d--musl--stable-2024.05-1.tar.xz

IREE_BUILD_OPTS=\
	-DIREE_BUILD_COMPILER=OFF \
	-DIREE_BUILD_TESTS=OFF \
	-DIREE_BUILD_SAMPLES=OFF \
	-DIREE_BUILD_DOCS=OFF \
	-DIREE_HAL_DRIVER_VULKAN=OFF

toolchan_root_abspath=$(shell realpath $(TOOLCHAIN_ROOT))
toolchain_path=$(toolchan_root_abspath)/toolchain-riscv64
sysroot_path=$(toolchain_path)/riscv64-buildroot-linux-musl/sysroot

default: iree-build-riscv64

iree-dist:
	git clone --depth 1 --branch $(IREE_GIT_TAG) https://github.com/iree-org/iree.git iree-dist
	cd iree-dist && git submodule update --init \
		third_party/benchmark \
		third_party/cpuinfo \
		third_party/flatcc \
		third_party/googletest \
		third_party/hip-build-deps \
		third_party/musl \
		third_party/spirv_cross \
		third_party/tracy \
		third_party/vulkan_headers \
		third_party/webgpu-headers \
		third_party/hsa-runtime-headers

iree-build-native: iree-dist
	cd $< && cmake -G "Unix Makefiles" \
		$(IREE_BUILD_OPTS) \
		-DCMAKE_INSTALL_PREFIX=../$@/install \
		-DCMAKE_BUILD_TYPE=RelWithDebInfo \
		. -B ../$@
	cd $< && cmake --build ../$@ --target install

orig_dir=$(patsubst %.tar.gz,%,$(patsubst %.tar.bz2,%,$(patsubst %.tar.xz,%,$(lastword $(subst /, ,$(TOOLCHAIN_URL))))))

.PHONY: toolchain-riscv64
toolchain-riscv64: $(toolchain_path)
$(toolchain_path):
	wget $(TOOLCHAIN_URL) -O $(TOOLCHAIN_ROOT)/toolchain-riscv64.tar.bz2
	tar -xvf $(TOOLCHAIN_ROOT)/toolchain-riscv64.tar.bz2 -C $(TOOLCHAIN_ROOT)
	mv $(TOOLCHAIN_ROOT)/$(orig_dir) $(toolchain_path)
	$(toolchain_path)/relocate-sdk.sh
	rm $(TOOLCHAIN_ROOT)/toolchain-riscv64.tar.bz2

iree-build-riscv64: iree-dist iree-build-native $(toolchain_path)
	cd $< && cmake -G "Unix Makefiles" \
		$(IREE_BUILD_OPTS) \
		-DIREE_ENABLE_CPUINFO=OFF \
		-DCMAKE_INSTALL_PREFIX=../$@/install \
		-DIREE_HOST_BIN_DIR=../iree-build-native/install/bin \
		-DCMAKE_TOOLCHAIN_FILE="../toolchain.generic.cmake" \
		-DTOOLCHAIN_TARGET=riscv64 \
		-DTOOLCHAIN_PATH=$(toolchain_path) \
		-DTOOLCHAIN_PREFIX=riscv64-buildroot-linux-musl- \
		. -B ../$@
	cd $< && cmake --build ../$@ --target install

.PHONY: clean
clean:
	rm -rf iree-dist iree-build-*

.PHONY: distclean
distclean: clean
	rm -rf $(toolchain_path)
