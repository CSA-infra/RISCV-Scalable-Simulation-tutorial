# Copyright (c) 2024 The Regents of the University of California
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

FROM --platform=${BUILDPLATFORM} ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt -y update && apt -y upgrade && apt -y install \
   libtool \
   libtool-bin \
   autoconf \
   python3 \
   python3-dev \
   automake \
   build-essential \
   git \
   vim

WORKDIR /scratch/sst-core
ENV SST_CORE_HOME=/scratch/sst-core/install
RUN git clone --depth 1 --branch v14.1.0_Final https://github.com/sstsimulator/sst-core.git source

WORKDIR /scratch/sst-elements
COPY sst-elements.patch /scratch/sst-elements/sst-elements.patch
ENV SST_ELEMENTS_HOME=/scratch/sst-elements/install
RUN git clone --depth 1 --branch v14.1.0_Final https://github.com/sstsimulator/sst-elements.git source

WORKDIR /scratch/sst-core/source
RUN ./autogen.sh
WORKDIR /scratch/sst-core/build
RUN ../source/configure --prefix=$SST_CORE_HOME --enable-mpi=no
RUN make -j$(nproc) install
ENV PATH=$SST_CORE_HOME/bin:$PATH

WORKDIR /scratch/sst-elements/source
RUN git apply /scratch/sst-elements/sst-elements.patch
RUN ./autogen.sh
WORKDIR /scratch/sst-elements/build
RUN ../source/configure --prefix=$SST_ELEMENTS_HOME --with-sst-core=$SST_CORE_HOME --enable-mpi=no
RUN make -j$(nproc) install


## pre-commit, as installed via apt in 24.04, attempts to create a cache
## directory at "${HOME}/.cache/pre-commit". If running docker with non-root,
## the HOME directory is set to "/". Since non-root users do not have permission
## to write to this directory, an error is returned when pre-commit is executed.
## pre-commit's default cache directory can be changed via the `XDG_CACHE_HOME`
## enivoronment variable. Here we set it to "/tmp". With this pre-commit will
## create a "/tmp/pre-commit" directory to use for caching. "/tmp" was chosen
## as it's a directory any user can access and write to. Given this only stores
## caching information, the "/tmp" directory being wiped is not a concern.
ENV XDG_CACHE_HOME=/tmp/
