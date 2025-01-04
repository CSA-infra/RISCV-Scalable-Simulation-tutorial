Application-oriented system modeling and optimization
=====================================================

*i.e. how to lower an AI/ML model to simulated RISC-V hardware for system-level
exploration*

The goal of this tutorial is to introduce the attendees to architectural
simulation targeting machine learning workloads. The main tool we will be
using to model a sample RISC-V system and run applications on top is
\ `gem5 <https://www.gem5.org/>`__\ . The ML benchmarks are derived from
ONNX files, translated into machine-optimized code and executed though a
ligthweight runtime. This process is carried out with the help of the
\ `IREE <https://iree.dev/>`__\  workflow.

Prerequisites
-------------

- A Linux-based x86-64 system (native or WSL2/VM)
- Docker or Podman

Containerized environment
~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   The container is executed in privileged mode to
   allow mounting the disk image as a loop device. If you don’t like this,
   remove the corresponding option from ``docker-compose.yaml``.

Dealing with all the software dependencies that this setup needs can be
complicated. For this reason, a container file has been provided, which
allows to generate a virtual environment with all the dependencies
installed. Assuming that Docker is present in your system, you can prepare
the environment this way:

::

   cd docker
   docker compose up -d

If it doesn’t work, try with ``docker-compose`` alternatively.

To enter the container:

::

   docker exec -it docker_vlsid-iree-gem5_1 /bin/bash

If you stop the container (e.g. reboot), you can easily return back to
it with:

::

   docker start docker_vlsid-iree-gem5_1
   docker exec -it docker_vlsid-iree-gem5_1 /bin/bash

Finally, if you want to destroy the container, you can do it with:

::

   cd docker
   docker compose down

The working directory inside the container is ``/opt/vlsid-iree-gem5``.
We will assume that every command is executed from that folder.

Environment Setup
-----------------

Part 1: Prepare benchmark
~~~~~~~~~~~~~~~~~~~~~~~~~

The IREE workflow is used to first convert a ML model to a supported
intermediate representation, then compile and optimize the model for a
target architecture. The output of the process is a Virtual Machine
FlatBuffer (VMFB) file than can be run by the IREE runtime.

A simple MNIST image classification model will be used as example, but
the process is generalizable to other models too. The file format for the
model is ONNX. Note that IREE also supports other formats (e.g. TF/TFLite),
it is possible to convert them to MLIR using the right importers.

- Download ONNX model

::

   wget https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/mnist/model/mnist-8.onnx -O mnist-8-orig.onnx

- `Upgrade ONNX
  opset <https://iree.dev/guides/ml-frameworks/onnx/#troubleshooting>`__

::

   ./convert_onnx_model.py mnist-8-orig.onnx mnist-8.onnx

- Use IREE to convert ONNX file to MLIR Torch ONNX dialect

::

   iree-import-onnx mnist-8.onnx > mnist-8.mlir

- Compile MLIR model to VMFB

::

   iree-compile --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-triple=riscv64 --iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c mnist-8.mlir -o mnist-8.vmfb

Part 2: Compile IREE run module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The IREE run module allows the execution of a compiled module using the
IREE runtime. This module has to be added to the final disk image
together with the benchmarks, since we don’t want to pull the entire
IREE distribution.

Even if pre-built binaries are available, as of now they are not
compiled for any RISC-V architecture. Thus, we will have to compile this
module from source. A Makefile has been provided to simplify the
process.

::

   make -C iree

Part 3: Compile m5 utility
~~~~~~~~~~~~~~~~~~~~~~~~~~

The m5 utility is used to send pseudo-instructions to the simulator.
This allows a number of operations, like checkpointing, resetting
statistics, etc. We want to include this utility in our final image.
Note that will need the cross-compiler employed in the previous step to
generate the binary.

- Get the gem5 simulator

::

   git clone https://github.com/gem5/gem5.git -b v24.1.0.1

- Compile the m5 utility

::

   export PATH=$PATH:$(realpath toolchain-riscv64/bin)
   scons riscv.CROSS_COMPILE=riscv64-buildroot-linux-musl- -C gem5/util/m5 build/riscv/out/m5

Part 4: Prepare RISC-V disk image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. warning::
   If using Podman or rootless Docker, this steps must be done
   outside the container, as they typically require sudo permissions.
   Pay attention when executing each command!

The last part of the setup consists in packing the benchmarks and IREE
runtime into a disk image. For this task, we will use a pre-built
minimal image from the gem5 community and modify it.

- Get and extract `base
  image <https://resources.gem5.org/resources/riscv-disk-img?version=1.0.0>`__

::

   wget https://storage.googleapis.com/dist.gem5.org/dist/develop/images/riscv/busybox/riscv-disk.img.gz
   gzip -d riscv-disk.img.gz
   cp riscv-disk.img vlsid-disk.img

- Mount image

::

   mkdir /tmp/rootfs
   sudo mount vlsid-disk.img /tmp/rootfs

- Copy benchmark

::

   sudo cp mnist-8.vmfb /tmp/rootfs/root/

- Copy IREE run module

::

   sudo cp iree/iree-build-riscv64/install/bin/iree-run-module /tmp/rootfs/bin/

- Copy m5 utility

::

   sudo cp gem5/util/m5/build/riscv/out/m5 /tmp/rootfs/sbin/

- Unmount image

::

   sudo umount /tmp/rootfs

Machine Learning Workload Execution
-----------------------------------

At this point, we are ready to run the experiment. A gem5 configuration
file is present in this directory, which is derived from the
``riscv-fs.py`` sample script of gem5. The main difference is that
instead of using the default disk image it will pick the one that we
have just generated.

- Compile gem5

.. note::
   This step will take a while.

::

   scons build/RISCV/gem5.opt -C gem5 -j$(nproc)

- Compile m5term

::

   make -C gem5/util/term

- Run the script

.. note::
   This step will take a while. We will speed up following
   executions through checkpointing.

::

   ./build/RISCV/gem5.opt vlsid-riscv-fs.py

While the simulation is running, its output is not immediately visible,
as it is redirected to a separate console. To view it, open another
terminal and use the m5term utility.

::

   ./gem5/util/term/m5term 3456

The boot process is going to take several minutes. After that, you will
se a login shell. Enter user “root” and password “root” to proceed.
After login, you can launch your IREE benchmark. This is the command to
execute for MNIST:

::

   iree-run-module --module=/root/mnist-8.vmfb --device=local-task --input="1x1x28x28xf32=0"

For simplicity we are assuming an input tensor filled with zeros. You
should see this output after some time:

::

   EXEC @CNTKGraph
   result[0]: hal.buffer_view
   1x10xf32=[-0.044856 0.00779166 0.0681008 0.0299937 -0.12641 0.140219 -0.0552849 -0.0493838 0.0843221 -0.0545404]

Congratulations! You are ready to go!

Extra: Checkpoints
------------------

You will have noticed that booting the Linux kernel and reaching the
login shell takes several minutes, even with a minimal image like the
one we are using. We want to avoid waiting so long for each one of the
experiments. One of the commonly used techniques to deal with these
situations is checkpointing: we can “take a picture” of the system at a
certain moment of time and start other simulations from that point.
Technically speaking, this requires saving the main memory content and
the processors context. Cache content is not saved, but since we will
execute our benchmarks from scratch this is not a big deal.

In order to dump a checkpoint, after entering the shell in the simulated
environment type this command:

::

   m5 checkpoint

After terminating the simulation, you will see that in the output folder
(e.g. ``m5out``) a folder named ``cpt.<somenumber>`` has appeared. This
contains the checkpoint we have just dumped. We strongly suggest to move
this folder outside the ``m5out`` directory.

::

   mv m5out/cpt.<somenumber> checkpoint

From now on, it will be possible to execute a simulation starting from
this checkpoint. It is sufficient to add an argument to the gem5
command, specifying the position of the folder containing the checkpoint
files:

::

   ./build/RISCV/gem5.opt vlsid-riscv-fs.py --restore-from checkpoint

This way, you will be immediately dropped to the shell. Huge
improvement!

Experimental Studies
--------------------

Now that you are able to run complete simulations, it is time to explore
a few knobs and analyze their impact on the system performance.

Part 1: Change CPU model
~~~~~~~~~~~~~~~~~~~~~~~~

The gem5 simulator supports different `CPU
models <https://raw.githubusercontent.com/gem5bootcamp/gem5-bootcamp-env/main/assets/slides/using-gem5-05-gem5-cpus-tutorial%202.pdf>`__.
By default, the script runs with an *atomic* CPU, which implies atomic
accesses to the memory system with fixed latencies. This model is fast
and simple, but inaccurate.

The first task is to replace the CPU type with a more detailed one.
There are three possible choices:

- **TimingSimpleCPU:** simple timing CPU, 1-stage pipeline
- **MinorCPU:** in-order CPU, 4-stages pipeline
- **O3CPU:** out-of-order CPU, 7-stages pipeline

These CPU models are highly configurable, but for this experiment it is
fine to stick with the default parameters set.

To implement such change, open the ``vlsid-riscv-fs.py`` script and
change ``CPUTypes.ATOMIC`` (line 78) to ``CPUTypes.TIMING``,
``CPUTypes.MINOR`` and ``CPUTypes.O3``. After each execution, have a
look at the ``stats.txt`` file in the output folder (default:
``m5out``). In particular, look at how these statistics change:

::

   simSeconds -> Simulated system execution time
   hostSeconds -> Host system simulation time
   board.processor.cores.core.ipc -> IPC of simulated CPU
   board.memory.mem_ctrl.dram.bwTotal::total -> DRAM memory bandwidth

**Tip 1:** Wrap your benchmark execution around the commands “m5
resetstats” and “m5 exit”, to make sure that the statistics only reflect
the benchmark execution and not the system boot or idle time. E.g.:

::

   m5 resetstats && iree-run-module [...] && m5 exit

**Tip 2:** You can specify different output folders for each experiment.
E.g.:

::

   gem5.opt -d ./experiment1 vlsid-riscv-fs.py

Part 2: Change cache hierarchy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The cache configuration can have a significant impact on the system
performance, depending on the data locality and access patterns of the
executed applications. This is one of the knobs we can easily change in
the ``vlsid-riscv-fs.py`` configuration file (line 70).

The second task consists in performing the experiments after applying
the following modifications (one by one):

- Decrease L1I (instruction cache) and L1D (data cache) size from 32 kB
  to 8 kB
- Increase L2 (last-level cache) size from 512 kB to 2 MB

Use MinorCPU or O3CPU. Compare the output statistic with the baseline
configuration, to check if there is a change in performance and how
appreciable that is. You can also have a look at cache-specific metrics,
e.g. the miss rates:

::

   board.cache_hierarchy.l1d-cache-0.overallMissRate::total
   board.cache_hierarchy.l1i-cache-0.overallMissRate::total
   board.cache_hierarchy.l2-cache-0.overallMissRate::total

Part 3: Vectorization
~~~~~~~~~~~~~~~~~~~~~

The RISC-V architecture we are simulating supports the RVV vector
extension v1.0. This means that the IREE compiler can optimize the
application by enabling SIMD support. The default VLEN for the simulated
hardware is of 256 bits.

For this step, we will need to recompile the benchmark and add it to the
disk image. The following command will create an RVV-enabled benchmark:

::

   iree-compile --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-target-triple=riscv64 --iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c,+v,+zvl256b -riscv-v-vector-bits-min=256 -riscv-v-fixed-length-vector-lmul-max=8 mnist-8.mlir -o mnist-8
   -v.vmfb

Execute this new version of the benchmark and compare the output with
the non-vectorized version. You should notice an improvement of the
performance.

**Note:** Like other microarchitectural parameters, the latencies of the
vector units are not calibrated on any specific design, and default
values are used. Do not expect fully realistic numbers.

Part 4: New benchmarks
~~~~~~~~~~~~~~~~~~~~~~

.. warning::
   The execution time can be much higher for more complex
   benchmarks, even in atomic mode. We suggest you to try out these
   tests after the tutorial, keeping the simulations as background tasks
   until they complete.

Now that you know how to run the full workflow, you can try out new
benchmarks. Bear in mind that not all the models are supported with the
current version of IREE, and compatibility issues may arise when
compiling. We will provide you with a few examples that are guaranteed
to succeed.

::

   https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/mobilenet/model/mobilenetv2-10.onnx
   https://github.com/onnx/models/raw/refs/heads/main/validated/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx

The launch commands for these models are:

::

   iree-run-module --module=/root/mobilenetv2-10.vmfb --device=local-task --input="1x1x672x672xf32=0"
   iree-run-module --module=/root/super-resolution-10.vmfb --device=local-task --input="1x1x224x224xf32=0"

**Tip:** If you want to store multiple models in your image, or models
that exceed the image capacity, you may run out of space. You can resize
the image to a bigger size (e.g. 150 MB) with the following commands:

::

   e2fsck -f vlsid-disk.img
   resize2fs vlsid-disk.img 150M
