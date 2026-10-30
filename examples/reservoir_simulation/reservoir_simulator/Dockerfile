# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# The build and run environment: the NVIDIA HPC SDK, which brings nvcc, the CUDA
# libraries, NCCL and a CUDA-aware OpenMPI (HPC-X) as one matched set, plus the
# system packages HYPRE, AmgX, Ginkgo and OPM Flow need.
#
# build_container.sh chooses NVHPC_IMAGE from the CUDA version the driver on this
# machine reports, so nothing here names a GPU or a CUDA version.  The libraries
# and the simulator are compiled later, by ./nvrs.sh install inside the running
# container, where nvidia-smi can see the GPU they are built for.
#
# The images are multi-architecture, so this builds on x86_64 and aarch64.
ARG NVHPC_IMAGE=nvcr.io/nvidia/nvhpc:26.5-devel-cuda13.2-ubuntu22.04
FROM ${NVHPC_IMAGE}

RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

# Build tools and the libraries the install scripts expect.
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    gnupg build-essential gfortran pkg-config file    \
    software-properties-common apt-utils              \
    automake autoconf autotools-dev libtool flex bison\
    git git-lfs curl wget unzip bzip2                 \
    tmux vim htop gdb strace                          \
    libssl-dev libncurses5-dev zlib1g-dev             \
    liblz4-dev libbz2-dev libbsd-dev                  \
    libnuma-dev libtbb-dev                            \
    hwloc libhwloc-dev libevent-dev                   \
    python3 python-is-python3 python3-pip             \
    python3-venv python3-dev virtualenv               \
    python3-numpy python3-matplotlib python3-tk libgl1\
    libfftw3-dev libblas-dev liblapack-dev

# Boost, for OPM Flow.  The headers and the per-component CMake configs come from
# different packages: with only libboost-dev, opm-common's configure dies looking
# for boost_system.
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libboost-all-dev

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y locales && localedef -i en_US -f UTF-8 en_US.UTF-8

# CMake.  Not pinned to one release, because Kitware's repository drops old patch
# releases and a pin to one that has gone stops the build here.  The newest 3.x is
# taken because OPM's configure does not accept CMake 4.
RUN wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor - \
      > /usr/share/keyrings/kitware-archive-keyring.gpg && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' \
      > /etc/apt/sources.list.d/kitware.list && \
    apt-get update -y && \
    CMAKE_V="$(apt-cache madison cmake | awk '{print $3}' | grep '^3\.' | head -1)" && \
    test -n "$CMAKE_V" && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y "cmake=$CMAKE_V" "cmake-data=$CMAKE_V"

# The image has no unprivileged user, and mpirun refuses root without these.
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# The repository is mounted at /workspace/project by build_container.sh rather than
# copied in, so an edit on the host is an edit in the container.
WORKDIR /workspace/project

CMD ["/bin/bash"]
