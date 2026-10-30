# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# The container the GB200 results in this repository were produced in.
#
# CUDA 12.8 on Ubuntu 22.04, with UCX, PMIx and OpenMPI built from source so
# that MPI is CUDA-aware and Slurm-aware.  That last part is the reason this
# file exists rather than a stock image: NVRS is an MPI program whose ranks
# exchange device buffers directly, and a stock OpenMPI staged through host
# memory -- or refused to launch under srun -- which is a different measurement
# from the one the papers report.
#
# Build and run it with ./build_container.sh; see that script for the flags.
#
# Multi-architecture note: the base image is multi-arch, so this file builds on
# x86_64 and on aarch64 (GB200, Grace-Blackwell) without modification.  Nothing
# below hard-codes an architecture.

FROM nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu22.04

# upgrade the image:
RUN apt update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt -y install tzdata
RUN apt update -y --fix-missing && apt upgrade -y

# install system and dev dependencies:
RUN apt install -y                               \
    gnupg  build-essential                       \
    software-properties-common                   \
    apt-utils sudo libtiff5-dev openssh-server   \
    libssl-dev libncurses5-dev unzip flex        \
    libx11-dev libxt-dev                         \
    tmux screen git git-lfs vim htop curl wget   \
    automake autoconf libtool flex bison         \
    wget git                                     \
    libnuma-dev libibverbs-dev librdmacm-dev     \
    libslurm-dev slurm-client hwloc pkg-config   \
    hwloc libhwloc-dev libevent-dev              \
    bzip2 file gcc gfortran                      \
    gdb autotools-dev libtool automake           \
    liblz4-dev libbz2-dev strace unzip zlib1g-dev\
    python3 python-is-python3 python3-pip        \
    python3-venv python3-dev virtualenv          \
    python3-matplotlib python3-tk libgl1         \
    libtbb-dev libbsd-dev libfuse3-dev           \
    linux-tools-common linux-tools-generic

# additional packages for GEOS code
RUN apt install -y                      \
  libfftw3-3 libfftw3-dev pkg-config    \
  libbotan-2-dev libzbar-dev libssl-dev \
  libblas-dev liblapack-dev

# Boost, for OPM Flow.  The headers and the per-component CMake configs come
# from different packages: with only libboost-dev installed, BoostConfig.cmake
# is present and every boost_<component>-config.cmake is missing, so
# opm-common's configure dies with
#
#   Could not find a package configuration file provided by "boost_system"
#
# It belongs here rather than in install_opm_gpu.sh because dpkg needs real
# root, which a user inside this container under Slurm does not have: apt
# downloads the packages happily and then fails at the install step with
# "requested operation requires superuser privilege".
RUN apt install -y libboost-all-dev

# sudo without passwd:
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
# locales:
RUN apt install -y locales && localedef -i en_US -f UTF-8 en_US.UTF-8
# start ssh:
RUN /etc/init.d/ssh start

# build and install UCX from source:
#
# Two adjustments, for different reasons.
#
# 1. cuMemGetHandleForAddressRange.  UCX 1.18.1 names the unversioned typedef
#    and CUDA 12.8 ships only PFN_..._v11070.  The rename is applied ONLY when
#    the unversioned name is genuinely absent from the headers, so a CUDA that
#    does define it is left alone rather than rewritten to a symbol it may not
#    have.  This one is load-bearing: cuda_copy_md.c is the CUDA transport,
#    which is the whole reason this image exists.
#
# 2. UCX's own test programs are NOT built, and that is the fix for a real
#    failure rather than a tidy-up.
#
#    An earlier version of this file carried a second sed that rewrote
#    test/apps/test_cuda_hook.c from a 3-argument cuCtxCreate to a 4-argument
#    one.  That is wrong against this CUDA and broke the build 102 s into the
#    stage.  CUDA 12.8's cuda.h still maps cuCtxCreate to cuCtxCreate_v2:
#
#        CUresult CUDAAPI cuCtxCreate(CUcontext*, unsigned int, CUdevice);
#
#    so UCX's original 3-argument call was already correct, and the patch
#    turned it into "too many arguments to function" -- which -Werror makes
#    fatal.  Patching it the other way would work today and break on the CUDA
#    where cuCtxCreate becomes _v4.
#
#    These are UCX's test apps.  Nothing in this repository runs them, and
#    pinning them to one CUDA version buys nothing.  Building `src` alone
#    sidesteps the question permanently and is faster.  ucx_info and
#    ucx_perftest live under src/tools, so both survive -- which matters,
#    because build_container.sh check calls ucx_info.
WORKDIR /opt
RUN git clone --branch v1.18.1 --depth 1 --recurse-submodules https://github.com/openucx/ucx.git ucx_sources && \
    cd ucx_sources && \
    if grep -rqs 'PFN_cuMemGetHandleForAddressRange[^_]' /usr/local/cuda/include/cudaTypedefs.h; then \
        echo "== CUDA headers define the unversioned typedef; no rename needed"; \
    else \
        echo "== renaming cuMemGetHandleForAddressRange to its _v11070 typedef"; \
        sed -i 's/PFN_cuMemGetHandleForAddressRange\b/PFN_cuMemGetHandleForAddressRange_v11070/g' \
            src/uct/cuda/cuda_copy/cuda_copy_md.c; \
    fi && \
    ./autogen.sh && \
    ./contrib/configure-release --prefix=/opt/ucx --with-cuda=/usr/local/cuda --enable-mt --with-verbs --with-rdmacm && \
    make -j"$(nproc)" -C src && \
    make -C src install && \
    (make install-pkgconfigDATA || true)

ENV UCX_HOME=/opt/ucx

# Build and install PMIx from source:
WORKDIR /opt
RUN git clone --branch v4.2.8 --depth 1 --recurse-submodules https://github.com/openpmix/openpmix.git  pmix_sources && \
    cd pmix_sources && \
    ./autogen.pl && \
    ./configure --prefix=/opt/pmix && \
    make -j"$(nproc)" && \
    make install
ENV PMIX_HOME=/opt/pmix

# build and install OpenMPI with UCX, PMIx, CUDA, and Slurm support:
WORKDIR /opt
RUN git clone --branch v4.1.5 --depth 1 --recurse-submodules https://github.com/open-mpi/ompi.git ompi_sources && \
    cd ompi_sources && \
    ./autogen.pl && \
    mkdir build && cd build && \
    ../configure --prefix=/opt/openmpi \
        --with-cuda=/usr/local/cuda \
        --with-ucx=${UCX_HOME} \
        --with-pmix=${PMIX_HOME} \
        --with-slurm \
        --enable-mpi-cxx \
        --enable-mpi1-compatibility \
        --with-hwloc=internal && \
    make -j"$(nproc)" && \
    make install

ENV CUDA_HOME=/usr/local/cuda
ENV OMPI_HOME=/opt/openmpi
ENV PATH=/opt/ucx/bin:/opt/openmpi/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/openmpi/lib:/opt/ucx/lib:/opt/pmix/lib:$LD_LIBRARY_PATH

# set OpenMPI environment to use UCX and avoid problematic BTLs:
#
# Without pml=ucx the run can fall back to a transport that is not CUDA-aware,
# which still works and is quietly several times slower -- the worst kind of
# difference, because nothing fails.
ENV OMPI_MCA_pml=ucx
ENV OMPI_MCA_btl=^vader,tcp,openib,uct

# install latest cmake, needed for RAJA:
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
RUN echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' |  tee /etc/apt/sources.list.d/kitware.list >/dev/null
RUN apt update -y --fix-missing
RUN apt install -y cmake=3.30.8-0kitware1ubuntu22.04.1 cmake-data=3.30.8-0kitware1ubuntu22.04.1

# The repository is mounted at /workspace/project by build_container.sh rather
# than copied in, so an edit on the host is an edit in the container and a
# rebuilt simulator does not need a rebuilt image.
WORKDIR /workspace/project

CMD ["/bin/bash"]
