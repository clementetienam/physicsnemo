# SPDX-License-Identifier: Apache-2.0
# Authors: Clement Etienam, Tsubasa Onishi, Oleg Ovcharenko,
#          Nick Luiken, Issam Said
#
# Direct nvcc build (mirrors Mixed_precision/src/run_hypre_cusolver_mixed_cpr.py
# link line: -lHYPRE -lcusparse -lcublas -lcusolver -lcudart).  Prefer CMake for
# production; this Makefile is the quick single-command path on a GPU node.
#
#   source ../../../mixed_precision_emmulation/hypre_env.sh   # sets HYPRE_FP32_DIR
#   make                                        # CUDA arch auto-detected
#   make FP64_PRECOND=1                         # all-FP64 comparison build
#   make TENSORCORE=1 CUTLASS_DIR=<checkout>    # adds JACOBIAN TENSORCORE
#
# FP64_PRECOND=1 builds nvrs_cuda_fp64: the same solver with the whole CPR
# preconditioner in FP64 (AmgX dDDI, an FP64 hierarchy for HYPRE and Ginkgo),
# linked against the FP64 HYPRE (HYPRE_FP64_DIR).  Its objects go to
# build_fp64/, so the default mixed-precision build is left as it is.
#
# TENSORCORE=1 builds nvrs_cuda_tc: the same simulator with the optional Tensor
# Core Jacobian compiled in, objects in build_tc/.  Without it none of that code
# is compiled.  CUTLASS_DIR is only needed for TF32; without it the keyword
# still runs with TCPRECISION FP32.
#
# hypre_env.sh must be sourced in every new shell.  Without it HYPRE_DIR is
# empty, which used to expand to -I/include and fail with "cannot open source
# file HYPRE.h" -- three compiler errors that say nothing about the real cause.

FP64_PRECOND ?= 0
FP64 := $(filter 1,$(FP64_PRECOND))
TENSORCORE ?= 0
TC := $(filter 1,$(TENSORCORE))
HYPRE_DIR ?= $(if $(FP64),$(HYPRE_FP64_DIR),$(HYPRE_FP32_DIR))
HYPRE_INC := $(HYPRE_DIR)/include
# lib or lib64, whichever the HYPRE configure produced.
HYPRE_LIB := $(firstword $(wildcard $(HYPRE_DIR)/lib $(HYPRE_DIR)/lib64))

# Fail on the actual problem rather than on its symptom.  Not for `clean`,
# which has no business needing a HYPRE installation to delete object files.
ifneq ($(MAKECMDGOALS),clean)
  ifeq ($(strip $(HYPRE_DIR)),)
    $(error HYPRE_DIR is empty. Run: source ../../../mixed_precision_emmulation/hypre_env.sh)
  endif
  ifeq ($(wildcard $(HYPRE_DIR)/include/HYPRE.h),)
    $(error No HYPRE.h under $(HYPRE_DIR)/include. Check HYPRE_$(if $(FP64),FP64,FP32)_DIR in hypre_env.sh)
  endif
  ifeq ($(strip $(HYPRE_LIB)),)
    $(error No lib or lib64 under $(HYPRE_DIR). That prefix has headers but no library)
  endif
endif
# CUDA_ARCH: leave unset to detect; set to a number (e.g. 100 for GB200) to
# pin one SM, or to native/all/all-major to pass that through to nvcc.
CUDA_ARCH ?= auto
# The MPI wrapper must be the one belonging to the MPI everything else links
# against.  A bare "mpicxx" is whatever comes first on PATH, and on a machine
# with a source-built Open MPI in /opt and a packaged one on PATH those are not
# the same, which shows up as:
#
#   src/well_report.cu:18:10: fatal error: mpi.h: No such file or directory
#
# Look beside the library first.  MPICXX=... still overrides.
MPICXX    ?= $(firstword $(wildcard $(OMPI_HOME)/bin/mpicxx) \
                         $(wildcard /opt/openmpi/bin/mpicxx) \
                         $(shell command -v mpicxx 2>/dev/null) mpicxx)
NVCC      ?= nvcc

# --- Detecting the architecture --------------------------------------------
# -arch=native would be the obvious answer and is the wrong one here.  It asks
# the GPU visible to the COMPILER, which on a cluster is a login node that
# usually has no GPU at all -- and when it does have one, it is not necessarily
# the same part the job will land on.  Getting that wrong is not a build error
# but a runtime one, and an obscure one: the fatbin has no image for the real
# device and every kernel launch fails with "no kernel image is available".
#
# So: ask nvidia-smi what is here, ask nvcc whether it can target it, and use
# the answer only if both agree.  If either is unavailable -- a login node, an
# older toolkit -- fall back to a fatbin covering the architectures this code
# is actually run on, filtered to the ones this nvcc knows about.  That build
# is slower and larger, and it runs anywhere.
#
# sm_100 (GB200) needs CUDA 12.8 or newer; on an older
# toolkit it is dropped from the list rather than failing the build.
# compute_cap comes back as e.g. "10.7"; strip the dot to get 107.  Hoisted
# above NVCC_ARCH_LE, which needs it.
DETECTED_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
NVCC_ARCHES := $(shell $(NVCC) --list-gpu-arch 2>/dev/null | sed 's/compute_//')
ARCH_WANTED := 70 80 90 100 120
# The newest architecture this nvcc can target, numerically.  Make's $(sort) is
# lexicographic and would rank 90 above 120, so the ordering is done in shell.
NVCC_MAX_ARCH := $(shell $(NVCC) --list-gpu-arch 2>/dev/null | sed 's/compute_//' | sort -n | tail -1)
# The newest architecture nvcc knows that is NOT NEWER THAN the GPU in front of
# us.  That distinction is the whole point: PTX is forward-compatible, so the
# driver will JIT compute_103 onto an sm_107, but it cannot run PTX built for
# compute_121 on a device that is only 10.7.  Taking the toolkit's maximum
# would do exactly that.
NVCC_ARCH_LE := $(shell $(NVCC) --list-gpu-arch 2>/dev/null | sed 's/compute_//' | sort -n |                   awk -v d="$(DETECTED_ARCH)" '$$1<=d{m=$$1} END{if (m) print m}')

ifeq ($(CUDA_ARCH),auto)
  ifeq ($(strip $(NVCC_ARCHES)),)
    # Cannot ask nvcc, so trust the device if we found one.
    ARCH_USE := $(DETECTED_ARCH)
  else
    ARCH_USE := $(filter $(DETECTED_ARCH),$(NVCC_ARCHES))
  endif

  # True only when a GPU was seen AND nvcc knows some architecture AND the two
  # do not intersect: that is "hardware newer than the toolkit".  $(and) rather
  # than concatenation, which would also fire with no GPU at all.
  ARCH_TOO_NEW := $(if $(strip $(DETECTED_ARCH)),$(if $(strip $(NVCC_ARCH_LE)),yes))

  ifneq ($(strip $(ARCH_USE)),)
    ARCHFLAG   := -arch=sm_$(ARCH_USE)
    ARCH_ORIGIN = detected on this machine
  else ifeq ($(ARCH_TOO_NEW),yes)
    # A GPU IS present and this nvcc cannot target it: hardware newer than the
    # toolkit.  Hecate's Vera Rubin reports sm_107 against a CUDA 12.8 that
    # stops at 120, and "nvcc fatal: Unsupported gpu architecture" is as far as
    # the old path got.
    #
    # code=compute_NN emits PTX rather than SASS, and the driver JIT-compiles it
    # at load.  That is the supported way to run on a GPU newer than the
    # toolkit, and it is why the fallback below also carries PTX: a fatbin of
    # SASS alone has nothing a future architecture can use.
    # BOTH, and the order matters for what the loader does at run time.
    #
    # code=sm_NN is a cubin, and a cubin is forward-compatible across
    # MINOR revisions within one major capability: sm_103 machine code
    # loads on a 10.7 device directly, with no JIT.  That is as close to
    # Issam's "just add the native SM" as this toolkit allows, since it
    # has no sm_107 to add.
    #
    # code=compute_NN is the PTX, kept as the fallback: if this
    # generation's compatibility rules do not cover the gap, the driver
    # JIT-compiles and the build behaves as it did before.  Shipping both
    # is strictly better than either alone and costs only size.
    ARCHFLAG   := -gencode arch=compute_$(NVCC_ARCH_LE),code=sm_$(NVCC_ARCH_LE) \
                  -gencode arch=compute_$(NVCC_ARCH_LE),code=compute_$(NVCC_ARCH_LE)
    ARCH_ORIGIN = sm_$(DETECTED_ARCH) is newer than this CUDA knows; SASS for sm_$(NVCC_ARCH_LE) (loads directly) plus PTX for compute_$(NVCC_ARCH_LE)
  else
    ifeq ($(strip $(NVCC_ARCHES)),)
      ARCH_FAT := $(ARCH_WANTED)
    else
      ARCH_FAT := $(filter $(ARCH_WANTED),$(NVCC_ARCHES))
    endif
    # SASS for each known architecture, plus PTX for the newest, so the binary
    # also runs on a GPU that did not exist when it was built.
    ARCHFLAG   := $(foreach a,$(ARCH_FAT),-gencode arch=compute_$(a),code=sm_$(a))                   $(if $(strip $(NVCC_MAX_ARCH)),-gencode arch=compute_$(NVCC_MAX_ARCH),code=compute_$(NVCC_MAX_ARCH))
    ARCH_ORIGIN = no usable GPU seen at build time, so a fatbin for $(ARCH_FAT) plus PTX
  endif
else ifneq ($(filter native all all-major,$(CUDA_ARCH)),)
  ARCHFLAG   := -arch=$(CUDA_ARCH)
  ARCH_ORIGIN = CUDA_ARCH=$(CUDA_ARCH)
else
  ARCHFLAG   := -arch=sm_$(CUDA_ARCH)
  ARCH_ORIGIN = pinned by CUDA_ARCH=$(CUDA_ARCH)
endif

# Below sm_60 there is no double-precision atomicAdd, which the residual norm
# reduction in every assembly kernel uses.  Say so here rather than let it
# surface as four overload errors deep in a kernel.
ifneq ($(filter -arch=sm_3% -arch=sm_5%,$(ARCHFLAG)),)
  $(error $(ARCHFLAG) is below sm_60, which has no double-precision atomicAdd. \
    Pass the real architecture as CUDA_ARCH=<number>)
endif

$(info CUDA architecture  : $(ARCHFLAG)  [$(ARCH_ORIGIN)])

# --- AmgX (optional) -------------------------------------------------------
# AMGX_DIR points at an AmgX install with include/amgx_c.h and lib/libamgx*.
# Found -> LINSOLVER AMGX works and is the deck default.  Not found -> the
# backend compiles out entirely and the binary runs on HYPRE, saying so at
# startup rather than pretending.
# AMGX_ROOT was never defined, so this defaulted to empty and AmgX was found
# only when amgx_env.sh had been sourced.  When that file is missing -- and the
# install itself can be perfectly present without it -- the build silently
# dropped to HYPRE only, which is how a tree with a working AmgX produced
# "AmgX: not found".  The prefix install_amgx.sh writes is checked directly.
AMGX_ROOT ?= $(firstword $(wildcard $(CURDIR)/../../amgx-install $(CURDIR)/../../../amgx-install))
AMGX_DIR  ?= $(AMGX_ROOT)
# Find the header, then derive the include directory from where it is, rather
# than assuming a layout.  AmgX installs put amgx_c.h under include/ from a
# `make install`, and under base/include/ when the source tree is used in
# place; a prefix assembled by hand can have either.  Testing only the first
# and reporting "not found" for the others is how an install with the library
# present, the prefix correct and AMGX_DIR set still built HYPRE only.
AMGX_HDR  := $(firstword $(wildcard $(AMGX_DIR)/include/amgx_c.h                                     $(AMGX_DIR)/base/include/amgx_c.h                                     $(AMGX_DIR)/include/*/amgx_c.h))
AMGX_INC  := $(patsubst %/,%,$(dir $(AMGX_HDR)))
AMGX_LIB  := $(firstword $(wildcard $(AMGX_DIR)/lib $(AMGX_DIR)/lib64))
HAVE_AMGX := $(if $(AMGX_HDR),1,)

# --- Ginkgo (optional) -----------------------------------------------------
# GINKGO_DIR points at a Ginkgo install (include/ginkgo/ginkgo.hpp and
# lib/libginkgo*.so), as benchmark/install_ginkgo.sh writes it.  Found ->
# LINSOLVER GINKGO works (FP16 multigrid by default).  Not found -> the backend
# compiles to stubs and a deck asking for it falls back to AmgX, saying so.
GINKGO_ROOT ?= $(firstword $(wildcard $(CURDIR)/../../ginkgo-install $(CURDIR)/../../../ginkgo-install))
GINKGO_DIR  ?= $(GINKGO_ROOT)
GINKGO_HDR  := $(wildcard $(GINKGO_DIR)/include/ginkgo/ginkgo.hpp)
GINKGO_LIB  := $(patsubst %/,%,$(dir $(firstword $(wildcard $(GINKGO_DIR)/lib/libginkgo.so $(GINKGO_DIR)/lib64/libginkgo.so))))
# Every component library the install has: libginkgo itself, then the
# executor modules (cuda, omp, reference, ...) it loads.  Linking the set that
# exists rather than a fixed list keeps a build without, say, the OpenMP
# module from failing on a library that was never built.
GINKGO_LIBS := $(patsubst lib%.so,-l%,$(notdir $(wildcard $(GINKGO_LIB)/libginkgo*.so)))
HAVE_GINKGO := $(if $(and $(GINKGO_HDR),$(GINKGO_LIB)),1,)

# --- Tensor Core Jacobian (optional, TENSORCORE=1) ----------------------------
ifeq ($(TC),1)
  DEFS += -DNVRS_HAVE_TENSORCORE=1
  CUTLASS_DIR ?= $(CUTLASS_HOME)
  HAVE_CUTLASS := $(if $(wildcard $(CUTLASS_DIR)/include/cutlass/cutlass.h),1,)
  ifeq ($(HAVE_CUTLASS),1)
    INCLUDES += -I$(CUTLASS_DIR)/include
    DEFS     += -DNVRS_HAVE_CUTLASS=1
  endif
endif

# --- NCCL (optional) --------------------------------------------------------
# Stream-ordered halo exchanges and reductions for the Krylov tail, which is
# what lets CUDAGRAPH capture it on several ranks.  NCCL_DIR (or NCCL_HOME)
# names an install; otherwise the NVIDIA HPC SDK's comm_libs/nccl beside the
# nvcc on PATH, the CUDA toolkit and the system include paths are searched.
# Not found -> several ranks keep MPI in the tail.
NVCC_BINDIR := $(dir $(shell command -v $(NVCC) 2>/dev/null))
NCCL_DIR  ?= $(or $(NCCL_HOME),$(patsubst %/include/nccl.h,%,$(firstword $(wildcard                /usr/local/nccl/include/nccl.h                $(NVCC_BINDIR)../../comm_libs/nccl/include/nccl.h                $(NVCC_BINDIR)../../../comm_libs/nccl/include/nccl.h                $(NVCC_BINDIR)../../comm_libs/*/nccl/include/nccl.h \
               $(NVHPC_ROOT)/comm_libs/nccl/include/nccl.h \
               $(sort $(wildcard /opt/nvidia/hpc_sdk/*/*/comm_libs/nccl/include/nccl.h)) \
               $(sort $(wildcard /opt/nvidia/hpc_sdk/*/*/comm_libs/*/nccl/include/nccl.h))))))
NCCL_HDR  := $(firstword $(wildcard $(NCCL_DIR)/include/nccl.h                                     $(CUDA_HOME)/include/nccl.h                                     /usr/include/nccl.h /usr/local/cuda/include/nccl.h))
NCCL_INC  := $(patsubst %/,%,$(dir $(NCCL_HDR)))
NCCL_SO   := $(firstword $(wildcard $(NCCL_DIR)/lib/libnccl.so $(NCCL_DIR)/lib64/libnccl.so $(CUDA_HOME)/lib64/libnccl.so /usr/lib/x86_64-linux-gnu/libnccl.so /usr/lib/aarch64-linux-gnu/libnccl.so /usr/lib64/libnccl.so))
NCCL_LIB  := $(patsubst %/,%,$(dir $(NCCL_SO)))
HAVE_NCCL := $(if $(and $(NCCL_HDR),$(NCCL_SO)),1,)
ifeq ($(HAVE_NCCL),1)
  INCLUDES += -I$(NCCL_INC)
  DEFS     += -DNVRS_HAVE_NCCL=1
endif
ifeq ($(HAVE_GINKGO),1)
  INCLUDES += -I$(GINKGO_DIR)/include
  DEFS     += -DNVRS_HAVE_GINKGO=1
endif
# -ccbin names the HOST compiler, and that is not enough: nvcc preprocesses .cu
# files with its own front end, which never sees the include paths the mpicxx
# wrapper would have added.  It works on machines where MPI happens to live in
# /usr/include and fails everywhere else.  So ask the wrapper what it would
# have passed, and keep only the -I flags -- the rest of its output (-pthread
# and friends) is for the host compiler and nvcc rejects it.
MPI_SHOW      := $(shell $(MPICXX) -showme:compile 2>/dev/null || \
                         $(MPICXX) -show 2>/dev/null)
MPI_INC_FLAGS := $(filter -I%,$(MPI_SHOW))

NVCCFLAGS := -O3 -std=c++17 $(ARCHFLAG) --extended-lambda \
             --expt-relaxed-constexpr -rdc=true -ccbin $(MPICXX) \
             $(MPI_INC_FLAGS)
# nvcc does not accept -Wl,... itself; it has to be handed to the host linker
# explicitly, or the link dies with "Unknown option '-Wl,-rpath,...'".
#
# The rpath is not decoration.  The FP32 and FP64 builds of HYPRE install the
# same libHYPRE.so soname, so without it the loader takes whichever happens to
# be on LD_LIBRARY_PATH -- and a run that silently picks up the FP64 library
# still works, just not in the precision the whole design rests on.
LDFLAGS   := -L$(HYPRE_LIB) -Xlinker -rpath -Xlinker $(HYPRE_LIB) \
             -lHYPRE -lcusparse -lcublas -lcusolver -lcudart
ifeq ($(HAVE_AMGX),1)
  LDFLAGS += -L$(AMGX_LIB) -Xlinker -rpath -Xlinker $(AMGX_LIB) -lamgxsh
endif
ifeq ($(HAVE_NCCL),1)
  LDFLAGS += -L$(NCCL_LIB) -Xlinker -rpath -Xlinker $(NCCL_LIB) -lnccl
endif
ifeq ($(HAVE_GINKGO),1)
  LDFLAGS += -L$(GINKGO_LIB) -Xlinker -rpath -Xlinker $(GINKGO_LIB) $(GINKGO_LIBS)
endif
# provisions, so it goes on the rpath too.

# Header dependencies.  Without these, editing a header rebuilds only the
# translation units that happen to be listed on the command line, and the rest
# keep an object file compiled against the OLD definition.
#
# When the header is deck.hpp or linear_solver.hpp, that is not a stale
# constant, it is a different struct layout: one object writes Deck::wells at
# one offset and another reads it from somewhere else.  The result is not a
# link error, it is a segfault a long way from the edit -- the last one landed
# in make_reservoir, three files away from the header that changed.
#
# -MMD writes a .d file per object listing the headers it actually included;
# -MP adds phony targets so a deleted header does not wedge the build.
#
# -MF is not optional here.  Unlike gcc, nvcc does not put the .d beside the
# -o target: it writes <source-basename>.d into the working directory, so
# compiling src/physics_host.cu produced ./physics_host.d while this makefile
# included src/physics_host.d.  Nothing errored, the include just matched
# nothing, and header tracking silently did nothing at all -- which is how a
# deck.hpp change left physics_host.o compiled against the previous Deck
# layout for a second time.  Name the file explicitly.
DEPFLAGS   = -MMD -MP -MF $(@:.o=.d)

SRC := src/main.cu src/halo.cu src/sequential.cu src/physics_host.cu src/assemble_blackoil.cu \
       src/assemble_compositional.cu src/cpr_solver.cu src/io.cu \
       src/amgx_backend.cu src/field_model.cu src/halo_conn.cu src/pvt_upload.cu \
       src/well_report.cu src/ginkgo_kernels.cu src/nccl_comm.cu
CPPSRC := src/ginkgo_backend.cpp src/deck.cpp src/ecl_io.cpp src/cpgrid.cpp src/schedule.cpp src/equil.cpp src/field_partition.cpp src/units_field.cpp src/pvt_tables.cpp
SRC += $(if $(TC),src/tc_jacobian.cu,)
OBJDIR := $(if $(or $(FP64),$(TC)),build$(if $(FP64),_fp64,)$(if $(TC),_tc,)/,)
EXE    := nvrs_cuda$(if $(FP64),_fp64,)$(if $(TC),_tc,)
OBJ := $(addprefix $(OBJDIR),$(SRC:.cu=.o) $(CPPSRC:.cpp=.o))
DEP := $(OBJ:.o=.d)

# An object with no .d file beside it carries no header dependencies, so no
# header change can ever rebuild it.  That is not hypothetical: objects built
# before the -MF fix above have no .d at all, and a checkout that pulls a
# header change on top of them rebuilds only the .cpp that changed.  The rest
# keep the old layout, and the result is a segfault in a member container a
# long way from the edit -- which is how a cpgrid.hpp change crashed inside a
# std::map operator[] with the reader itself untouched.
#
# So an object without its dependency file is deleted rather than trusted.  It
# costs one recompilation of that file and removes the whole class of failure.
# The configuration the objects were compiled under.  Header dependencies
# cannot express this: turning AmgX off changes a -D flag, not a header, so
# make sees nothing to rebuild.  What follows is cpr_solver.o still calling
# setup_amgx() while amgx_backend.o no longer defines it, and a link failure
# naming three symbols and neither AmgX nor the flag that changed.
CONFIG_SIG   := $(if $(FP64),prec=fp64 ,)$(if $(TC),tc cutlass=$(if $(HAVE_CUTLASS),1,0) ,)arch=$(CUDA_ARCH) amgx=$(if $(HAVE_AMGX),1,0) ginkgo=$(if $(HAVE_GINKGO),1,0) hypre=$(if $(HYPRE_INC),1,0) nccl=$(if $(HAVE_NCCL),1,0)
CONFIG_STAMP := .build_config$(if $(FP64),_fp64,)$(if $(TC),_tc,)
CONFIG_WAS   := $(strip $(shell cat $(CONFIG_STAMP) 2>/dev/null))
ifneq ($(CONFIG_WAS),$(strip $(CONFIG_SIG)))
  ifneq ($(CONFIG_WAS),)
    $(info build configuration changed, rebuilding every object)
    $(info   was: $(CONFIG_WAS))
    $(info   now: $(strip $(CONFIG_SIG)))
    $(shell rm -f $(OBJ) $(OBJ:.o=.d))
  endif
  $(shell printf '%s' '$(strip $(CONFIG_SIG))' > $(CONFIG_STAMP))
endif

UNTRACKED := $(foreach o,$(OBJ),$(if $(wildcard $(o)),$(if $(wildcard $(o:.o=.d)),,$(o))))
ifneq ($(strip $(UNTRACKED)),)
  $(info rebuilding $(words $(UNTRACKED)) object(s) with no header dependencies recorded)
  $(shell rm -f $(UNTRACKED))
endif

# CUTLASS needs relaxed constexpr, and only this file includes it.
$(OBJDIR)src/tc_jacobian.o: NVCCFLAGS += --expt-relaxed-constexpr

$(EXE): $(OBJ)
	@echo "  AmgX: $(if $(HAVE_AMGX),$(AMGX_DIR),not found -- HYPRE only)"
ifneq ($(HAVE_AMGX),1)
	@echo "        AMGX_DIR=$(if $(AMGX_DIR),$(AMGX_DIR),<unset>)"
	@echo "        looked for amgx_c.h under include/, base/include/ and include/*/"
	@echo "        library present: $(if $(wildcard $(AMGX_LIB)/libamgxsh.so),yes at $(AMGX_LIB),no)"
	@echo "        a library without a header is an incomplete install; reinstall with FORCE_AMGX=1"
endif
ifeq ($(TC),1)
	@echo "  Tensor Core: on, $(if $(HAVE_CUTLASS),TF32 with CUTLASS at $(CUTLASS_DIR),no CUTLASS -- TCPRECISION FP32 only)"
endif
	@echo "  NCCL: $(if $(HAVE_NCCL),$(NCCL_LIB),not found -- multi-rank Krylov tail keeps MPI, CUDAGRAPH one rank only)"
	@echo "  Ginkgo: $(if $(HAVE_GINKGO),$(GINKGO_DIR) [$(GINKGO_LIBS)],not found -- LINSOLVER GINKGO falls back to AmgX)"
ifneq ($(HAVE_GINKGO),1)
	@echo "        GINKGO_DIR=$(if $(GINKGO_DIR),$(GINKGO_DIR),<unset>); install with benchmark/install_ginkgo.sh"
endif
	$(NVCC) $(NVCCFLAGS) $(OBJ) -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(DEFS) $(INCLUDES) $(DEPFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(DEFS) $(INCLUDES) $(DEPFLAGS) -c $< -o $@

-include $(DEP)

# *.d catches any stragglers an earlier build wrote to the working directory,
# before -MF was passed.
clean:
	rm -f $(OBJ) $(DEP) *.d $(EXE)
	rm -f $(CONFIG_STAMP)

.PHONY: clean

# The Tensor Core Jacobian against the analytic kernels, on random states.
# Needs a GPU, nothing else at run time.  ./test_tc_jacobian tf32 tests the GEMM.
test_tc_jacobian: tests/test_tc_jacobian.cu src/tc_jacobian.cu src/tc_assemble.inc
	$(NVCC) $(NVCCFLAGS) --expt-relaxed-constexpr $(DEFS) -DNVRS_HAVE_TENSORCORE=1 $(INCLUDES) tests/test_tc_jacobian.cu src/tc_jacobian.cu -o $@ $(LDFLAGS)
