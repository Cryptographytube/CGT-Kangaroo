# ============================================================================
#  cryptographytube  -  secp256k1 Pollard Kangaroo ECDLP solver
#  Author: sisujhon
#
#  Builds ONE fat binary that runs on every GPU this CUDA toolkit supports.
#  Architectures are probed against the local nvcc, so the same Makefile works
#  on CUDA 11 (Maxwell..Ampere), CUDA 12 (Pascal..Blackwell) and CUDA 13
#  (Turing..Blackwell, older ones dropped) without editing.
#
#  Usage:
#    make                 build build/cryptographytube(.exe)  [all GPUs]
#    make GPU=native      build only for the card in this machine (fast)
#    make selftest        build the GPU-vs-CPU field self-test
#    make clean           remove build artefacts
# ============================================================================

NVCC     ?= nvcc
TARGET   ?= cryptographytube
BUILDDIR ?= build

SRC_CPP := cgt_field.cpp cgt_ec.cpp cgt_kangaroo.cpp cgt_main.cpp
SRC_CU  := cgt_gpu.cu
HDRS    := cgt_defs.h cgt_uint.h cgt_ec.h cgt_kangaroo.h cgt_gpu.h cgt_gpu_field.cuh

ifeq ($(OS),Windows_NT)
  EXE   := .exe
  MKDIR  = if not exist "$(BUILDDIR)" mkdir "$(BUILDDIR)"
  RMDIR  = if exist "$(BUILDDIR)" rmdir /s /q "$(BUILDDIR)"
  DEVNULL = nul
else
  EXE   :=
  MKDIR  = mkdir -p $(BUILDDIR)
  RMDIR  = rm -rf $(BUILDDIR)
  DEVNULL = /dev/null
endif

OUT := $(BUILDDIR)/$(TARGET)$(EXE)

# ---- architecture probe ----------------------------------------------------
#   52/61   Maxwell / Pascal  (GTX 750, 10xx)
#   70/75   Volta / Turing    (RTX 20xx, GTX 16xx)
#   80/86   Ampere            (RTX 3050 ... 3090)
#   89      Ada               (RTX 4050 ... 4090)
#   90      Hopper
#   100/120 Blackwell         (RTX 5060 ... 5090)
CANDIDATES := 52 61 70 75 80 86 89 90 100 120

# keep only the architectures this nvcc actually accepts
SUPPORTED := $(strip $(foreach a,$(CANDIDATES),\
    $(shell $(NVCC) -arch=compute_$(a) --dryrun -ptx -x cu /dev/null \
        > $(DEVNULL) 2>&1 && echo $(a))))

ifeq ($(GPU),native)
  GENCODE := -arch=native
else ifeq ($(SUPPORTED),)
  # probe unavailable (e.g. no /dev/null on this shell) - fall back to a list
  # that every CUDA >= 11.8 accepts, plus PTX for forward compatibility
  GENCODE := -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_89,code=sm_89 \
             -gencode arch=compute_90,code=sm_90 \
             -gencode arch=compute_90,code=compute_90
else
  NEWEST  := $(lastword $(SUPPORTED))
  GENCODE := $(foreach a,$(SUPPORTED),-gencode arch=compute_$(a),code=sm_$(a)) \
             -gencode arch=compute_$(NEWEST),code=compute_$(NEWEST)
endif

NVCCFLAGS := -O3 --use_fast_math -I . $(GENCODE)

.PHONY: all clean selftest archinfo
all: $(OUT)

archinfo:
	@echo "nvcc      : $(NVCC)"
	@echo "supported : $(SUPPORTED)"
	@echo "gencode   : $(GENCODE)"

$(OUT): $(SRC_CPP) $(SRC_CU) $(HDRS)
	@$(MKDIR)
	$(NVCC) $(NVCCFLAGS) $(SRC_CPP) $(SRC_CU) -o $(OUT)
	@echo BUILD_DONE $(OUT)

selftest: cgt_field.cpp cgt_gpu_selftest.cu cgt_gpu_field.cuh
	@$(MKDIR)
	$(NVCC) $(NVCCFLAGS) cgt_field.cpp cgt_gpu_selftest.cu \
	    -o $(BUILDDIR)/gpu_selftest$(EXE)
	@echo SELFTEST_BUILT $(BUILDDIR)/gpu_selftest$(EXE)

clean:
	@$(RMDIR)
	@echo CLEAN_DONE
