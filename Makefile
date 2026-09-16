# astrosim_fused -- race-free, kernel-fused CUDA build.
#
# Build ONLY on the ARM/GH200 side of Roihu (roihu-gpu.csc.fi):
#   module load nvhpc/26.3
#   make -j
#
# Never run the resulting binary directly on the login node -- it links
# against the CUDA driver and needs an actual GPU, which login nodes do not
# have. Run it through srun/sbatch; see scripts/run_gh200.sbatch.
#
# Override the target GPU architecture if it isn't a GH200 (Hopper, sm_90):
#   make ARCH=sm_80        # e.g. A100
#
# Trade fidelity for ~2x memory bandwidth with single precision:
#   make PRECISION=-DASTROSIM_USE_FLOAT
#
# --use_fast_math is deliberately NOT enabled by default: it swaps exp/log/
# cos for lower-precision intrinsics, which perturbs the RK4 integration and
# the AdEx spike threshold crossing -- a real change to the simulated
# dynamics, not just a speed knob. Opt in explicitly if that trade is wanted:
#   make FASTMATH=--use_fast_math

NVCC       ?= nvcc
ARCH       ?= sm_90
PRECISION  ?=
FASTMATH   ?=
CXXSTD     ?= c++17
OPT        ?= -O3
BIN        := astrosim_fused
SRC        := src/astrosim_fused.cu

.PHONY: all clean run-help
all: $(BIN)

$(BIN): $(SRC)
	$(NVCC) $(OPT) -arch=$(ARCH) -std=$(CXXSTD) -lineinfo $(FASTMATH) $(PRECISION) -o $(BIN) $(SRC)

clean:
	rm -f $(BIN)

run-help: $(BIN)
	@echo "Do not run $(BIN) here -- this is a login node with no GPU."
	@echo "Submit scripts/run_gh200.sbatch instead: sbatch scripts/run_gh200.sbatch"
