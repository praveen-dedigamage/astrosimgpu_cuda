# astrosim_fused

A standalone, race-free, kernel-fused CUDA reimplementation of
[`astrosimgpu`](../astrosimgpu)'s tripartite neuron-astrocyte model. Lives
outside the `astrosimgpu` tree on purpose: it is a from-scratch GPU-native
program, not a patch to the existing CPU/OpenMP/Kokkos/OpenMP-offload
backends there.

It is one file, `src/astrosim_fused.cu`, plus a `Makefile` and an sbatch
script. Read the file's header comment first -- it explains the design in
full; this README is the short version plus how to build/run/validate it.

## Why "race-free" and "fused" go together here

`astrosimgpu`'s own delivery code (`src/network.cpp::deliver_spikes` /
`deliver_sic`) scatters: a spiking cell pushes its output into every target
it connects to, so concurrent sources landing on the same target need an
atomic add -- which is exactly what its last few commits added
(`#pragma omp atomic` in `deliver_spikes`/`deliver_sic`).

This program gathers instead: connectivity is stored **inverted, by target**,
and every cell reads its own incoming edges and writes only its own state.
No two threads ever touch the same memory, so there is nothing to
atomically protect. That, in turn, is what makes fusing the whole per-step
pipeline into two kernel launches possible: `apply_arrivals`,
`deliver_spikes` and `deliver_sic` from the reference model all become the
gather portion of one kernel each, instead of separate scatter/consume
phases that need a barrier between them.

Two properties of the model's actual parameters make that gather cheap
rather than merely correct:

- Every synapse of a given type shares one scalar weight and one scalar
  delay (`SynapseParams::w_e/w_i/w_n2a/w_a2n`, `d_e/d_i/d_a2n`). No per-edge
  weight or delay is ever stored in the reference code, so the inverted
  connectivity here needs only a bare list of source ids per target -- no
  parallel weight/delay arrays to fetch.
- Tsodyks-Markram short-term-plasticity state depends only on a
  presynaptic neuron's own spike times (the same global `StpParams` drives
  every outgoing connection type from that neuron, updated at the same
  instants). The reference code stores `(x, u)` per synapse; this collapses
  that, exactly, to one `(x, u)` per presynaptic **neuron**. See the file
  header for the full argument.

Net effect: the entire simulation state is structure-of-arrays, every
per-step write is to a location owned by exactly one thread, and the only
atomic operation anywhere in the file is a diagnostic spike counter that
feeds the terminal summary and nothing else.

## Target configuration: neurons and astrocytes scale together

This reproduces `astrosimgpu/scripts/roihu/gen_neuron_scaling_configs.py`'s
fixed-in-degree law exactly (the sweep added in commit `1125b68`):

```
N_E = round(0.8 * N_total)
N_I = N_total - N_E
N_A = round(N_total / 5)          <- astrocytes grow WITH neurons
p_primary = min(1, 80 / N_E)      <- average in-degree held at 80 synapses/neuron
```

so passing `--n 100000` builds 80,000 excitatory + 20,000 inhibitory neurons
and 20,000 astrocytes, `--n 10000000` builds the full ten-million-cell sweep
point, and so on -- neuron and astrocyte counts always grow in the same
proportion. Biology parameters (Li-Rinzel astrocyte constants, AdEx neuron
constants, synapse weights/delays, background Poisson/noise drive) are
copied verbatim from that script, i.e. from `config/use_case.json`.

## Build

Only on the ARM/GH200 side of Roihu, same as `astrosimgpu` itself:

```bash
ssh roihu-gpu.csc.fi          # not roihu-cpu
module load nvhpc/26.3
cd astrosim_cuda_fused
make -j
```

Single precision instead of double, for ~2x memory bandwidth at some cost
to fidelity (the model was validated in double -- see `astrosimgpu/docs/
validation.md`):

```bash
make PRECISION=-DASTROSIM_USE_FLOAT
```

## Run

**Never run the binary on the login node** -- it links the CUDA driver and
refuses to start without a real GPU. Always go through `srun`/`sbatch`:

```bash
sbatch scripts/run_gh200.sbatch
# or, inside an interactive allocation:
srun --partition=gputest --gres=gpu:gh200:1 --time=00:10:00 \
    ./astrosim_fused --n 100000 --sim-ms 5000
```

CLI flags: `--n N_total`, `--seed S`, `--dt 0.1`, `--substeps 1`,
`--pre-ms 1000`, `--sim-ms 5000`, `--block 128`. `--help` prints all of them.

Each run prints connectivity size, build time, steps/second, spikes in the
recorded window, mean firing rate, and final mean calcium / fraction of
astrocytes above the SIC threshold -- enough to sanity-check the run without
adding any host-device traffic inside the timed loop.

## What this does and does not carry over from astrosimgpu

Carried over exactly: the Li-Rinzel astrocyte ODEs and RK4 integrator, the
AdEx neuron + alpha-conductance-cascade ODEs and RK4 integrator (including
the closed-form propagation of the conductance cascade and the refractory
clamp), the tripartite connectivity generation algorithm (geometric-skip
Bernoulli sampling, random astrocyte pools, the `max_astro_out_degree`
safety cap), Tsodyks-Markram STP, the Poisson/Gaussian background drive and
its exact RNG streams (`rng.hpp`'s splitmix64-based counter RNG, reproduced
function-for-function), and the fixed-in-degree scaling law.

Deliberately out of scope, to keep this a focused performance artifact
rather than a second full port of the CLI/JSON/analysis tooling: JSON config
loading (the scaling law is generated in code from `--n`), spike/calcium
recording to disk, and the post-run transient/correlation analysis in
`astrosimgpu/src/analysis.cpp`. Cross-check correctness against the CPU
build at a size small enough to run there (e.g. `--n 500` here against
`config/use_case.json`/`config/neuron_scale_500.json` on the CPU side) by
comparing mean firing rate and the fraction of astrocytes that cross the
SIC threshold -- both are printed by both programs.

## Possible further work

- **CUDA graphs** to remove the two-launch-per-step host overhead entirely;
  not done here because the per-step scalar arguments (`step`,
  `sic_refresh_now`, the noise index) change every iteration and capturing
  that correctly needs care this pass didn't spend.
- **Cooperative-groups persistent kernel** (one grid-wide launch, `grid.
  sync()` between the astrocyte and neuron phases) to remove launch latency
  altogether at very small N where launch overhead dominates.
- Per-edge weight/delay arrays, if a future config ever needs randomized
  per-synapse weights -- the current design's bandwidth advantage comes
  directly from not needing them, so this is a real trade, not a free one.
