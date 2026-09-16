// astrosim_fused.cu
//
// Standalone, race-free, kernel-fused CUDA reimplementation of astrosimgpu's
// tripartite neuron-astrocyte model (Jiang et al. 2025 / Li-Rinzel + AdEx).
//
// -----------------------------------------------------------------------
// WHY THIS IS RACE-FREE BY CONSTRUCTION (not by atomics)
// -----------------------------------------------------------------------
// The reference CPU code (astrosimgpu/src/network.cpp) delivers spikes by
// SCATTERING: a spiking source pushes its contribution into every target it
// is connected to, so two different sources landing on the same target need
// an atomic add. That is exactly what the last few commits in astrosimgpu
// added (#pragma omp atomic in deliver_spikes / deliver_sic).
//
// This program instead delivers by GATHERING: connectivity is stored
// inverted (per TARGET, not per source), and every cell reads its own
// incoming edges and writes only its own state. No two threads ever write
// the same memory location, so there is nothing to race on and no atomics
// are needed anywhere in the simulation state. The only atomic in the whole
// file is a single global counter incremented by spiking threads purely for
// a diagnostic spike count printed to the terminal -- it does not feed back
// into the dynamics.
//
// Two facts about the *actual* parameter set this model uses make the
// gather side of that trade cheap instead of merely "possible":
//
//   1. Every one of {w_e, w_i, w_n2a, w_a2n} and every one of
//      {d_e, d_i, d_a2n} is a SINGLE SCALAR shared by every synapse of that
//      type (see astrosimgpu/include/astrosimgpu/parameters.hpp:
//      SynapseParams). No per-edge weight or delay is ever stored. So the
//      inverted connectivity needs to hold only a source-id list per
//      target -- no parallel weight/delay arrays.
//
//   2. Tsodyks-Markram STP state (x, u, t_last) is updated, for a given
//      source neuron, at exactly the same instants (every time that neuron
//      spikes) with exactly the same elapsed time, on EVERY outgoing edge of
//      that neuron -- because SynapseParams::stp is one struct shared by
//      exc_primary_, inh_primary_ and neuron_astro_ alike. Two edges from
//      the same source therefore carry mathematically identical (x, u)
//      trajectories forever (same initial condition, same update rule
//      applied at the same times). The reference code stores one (x, u)
//      per SYNAPSE anyway; here it collapses, exactly (not approximately),
//      to one (x, u) per PRESYNAPTIC NEURON. That turns "gather scaled by
//      per-edge STP" into "gather raw values, scale once by a global
//      weight" -- the thing that makes an inverted CSR with no payload
//      arrays correct.
//
// Per-step work is exactly two kernel launches:
//   1. astro_step_kernel   -- gather IP3 input, Poisson drive, RK4 Li-Rinzel
//                              integration, write own SIC-history slot.
//   2. neuron_step_kernel  -- gather exc/inh conductance jumps and SIC
//                              current, Poisson drive, Gaussian noise, RK4
//                              AdEx + alpha-cascade integration, write own
//                              spike-history slot (with its own STP state).
// Every array is structure-of-arrays; every access pattern is either a
// unit-stride SoA read/write (the RK4 state) or a CSR gather where
// consecutive threads read consecutive row_start entries. That is what
// "maximum achievable bandwidth and occupancy" means for this workload: no
// thread ever waits on another thread's write, so the scheduler can keep
// every SM saturated with independent memory-bound work.
//
// -----------------------------------------------------------------------
// TARGET CONFIGURATION: the fixed-in-degree neuron-astrocyte scaling law
// -----------------------------------------------------------------------
// This reproduces astrosimgpu/scripts/roihu/gen_neuron_scaling_configs.py
// exactly (commit 1125b68, "Add astrocyte out-degree cap, sparse
// connectivity generation, and a fixed-in-degree neuron-scaling sweep"):
//
//   N_E = round(0.8 * N_total)
//   N_I = N_total - N_E
//   N_A = round(N_total / 5)                 // astrocytes grow WITH neurons
//   p_primary = min(1, K_SYN / N_E)          // K_SYN = 80 synapses/neuron
//
// so astrocyte count and synapse count both grow linearly with N_total and
// average in-degree (K_SYN) and average astrocyte in-degree (~K_SYN * 0.2 *
// N_E / N_A) stay constant as the network is scaled up -- neurons and
// astrocytes grow proportionally, as requested.
//
// -----------------------------------------------------------------------
// Build (on roihu-gpu.csc.fi, the ARM/GH200 login node -- see ../README.md):
//   module load nvhpc/26.3
//   nvcc -O3 -arch=sm_90 -std=c++17 -o astrosim_fused src/astrosim_fused.cu
// Run ONLY through srun/sbatch (see ../scripts/run_gh200.sbatch); a login
// node has no GPU driver and the binary will not start there.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <random>
#include <fstream>

// ----------------------------- floating point ---------------------------
// double by default: this is the precision the model was validated at
// (docs/validation.md). Build with -DASTROSIM_USE_FLOAT to trade fidelity
// for roughly 2x memory bandwidth.
#if defined(ASTROSIM_USE_FLOAT)
using real = float;
#else
using real = double;
#endif
using index_t = std::uint32_t;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t _e = (call);                                                  \
        if (_e != cudaSuccess) {                                                  \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                         cudaGetErrorString(_e));                                  \
            std::abort();                                                         \
        }                                                                         \
    } while (0)

// =========================================================================
// RNG -- splitmix64-style counter RNG, bit-for-bit the same algorithm as
// astrosimgpu/include/astrosimgpu/rng.hpp, so a run here draws the same
// Poisson/Gaussian statistics the reference model would for the same
// (seed, stream, lambda/std). __host__ __device__ so setup (host) and the
// per-step kernels (device) share one implementation.
// =========================================================================

__host__ __device__ inline std::uint64_t rng_bits(std::uint64_t seed, std::uint64_t stream,
                                                   std::uint64_t counter) {
    std::uint64_t z = seed;
    z += stream * 0x9E3779B97F4A7C15ULL;
    z += counter * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

__host__ __device__ inline real rng_uniform(std::uint64_t seed, std::uint64_t stream,
                                            std::uint64_t counter) {
    return static_cast<real>(rng_bits(seed, stream, counter) >> 11) *
           static_cast<real>(1.0 / 9007199254740992.0);
}

__host__ __device__ inline real rng_normal(std::uint64_t seed, std::uint64_t stream) {
    real u1 = rng_uniform(seed, stream, 0);
    if (u1 <= static_cast<real>(1e-300)) u1 = static_cast<real>(1e-300);
    const real u2 = rng_uniform(seed, stream, 1);
    return std::sqrt(static_cast<real>(-2.0) * std::log(u1)) *
           std::cos(static_cast<real>(6.283185307179586476925286766559) * u2);
}

// Knuth's algorithm, fresh (seed,stream) per draw -- matches
// CounterRng::poisson() called on a freshly constructed CounterRng, as
// drive_astrocytes()/NeuronPopulation::update() do in the reference code.
__host__ __device__ inline int rng_poisson(std::uint64_t seed, std::uint64_t stream, real lambda) {
    if (lambda <= 0.0) return 0;
    std::uint64_t c = 0;
    if (lambda > 30.0) {
        // Normal approximation, matching CounterRng::poisson's fallback.
        real u1 = rng_uniform(seed, stream, c++);
        if (u1 <= static_cast<real>(1e-300)) u1 = static_cast<real>(1e-300);
        const real u2 = rng_uniform(seed, stream, c++);
        const real z = std::sqrt(static_cast<real>(-2.0) * std::log(u1)) *
                        std::cos(static_cast<real>(6.283185307179586476925286766559) * u2);
        const real v = lambda + std::sqrt(lambda) * z;
        return v < 0.0 ? 0 : static_cast<int>(v + 0.5);
    }
    const real limit = std::exp(-lambda);
    real product = rng_uniform(seed, stream, c++);
    int count = 0;
    while (product > limit) {
        ++count;
        product *= rng_uniform(seed, stream, c++);
    }
    return count;
}

// Host-only stateful counter RNG (sequential draws), used exclusively during
// one-time connectivity/parameter generation -- a direct port of
// astrosimgpu/include/astrosimgpu/rng.hpp's CounterRng.
class HostRng {
public:
    HostRng(std::uint64_t seed, std::uint64_t stream) : seed_(seed), stream_(stream) {}
    real uniform() { return rng_uniform(seed_, stream_, counter_++); }
    real normal() {
        if (has_spare_) { has_spare_ = false; return spare_; }
        real u1;
        do { u1 = uniform(); } while (u1 <= 1e-300);
        const real u2 = uniform();
        const real r = std::sqrt(-2.0 * std::log(u1));
        const real theta = 6.283185307179586476925286766559 * u2;
        spare_ = r * std::sin(theta);
        has_spare_ = true;
        return r * std::cos(theta);
    }
    real normal(real mean, real stddev) { return mean + stddev * normal(); }
    real normal_redraw(real mean, real stddev, real lo, real hi, int max_tries = 1000) {
        for (int i = 0; i < max_tries; ++i) {
            const real v = normal(mean, stddev);
            if (v >= lo && v <= hi) return v;
        }
        return std::min(std::max(mean, lo), hi);
    }

private:
    std::uint64_t seed_, stream_, counter_ = 0;
    bool has_spare_ = false;
    real spare_ = 0.0;
};

inline real draw_scaled(real mean, bool randomize_enabled, real lower, real upper, real var,
                        HostRng& rng) {
    if (!randomize_enabled || var <= 0.0) return mean;
    const real lo = mean < 0.0 ? mean * upper : mean * lower;
    const real hi = mean < 0.0 ? mean * lower : mean * upper;
    return rng.normal_redraw(mean, var * std::abs(mean), lo, hi);
}

// =========================================================================
// Model constants
// =========================================================================

struct AstroConstants {
    real Kd_IP3_1, Kd_IP3_2, Kd_act, Kd_inh, Km_SERCA;
    real k_IP3R, rate_IP3R, rate_L, rate_SERCA, ratio_ER_cyt;
    real SIC_scale, SIC_th;
};

struct NeuronConstants {
    real C_m, g_L, E_L, V_th, Delta_T, a, tau_w;
    real V_peak, t_ref, E_ex, E_in, tau_syn_ex, tau_syn_in, I_e;
    real psc_init_ex, psc_init_in;  // e / tau_syn, population-uniform
};

// Li-Rinzel right-hand side. Identical maths to
// astrosimgpu/include/astrosimgpu/astrocyte_kernel.hpp::astro_derivatives.
__device__ __forceinline__ void astro_derivatives(const AstroConstants& c, real Ca_tot, real IP3_0,
                                                   real tau_IP3, real Ca, real IP3, real h,
                                                   real noise, real& dCa, real& dIP3, real& dh) {
    const real Ca_ER = (Ca_tot - Ca) / c.ratio_ER_cyt;
    const real m_inf = IP3 / (IP3 + c.Kd_IP3_1);
    const real n_inf = Ca / (Ca + c.Kd_act);
    const real m3 = m_inf * m_inf * m_inf;
    const real n3 = n_inf * n_inf * n_inf;
    const real h3 = h * h * h;
    const real J_channel = c.ratio_ER_cyt * c.rate_IP3R * m3 * n3 * h3 * (Ca_ER - Ca);
    const real J_pump = c.rate_SERCA * Ca * Ca / (c.Km_SERCA * c.Km_SERCA + Ca * Ca);
    const real J_leak = c.ratio_ER_cyt * c.rate_L * (Ca_ER - Ca);
    const real alpha_h = c.k_IP3R * c.Kd_inh * (IP3 + c.Kd_IP3_1) / (IP3 + c.Kd_IP3_2);
    const real beta_h = c.k_IP3R * Ca;
    dCa = J_channel - J_pump + J_leak + noise;
    dIP3 = (IP3_0 - IP3) / tau_IP3;
    dh = alpha_h * (1.0 - h) - beta_h * h;
}

__device__ __forceinline__ void astro_advance(const AstroConstants& c, real Ca_tot, real IP3_0,
                                              real tau_IP3, real delta_IP3, real ip3_input,
                                              real noise, real h_step, int substeps, real& Ca,
                                              real& IP3, real& h) {
    if (ip3_input != 0.0) IP3 += delta_IP3 * ip3_input;
    for (int s = 0; s < substeps; ++s) {
        real k1Ca, k1IP3, k1h, k2Ca, k2IP3, k2h, k3Ca, k3IP3, k3h, k4Ca, k4IP3, k4h;
        astro_derivatives(c, Ca_tot, IP3_0, tau_IP3, Ca, IP3, h, noise, k1Ca, k1IP3, k1h);
        astro_derivatives(c, Ca_tot, IP3_0, tau_IP3, Ca + 0.5 * h_step * k1Ca,
                          IP3 + 0.5 * h_step * k1IP3, h + 0.5 * h_step * k1h, noise, k2Ca, k2IP3,
                          k2h);
        astro_derivatives(c, Ca_tot, IP3_0, tau_IP3, Ca + 0.5 * h_step * k2Ca,
                          IP3 + 0.5 * h_step * k2IP3, h + 0.5 * h_step * k2h, noise, k3Ca, k3IP3,
                          k3h);
        astro_derivatives(c, Ca_tot, IP3_0, tau_IP3, Ca + h_step * k3Ca, IP3 + h_step * k3IP3,
                          h + h_step * k3h, noise, k4Ca, k4IP3, k4h);
        Ca += (h_step / 6.0) * (k1Ca + 2.0 * k2Ca + 2.0 * k3Ca + k4Ca);
        IP3 += (h_step / 6.0) * (k1IP3 + 2.0 * k2IP3 + 2.0 * k3IP3 + k4IP3);
        h += (h_step / 6.0) * (k1h + 2.0 * k2h + 2.0 * k3h + k4h);
        Ca = Ca < 0.0 ? 0.0 : (Ca > Ca_tot ? Ca_tot : Ca);
        h = h < 0.0 ? 0.0 : (h > 1.0 ? 1.0 : h);
        IP3 = IP3 < 0.0 ? 0.0 : IP3;
    }
}

// AdEx right-hand side. Identical maths to astrosimgpu/src/neuron.cpp::derivatives.
__device__ __forceinline__ void neuron_derivatives(const NeuronConstants& p, real V, real w,
                                                    real g_ex, real g_in, real I_ext, real& dV,
                                                    real& dw) {
    const real Vc = V < p.V_peak ? V : p.V_peak;
    const real I_spike = p.g_L * p.Delta_T * std::exp((Vc - p.V_th) / p.Delta_T);
    const real I_syn_ex = g_ex * (Vc - p.E_ex);
    const real I_syn_in = g_in * (Vc - p.E_in);
    dV = (-p.g_L * (Vc - p.E_L) + I_spike - I_syn_ex - I_syn_in - w + p.I_e + I_ext) / p.C_m;
    dw = (p.a * (Vc - p.E_L) - w) / p.tau_w;
}

// =========================================================================
// Inverted (gather) connectivity: CSR keyed by TARGET. No weight, no delay,
// no STP payload -- see the file header for why those collapse to scalars.
// =========================================================================
struct InCSR {
    index_t* row_start = nullptr;  // size = n_targets + 1
    index_t* src = nullptr;        // size = row_start[n_targets]
    index_t n_targets = 0;
    index_t n_edges = 0;
};

// Host-side CSR-by-target builder: counts in-degree, prefix-sums, scatters.
struct HostCSR {
    std::vector<index_t> row_start;
    std::vector<index_t> src;
    void build(index_t n_targets, const std::vector<std::pair<index_t, index_t>>& edges /* (src,dst) */) {
        row_start.assign(n_targets + 1, 0);
        for (auto& e : edges) row_start[e.second + 1]++;
        for (index_t i = 0; i < n_targets; ++i) row_start[i + 1] += row_start[i];
        src.assign(edges.size(), 0);
        std::vector<index_t> cursor(row_start.begin(), row_start.end() - 1);
        for (auto& e : edges) src[cursor[e.second]++] = e.first;
    }
};

InCSR upload_csr(const HostCSR& h) {
    InCSR d;
    d.n_targets = static_cast<index_t>(h.row_start.size() - 1);
    d.n_edges = static_cast<index_t>(h.src.size());
    CUDA_CHECK(cudaMalloc(&d.row_start, h.row_start.size() * sizeof(index_t)));
    CUDA_CHECK(cudaMemcpy(d.row_start, h.row_start.data(), h.row_start.size() * sizeof(index_t),
                          cudaMemcpyHostToDevice));
    if (!h.src.empty()) {
        CUDA_CHECK(cudaMalloc(&d.src, h.src.size() * sizeof(index_t)));
        CUDA_CHECK(cudaMemcpy(d.src, h.src.data(), h.src.size() * sizeof(index_t),
                              cudaMemcpyHostToDevice));
    }
    return d;
}

// =========================================================================
// Device state, structure-of-arrays throughout.
// =========================================================================
struct DeviceState {
    // Astrocytes.
    index_t n_astro = 0;
    real *Ca = nullptr, *IP3 = nullptr, *h = nullptr;
    real *Ca_tot = nullptr, *IP3_0 = nullptr, *tau_IP3 = nullptr, *delta_IP3 = nullptr;

    // Neurons (exc laid out first, then inh -- same convention as the CPU code).
    index_t n_neurons = 0, n_exc = 0, n_inh = 0;
    real *V = nullptr, *w = nullptr, *g_ex = nullptr, *dg_ex = nullptr, *g_in = nullptr,
         *dg_in = nullptr;
    real *I_sic_hold = nullptr;
    int* refractory_steps = nullptr;
    real *V_reset = nullptr, *b = nullptr;  // the only two randomised-per-cell neuron params

    // Consolidated STP state, one entry per PRESYNAPTIC NEURON (see file
    // header: exact, not approximate, collapse from per-synapse state).
    real *stp_x = nullptr, *stp_u = nullptr, *stp_t_last = nullptr;

    // History rings, slot-major so all cells at one slot are contiguous
    // (coalesced write by every thread of a launch).
    real* ring_neuron = nullptr;  // [ring_slots][n_neurons]: 0, or this neuron's STP multiplier if it just spiked
    real* ring_sic = nullptr;     // [ring_slots][n_astro]: this astrocyte's live SIC factor

    // Inverted connectivity.
    InCSR inc_exc;   // per neuron:   incoming excitatory primary sources
    InCSR inc_inh;   // per neuron:   incoming inhibitory primary sources
    InCSR inc_sic;   // per neuron:   incoming astrocyte (SIC) sources
    InCSR inc_e2a;   // per astrocyte: incoming excitatory third-factor sources

    // Diagnostics.
    unsigned long long* spike_counter = nullptr;
    unsigned char* ever_active = nullptr;  // per astrocyte: has Ca ever crossed SIC_th?
};

// =========================================================================
// Kernel 1: astrocyte gather + Poisson drive + RK4 + own SIC-history write.
// One thread per astrocyte. Every array access is unit-stride SoA except the
// CSR gather, which walks a contiguous row of source ids.
// =========================================================================
__global__ void astro_step_kernel(DeviceState s, AstroConstants c, real h_step, int substeps,
                                  real dt, std::int64_t step, int ring_slots, int d_e2a_steps,
                                  real w_n2a, real poiss_rate_astro, real poiss_weight_astro,
                                  real noise_std, bool independent_noise, real shared_noise,
                                  std::uint64_t seed) {
    const index_t a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= s.n_astro) return;

    // Gather: sum this astrocyte's incoming third-factor neurons' live
    // output at the delayed slot. Read-only w.r.t. every other thread.
    const int read_slot = static_cast<int>((step + ring_slots - d_e2a_steps) % ring_slots);
    const real* row = s.ring_neuron + static_cast<std::size_t>(read_slot) * s.n_neurons;
    real ip3_gathered = 0.0;
    for (index_t k = s.inc_e2a.row_start[a]; k < s.inc_e2a.row_start[a + 1]; ++k) {
        ip3_gathered += row[s.inc_e2a.src[k]];
    }
    real ip3_input = w_n2a * ip3_gathered;

    // Poisson background drive to IP3, fused in (matches Network::drive_astrocytes).
    if (poiss_rate_astro > 0.0) {
        const real lambda = poiss_rate_astro * dt * 1e-3;
        const int events = rng_poisson(seed ^ 0xC2B2AE3D27D4EB4FULL,
                                       static_cast<std::uint64_t>(step) * 1000003ULL + a, lambda);
        if (events > 0) ip3_input += poiss_weight_astro * static_cast<real>(events);
    }

    // 0x9E3779B97F4A7C15 matches astrosimgpu/src/astrocyte.cpp's noise_seed
    // (seed ^ that constant), which AstrocytePopulation::update passes into
    // the astrocyte kernel for exactly this draw.
    const real noise = (noise_std > 0.0 && independent_noise)
                            ? noise_std * rng_normal(seed ^ 0x9E3779B97F4A7C15ULL,
                                                     static_cast<std::uint64_t>(step) * 1000003ULL + a)
                            : shared_noise;

    real ca = s.Ca[a], ip3 = s.IP3[a], hh = s.h[a];
    astro_advance(c, s.Ca_tot[a], s.IP3_0[a], s.tau_IP3[a], s.delta_IP3[a], ip3_input, noise,
                 h_step, substeps, ca, ip3, hh);
    s.Ca[a] = ca;
    s.IP3[a] = ip3;
    s.h[a] = hh;

    // sic_factor(): astrosimgpu/src/astrocyte.cpp::sic_factor, verbatim.
    const real y = (ca - c.SIC_th) * 1000.0;
    const real factor = (y <= 1.0) ? 0.0 : c.SIC_scale * std::log(y);

    // Diagnostic-only, own-cell write: has this astrocyte EVER crossed
    // threshold, not just at this instant? Directly comparable to the CPU
    // reference's "astrocytes with transients" count, unlike a single
    // end-of-run snapshot of Ca.
    if (factor > 0.0) s.ever_active[a] = 1;

    // Own row, own slot: every astrocyte writes exactly one element here.
    s.ring_sic[static_cast<std::size_t>(step % ring_slots) * s.n_astro + a] = factor;
}

// =========================================================================
// Kernel 2: neuron gather (exc/inh/SIC) + Poisson + noise + RK4 AdEx + own
// spike-history write. One thread per neuron.
// =========================================================================
__global__ void neuron_step_kernel(DeviceState s, NeuronConstants pe, NeuronConstants pi,
                                   real h_step, int substeps, real dt, std::int64_t step,
                                   int ring_slots, int d_e_steps, int d_i_steps, int d_a2n_steps,
                                   real w_e, real w_i, real w_a2n, bool sic_refresh_now,
                                   real poiss_rate_exc, real poiss_weight_exc,
                                   real poiss_rate_inh, real poiss_weight_inh, real noise_std_exc,
                                   bool indep_noise_exc, real shared_noise_exc, real noise_std_inh,
                                   bool indep_noise_inh, real shared_noise_inh,
                                   std::uint64_t noise_idx_exc, std::uint64_t noise_idx_inh,
                                   real stp_U, real stp_tau_rec, real stp_tau_fac, bool stp_enabled,
                                   std::uint64_t seed) {
    const index_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= s.n_neurons) return;
    const bool excitatory = j < s.n_exc;
    const NeuronConstants& p = excitatory ? pe : pi;

    // --- Gather: excitatory primary conductance jump ---
    const int slot_ex = static_cast<int>((step + ring_slots - d_e_steps) % ring_slots);
    const real* row_ex = s.ring_neuron + static_cast<std::size_t>(slot_ex) * s.n_neurons;
    real exc_sum = 0.0;
    for (index_t k = s.inc_exc.row_start[j]; k < s.inc_exc.row_start[j + 1]; ++k) {
        exc_sum += row_ex[s.inc_exc.src[k]];
    }

    // --- Gather: inhibitory primary conductance jump ---
    const int slot_in = static_cast<int>((step + ring_slots - d_i_steps) % ring_slots);
    const real* row_in = s.ring_neuron + static_cast<std::size_t>(slot_in) * s.n_neurons;
    real inh_sum = 0.0;
    for (index_t k = s.inc_inh.row_start[j]; k < s.inc_inh.row_start[j + 1]; ++k) {
        inh_sum += row_in[s.inc_inh.src[k]];
    }

    real exc_input = w_e * exc_sum;      // w_e > 0
    real inh_input = (-w_i) * inh_sum;   // w_i < 0; inh_input accumulates a magnitude, as in NeuronPopulation::add_synaptic_input

    // --- Poisson background drive. NEST convention: a poisson_generator with
    // positive weight always drives the excitatory conductance, regardless
    // of which population the target cell belongs to (see
    // NeuronPopulation::update: `exc_input_[cell] += ...` unconditionally). ---
    const real poiss_rate = excitatory ? poiss_rate_exc : poiss_rate_inh;
    const real poiss_weight = excitatory ? poiss_weight_exc : poiss_weight_inh;
    if (poiss_rate > 0.0) {
        const real lambda = poiss_rate * dt * 1e-3;
        const int events = rng_poisson(seed ^ 0x2545F4914F6CDD1DULL,
                                       static_cast<std::uint64_t>(step) * 1000003ULL + j, lambda);
        if (events > 0) exc_input += poiss_weight * static_cast<real>(events);
    }

    // --- SIC: gather only on refresh steps, else hold the last value. ---
    if (sic_refresh_now) {
        const int slot_sic = static_cast<int>((step + ring_slots - d_a2n_steps) % ring_slots);
        const real* row_sic = s.ring_sic + static_cast<std::size_t>(slot_sic) * s.n_astro;
        real sic_sum = 0.0;
        for (index_t k = s.inc_sic.row_start[j]; k < s.inc_sic.row_start[j + 1]; ++k) {
            sic_sum += row_sic[s.inc_sic.src[k]];
        }
        s.I_sic_hold[j] = w_a2n * sic_sum;
    }
    const real I_sic = s.I_sic_hold[j];

    // --- Gaussian noise current, independent or shared, matching NeuronPopulation::update. ---
    real I_ext = I_sic;
    const real noise_std = excitatory ? noise_std_exc : noise_std_inh;
    if (noise_std > 0.0) {
        if (excitatory ? indep_noise_exc : indep_noise_inh) {
            const std::uint64_t idx = excitatory ? noise_idx_exc : noise_idx_inh;
            I_ext += noise_std * rng_normal(seed ^ 0x9E3779B185EBCA87ULL,
                                            idx * 1000033ULL + static_cast<std::uint64_t>(j));
        } else {
            I_ext += excitatory ? shared_noise_exc : shared_noise_inh;
        }
    }

    // --- Apply the synaptic jump once per communication step (before the
    // RK4 substep loop), exactly as NeuronPopulation::update does. ---
    real V = s.V[j], w = s.w[j];
    real g_ex = s.g_ex[j], dg_ex = s.dg_ex[j] + exc_input * p.psc_init_ex;
    real g_in = s.g_in[j], dg_in = s.dg_in[j] + inh_input * p.psc_init_in;
    int refractory = s.refractory_steps[j];
    bool spiked = false;

    for (int sub = 0; sub < substeps; ++sub) {
        const real ex_decay = std::exp(-h_step / p.tau_syn_ex);
        const real in_decay = std::exp(-h_step / p.tau_syn_in);
        const real g_ex_next = ex_decay * (g_ex + h_step * dg_ex);
        const real dg_ex_next = ex_decay * dg_ex;
        const real g_in_next = in_decay * (g_in + h_step * dg_in);
        const real dg_in_next = in_decay * dg_in;

        if (refractory > 0) {
            V = s.V_reset[j];
            real dV, dw;
            neuron_derivatives(p, V, w, g_ex, g_in, I_ext, dV, dw);
            w += h_step * dw;
        } else {
            const real g_ex_mid = 0.5 * (g_ex + g_ex_next);
            const real g_in_mid = 0.5 * (g_in + g_in_next);
            real k1V, k1w, k2V, k2w, k3V, k3w, k4V, k4w;
            neuron_derivatives(p, V, w, g_ex, g_in, I_ext, k1V, k1w);
            neuron_derivatives(p, V + 0.5 * h_step * k1V, w + 0.5 * h_step * k1w, g_ex_mid,
                               g_in_mid, I_ext, k2V, k2w);
            neuron_derivatives(p, V + 0.5 * h_step * k2V, w + 0.5 * h_step * k2w, g_ex_mid,
                               g_in_mid, I_ext, k3V, k3w);
            neuron_derivatives(p, V + h_step * k3V, w + h_step * k3w, g_ex_next, g_in_next, I_ext,
                               k4V, k4w);
            V += (h_step / 6.0) * (k1V + 2.0 * k2V + 2.0 * k3V + k4V);
            w += (h_step / 6.0) * (k1w + 2.0 * k2w + 2.0 * k3w + k4w);
        }

        g_ex = g_ex_next; dg_ex = dg_ex_next;
        g_in = g_in_next; dg_in = dg_in_next;

        if (refractory > 0) {
            --refractory;
        } else if (V >= p.V_peak) {
            V = s.V_reset[j];
            w += s.b[j];
            refractory = static_cast<int>(p.t_ref / h_step + 0.5);
            spiked = true;
        }
    }

    s.V[j] = V; s.w[j] = w;
    s.g_ex[j] = g_ex; s.dg_ex[j] = dg_ex;
    s.g_in[j] = g_in; s.dg_in[j] = dg_in;
    s.refractory_steps[j] = refractory;

    // --- Own consolidated STP update + own spike-history write. This is the
    // only per-neuron write in the kernel that is not purely local state:
    // it is written to ring_neuron[step % ring_slots][j], a location no
    // other thread ever touches. ---
    real out_value = 0.0;
    if (spiked) {
        real x = s.stp_x[j], u = s.stp_u[j];
        if (stp_enabled) {
            const real t_now = static_cast<real>(step) * dt;
            const real elapsed = t_now - s.stp_t_last[j];
            const real x_decay = std::exp(-elapsed / stp_tau_rec);
            const real u_decay = stp_tau_fac < 1e-10 ? 0.0 : std::exp(-elapsed / stp_tau_fac);
            x = 1.0 + (x - x * u - 1.0) * x_decay;
            u = stp_U + u * (1.0 - stp_U) * u_decay;
            s.stp_x[j] = x; s.stp_u[j] = u; s.stp_t_last[j] = t_now;
            out_value = x * u;
        } else {
            out_value = 1.0;
        }

        // Diagnostic-only atomic: this is the ONLY atomic operation anywhere
        // in the simulation, and it feeds a terminal counter, never any
        // simulated state. It does not compromise the race-free design
        // above -- it is a scalar telemetry reduction, not part of the model.
        atomicAdd(s.spike_counter, 1ULL);
    }
    s.ring_neuron[static_cast<std::size_t>(step % ring_slots) * s.n_neurons + j] = out_value;
}

// =========================================================================
// Host: connectivity generation, mirroring astrosimgpu/src/network.cpp's
// build_primary_connections (geometric-skip Bernoulli sampling + random
// astrocyte pools + max_astro_out_degree cap), then inverted to per-target
// CSR for the gather kernels above.
// =========================================================================
struct ScalingConfig {
    index_t n_total = 1000;
    index_t n_exc = 0, n_inh = 0, n_astro = 0;
    real p_primary = 0.0;
    static constexpr real K_SYN = 80.0;
    static constexpr real P_THIRD = 0.2;
    static constexpr real E_FRACTION = 0.8;
    static constexpr real ASTRO_RATIO = 5.0;
    static constexpr int POOL_SIZE = 5;
    static constexpr index_t MAX_ASTRO_OUT_DEGREE = 10000;

    static ScalingConfig make(index_t n_total) {
        ScalingConfig cfg;
        cfg.n_total = n_total;
        cfg.n_exc = std::max<index_t>(1, static_cast<index_t>(std::lround(n_total * E_FRACTION)));
        cfg.n_inh = std::max<index_t>(1, n_total - cfg.n_exc);
        cfg.n_astro = std::max<index_t>(1, static_cast<index_t>(std::lround(n_total / ASTRO_RATIO)));
        cfg.p_primary = std::min<real>(1.0, K_SYN / static_cast<real>(cfg.n_exc));
        return cfg;
    }
};

index_t geometric_skip(real p, index_t remaining, HostRng& rng) {
    if (remaining == 0 || p >= 1.0) return 0;
    if (p <= 0.0) return remaining;
    real u = rng.uniform();
    if (u >= 1.0) u = std::nextafter(static_cast<real>(1.0), static_cast<real>(0.0));
    const real g = std::floor(std::log1p(-u) / std::log1p(-p));
    return (g >= static_cast<real>(remaining)) ? remaining : static_cast<index_t>(g);
}

struct Connectivity {
    HostCSR inc_exc, inc_inh, inc_sic, inc_e2a;
    index_t n_exc_edges = 0, n_inh_edges = 0, n_sic_edges = 0, n_e2a_edges = 0;
};

Connectivity build_connectivity(const ScalingConfig& cfg, std::uint64_t seed) {
    const index_t n_exc = cfg.n_exc, n_inh = cfg.n_inh, n_astro = cfg.n_astro;
    const index_t n_neurons = n_exc + n_inh;
    HostRng rng(seed, 0x5EEDULL);

    // Skip past the astrocyte and neuron per-cell parameter draws so the
    // connectivity stream lines up with astrosimgpu's own draw order
    // (Network::build calls astro_.build then neurons_.build before
    // build_primary_connections). Those draws are redone with an identical
    // formula in init_populations(); reproducing the ORDER here keeps this
    // program's connectivity statistically equivalent to a from-scratch
    // build at the same seed even though the two draw sequences are kept
    // physically separate (see init_populations()).
    for (index_t i = 0; i < n_astro; ++i) { rng.normal(); rng.normal(); rng.normal(); rng.normal(); }
    for (index_t i = 0; i < n_neurons; ++i) { rng.normal(); rng.normal(); }

    // Random astrocyte pools, POOL_SIZE=5 per postsynaptic neuron, rejection
    // sampling (mirrors network.cpp's PoolType::Random branch; ASTRO_RATIO=5
    // keeps take*4 < n_astro in every case this sweep reaches, so the exact
    // fallback below is dead code here but included for fidelity at tiny N).
    std::vector<std::vector<index_t>> pool(n_neurons);
    const int take = std::min<int>(ScalingConfig::POOL_SIZE, static_cast<int>(n_astro));
    for (index_t post = 0; post < n_neurons; ++post) {
        pool[post].reserve(take);
        if (take > 0 && static_cast<std::int64_t>(take) * 4 >= static_cast<std::int64_t>(n_astro)) {
            std::vector<index_t> candidates(n_astro);
            for (index_t i = 0; i < n_astro; ++i) candidates[i] = i;
            for (int k = 0; k < take; ++k) {
                const auto pick = static_cast<std::size_t>(rng.uniform() * (candidates.size() - k)) + k;
                std::swap(candidates[k], candidates[std::min(pick, candidates.size() - 1)]);
                pool[post].push_back(candidates[k]);
            }
        } else {
            while (static_cast<int>(pool[post].size()) < take) {
                const auto candidate = static_cast<index_t>(rng.uniform() * static_cast<double>(n_astro)) % n_astro;
                bool dup = false;
                for (index_t existing : pool[post]) if (existing == candidate) { dup = true; break; }
                if (!dup) pool[post].push_back(candidate);
            }
        }
    }

    std::vector<index_t> astro_out_degree(n_astro, 0);  // max_astro_out_degree cap
    std::vector<std::pair<index_t, index_t>> e_exc, e_inh, e_e2a, e_sic;  // (src, dst)
    // Expected edge counts: each of the n_exc/n_inh presynaptic loops draws
    // Bernoulli(p_primary) against its post-population, so expected edges
    // per pre-loop ~= p_primary * post_count; p_primary = K_SYN / n_exc
    // keeps that near K_SYN regardless of network size. Reserve generously
    // (x1.3) to absorb variance without the vector reallocating/copying
    // itself at 10,000,000-neuron scale.
    const std::size_t edge_estimate =
        static_cast<std::size_t>(ScalingConfig::K_SYN * n_neurons * 1.3) + 64;
    e_exc.reserve(edge_estimate);
    e_inh.reserve(edge_estimate);
    e_e2a.reserve(static_cast<std::size_t>(edge_estimate * ScalingConfig::P_THIRD) + 64);
    e_sic.reserve(static_cast<std::size_t>(edge_estimate * ScalingConfig::P_THIRD) + 64);

    auto connect_block = [&](index_t post_offset, index_t post_count) {
        for (index_t pre = 0; pre < n_exc; ++pre) {
            index_t pos = 0;
            while (pos < post_count) {
                pos += geometric_skip(cfg.p_primary, post_count - pos, rng);
                if (pos >= post_count) break;
                const index_t post = post_offset + pos;
                ++pos;
                if (pre == post) continue;  // no autapses
                e_exc.emplace_back(pre, post);

                if (n_astro == 0 || rng.uniform() >= ScalingConfig::P_THIRD) continue;
                const auto& pl = pool[post];
                const index_t astro = pl[static_cast<std::size_t>(rng.uniform() * pl.size()) % pl.size()];
                if (astro_out_degree[astro] >= ScalingConfig::MAX_ASTRO_OUT_DEGREE) continue;
                e_e2a.emplace_back(pre, astro);   // neuron -> astro (third factor)
                e_sic.emplace_back(astro, post);  // astro -> neuron (SIC), unique_third_out=false so always added
                ++astro_out_degree[astro];
            }
        }
    };
    connect_block(0, n_exc);
    connect_block(n_exc, n_inh);

    for (index_t pre = 0; pre < n_inh; ++pre) {
        const index_t pre_global = n_exc + pre;
        index_t pos = 0;
        while (pos < n_neurons) {
            pos += geometric_skip(cfg.p_primary, n_neurons - pos, rng);
            if (pos >= n_neurons) break;
            const index_t post = pos;
            ++pos;
            if (pre_global == post) continue;
            e_inh.emplace_back(pre, post);
        }
    }

    Connectivity out;
    out.inc_exc.build(n_neurons, e_exc);
    out.inc_inh.build(n_neurons, e_inh);
    out.inc_sic.build(n_neurons, e_sic);
    out.inc_e2a.build(n_astro, e_e2a);
    out.n_exc_edges = static_cast<index_t>(e_exc.size());
    out.n_inh_edges = static_cast<index_t>(e_inh.size());
    out.n_sic_edges = static_cast<index_t>(e_sic.size());
    out.n_e2a_edges = static_cast<index_t>(e_e2a.size());
    return out;
}

// =========================================================================
// Biology parameters -- copied verbatim from
// astrosimgpu/scripts/roihu/gen_neuron_scaling_configs.py, which is itself
// astrosimgpu/config/use_case.json's biology blocks.
// =========================================================================
struct Biology {
    // Astrocyte (Li-Rinzel).
    real Ca_tot = 1.8958264782990153, IP3_0 = 0.010925583871830699;
    real tau_IP3 = 1058.7589568400845, delta_IP3 = 0.04911592657233997;
    real SIC_scale = 1.0, Kd_IP3_1 = 0.13, Kd_IP3_2 = 0.9434, Kd_act = 0.08234, Kd_inh = 1.049;
    real Km_SERCA = 0.1, SIC_th = 0.19669, k_IP3R = 0.0002, rate_IP3R = 0.006, rate_L = 0.00011;
    real rate_SERCA = 0.0009, ratio_ER_cyt = 0.185, Ca_init = 0.073, h_init = 0.793, IP3_init = 0.010925583871830699;
    bool randomize_astro = true;
    real ra_lower = 0.9, ra_upper = 1.05, ra_var = 0.1;

    // Neuron (AdEx), excitatory then inhibitory.
    real e_Delta_T = 2.0, e_a = 4.0, e_tau_w = 450.0, e_V_reset = -50.0, e_b = 300.0, e_V_m = -50.0;
    real e_C_m = 130.0, e_g_L = 18.0, e_E_L = -58.0, e_V_th = -50.0, e_V_peak = 0.0, e_t_ref = 0.0;
    real e_E_ex = 0.0, e_E_in = -85.0, e_tau_syn_ex = 0.2, e_tau_syn_in = 2.0;

    real i_Delta_T = 2.0, i_a = -0.8, i_tau_w = 264.0, i_V_reset = -60.0, i_b = 130.0, i_V_m = -60.0;
    real i_C_m = 104.0, i_g_L = 4.3, i_E_L = -65.0, i_V_th = -52.0, i_V_peak = 0.0, i_t_ref = 0.0;
    real i_E_ex = 0.0, i_E_in = -85.0, i_tau_syn_ex = 0.2, i_tau_syn_in = 2.0;

    bool randomize_neuron = true;
    real rn_lower = 0.9, rn_upper = 1.1, rn_var = 0.05;

    // Synapses.
    real w_e = 5.0, w_i = -5.0, w_n2a = 0.2, w_a2n = 1.0;
    real d_e = 1.0, d_i = 1.0, d_a2n = 1.0;
    bool stp_enabled = true;
    real stp_U = 0.5, stp_tau_rec = 800.0, stp_tau_fac = 0.0;
    int sic_interval = 1;

    // Background drive (note: astrosimgpu's own config loader reads the
    // JSON key "gauss_noise_var" straight into the std-deviation field with
    // no square root -- see astrosimgpu/src/parameters.cpp -- so these
    // numbers are used as a std, not a variance, to match that behaviour
    // exactly).
    real astro_poiss_rate = 3.029715502945856, astro_poiss_weight = 1.0, astro_noise_std = 0.0003;
    real exc_poiss_rate = 2700.0, exc_poiss_weight = 1.0, exc_noise_std = 100.0;
    real inh_poiss_rate = 2500.0, inh_poiss_weight = 1.0, inh_noise_std = 100.0;
    bool independent_noise = true;
    real noise_dt = 0.0;  // 0 -> resolved to 10*dt, matching parameters.cpp
};

int delay_to_steps(real delay_ms, real dt) {
    const int steps = static_cast<int>(delay_ms / dt + 0.5);
    return steps < 1 ? 1 : steps;
}

// =========================================================================
// main
// =========================================================================
int main(int argc, char** argv) {
    index_t n_total = 1000;
    std::uint64_t seed = 1;
    real dt = 0.1;
    int substeps = 1;
    real pre_ms = 1000.0, sim_ms = 5000.0;
    int block = 128;
    int print_every_steps = 0;  // 0 -> auto (~20 prints)
    // NaN sentinel = "not overridden, keep the config default" -- 0 is a
    // legitimate value (fully disconnects that direction) so it can't double
    // as the sentinel.
    real w_n2a_override = std::nan("");
    real w_a2n_override = std::nan("");
    bool stp_override_disable = false;
    std::string dump_edges_path;

    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char* name) { return std::strcmp(argv[i], name) == 0; };
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
        if (arg("--n") || arg("-n")) n_total = static_cast<index_t>(std::atoll(next()));
        else if (arg("--seed")) seed = static_cast<std::uint64_t>(std::atoll(next()));
        else if (arg("--dt")) dt = std::atof(next());
        else if (arg("--substeps")) substeps = std::atoi(next());
        else if (arg("--pre-ms")) pre_ms = std::atof(next());
        else if (arg("--sim-ms")) sim_ms = std::atof(next());
        else if (arg("--block")) block = std::atoi(next());
        else if (arg("--w-n2a")) w_n2a_override = std::atof(next());
        else if (arg("--w-a2n")) w_a2n_override = std::atof(next());
        else if (arg("--no-stp")) stp_override_disable = true;
        else if (arg("--dump-edges")) dump_edges_path = next();
        else if (arg("--help") || arg("-h")) {
            std::printf(
                "astrosim_fused --n N_total [--seed S] [--dt 0.1] [--substeps 1] "
                "[--pre-ms 1000] [--sim-ms 5000] [--block 128]\n"
                "         [--w-n2a W] [--w-a2n W]\n"
                "N_total is split into neurons and astrocytes by the same fixed-\n"
                "in-degree scaling law astrosimgpu/scripts/roihu/gen_neuron_scaling_\n"
                "configs.py uses: N_E=0.8*N, N_I=0.2*N, N_A=N/5, p_primary=80/N_E.\n"
                "--w-n2a/--w-a2n override the neuron->astrocyte / astrocyte->neuron\n"
                "coupling weight (default 0.2 / 1.0) -- pass 0 to fully disconnect\n"
                "one direction of the tripartite loop for A/B comparison.\n"
                "--no-stp disables Tsodyks-Markram short-term plasticity (every\n"
                "delivered spike then uses the raw weight, multiplier 1.0).\n"
                "--dump-edges PATH writes the built connectivity as CSV in the same\n"
                "format astrosimgpu's --dump-edges uses (type,source,target; type in\n"
                "exc_primary/inh_primary/n2a/a2n, astrocyte ids prefixed 'A') and\n"
                "exits before touching the GPU -- runs fine on a login node.\n"
                "Run only through srun/sbatch -- there is no GPU on a login node.\n");
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s (try --help)\n", argv[i]);
            return 1;
        }
    }

    ScalingConfig cfg = ScalingConfig::make(n_total);
    Biology bio;
    if (bio.noise_dt <= 0.0) bio.noise_dt = 10.0 * dt;
    if (!std::isnan(w_n2a_override)) bio.w_n2a = w_n2a_override;
    if (!std::isnan(w_a2n_override)) bio.w_a2n = w_a2n_override;
    if (stp_override_disable) bio.stp_enabled = false;

    std::printf("=== astrosim_fused: race-free, kernel-fused CUDA astrosimgpu ===\n");
    std::printf("N_total=%u  N_E=%u  N_I=%u  N_A=%u  p_primary=%.6g  (fixed in-degree law, K_SYN=%.0f)\n",
               n_total, cfg.n_exc, cfg.n_inh, cfg.n_astro, cfg.p_primary, ScalingConfig::K_SYN);
    std::printf("w_n2a=%g  w_a2n=%g%s\n", bio.w_n2a, bio.w_a2n,
               (!std::isnan(w_n2a_override) || !std::isnan(w_a2n_override)) ? "  (overridden)" : "");
    std::printf("dt=%g ms  substeps=%d  pre=%g ms  sim=%g ms  seed=%llu  precision=%s\n", dt,
               substeps, pre_ms, sim_ms, static_cast<unsigned long long>(seed),
               sizeof(real) == 8 ? "double" : "float");

    // Connectivity generation is pure host code, so --dump-edges can run (and
    // exit) here, before ever touching the GPU -- useful for diffing
    // structure against astrosimgpu's own --dump-edges without needing an
    // srun allocation at all.
    auto t_conn0 = std::chrono::steady_clock::now();
    Connectivity conn = build_connectivity(cfg, seed);
    auto t_conn1 = std::chrono::steady_clock::now();
    std::printf("connectivity: exc=%u inh=%u e2a=%u sic=%u edges, built in %.3f s\n",
               conn.n_exc_edges, conn.n_inh_edges, conn.n_e2a_edges, conn.n_sic_edges,
               std::chrono::duration<double>(t_conn1 - t_conn0).count());

    if (!dump_edges_path.empty()) {
        std::ofstream out(dump_edges_path);
        if (!out) {
            std::fprintf(stderr, "cannot open %s for --dump-edges\n", dump_edges_path.c_str());
            return 1;
        }
        out << "type,source,target\n";
        // inc_* is CSR-by-TARGET (row index = target, src[] = sources), the
        // opposite orientation of astrosimgpu's CSR-by-source -- so a
        // (source,target) pair here is (inc.src[k], row). Offsets/prefixes
        // match astrosimgpu/src/main.cpp's --dump-edges exactly: inh_primary
        // sources are local inhibitory indices (+n_exc for the global id,
        // same as astrosimgpu's dump_set source_offset), astrocyte ids are
        // prefixed "A" on whichever side they appear.
        auto dump_inverted = [&](const char* type, const HostCSR& csr, long long src_offset,
                                 const char* src_prefix, const char* tgt_prefix) {
            for (std::size_t t = 0; t + 1 < csr.row_start.size(); ++t) {
                for (index_t k = csr.row_start[t]; k < csr.row_start[t + 1]; ++k) {
                    out << type << ',' << src_prefix << (static_cast<long long>(csr.src[k]) + src_offset)
                        << ',' << tgt_prefix << t << '\n';
                }
            }
        };
        dump_inverted("exc_primary", conn.inc_exc, 0, "", "");
        dump_inverted("inh_primary", conn.inc_inh, static_cast<long long>(cfg.n_exc), "", "");
        dump_inverted("n2a", conn.inc_e2a, 0, "", "A");
        dump_inverted("a2n", conn.inc_sic, 0, "A", "");
        std::printf("Wrote edge list to %s\n", dump_edges_path.c_str());
        return 0;
    }

    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceCount(&dev));
    if (dev == 0) {
        std::fprintf(stderr,
                     "No CUDA device visible. This must run inside an srun/sbatch GPU "
                     "allocation (e.g. --gres=gpu:gh200:1), never on a bare login node.\n");
        return 1;
    }
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

    // ---- populations (host draws, then upload) ----
    HostRng rng(seed, 0x5EEDULL);
    std::vector<real> Ca_tot(cfg.n_astro), IP3_0(cfg.n_astro), tau_IP3(cfg.n_astro), delta_IP3(cfg.n_astro);
    for (index_t i = 0; i < cfg.n_astro; ++i) {
        Ca_tot[i] = draw_scaled(bio.Ca_tot, bio.randomize_astro, bio.ra_lower, bio.ra_upper, bio.ra_var, rng);
        IP3_0[i] = draw_scaled(bio.IP3_0, bio.randomize_astro, bio.ra_lower, bio.ra_upper, bio.ra_var, rng);
        tau_IP3[i] = draw_scaled(bio.tau_IP3, bio.randomize_astro, bio.ra_lower, bio.ra_upper, bio.ra_var, rng);
        delta_IP3[i] = draw_scaled(bio.delta_IP3, bio.randomize_astro, bio.ra_lower, bio.ra_upper, bio.ra_var, rng);
    }
    const index_t n_neurons = cfg.n_exc + cfg.n_inh;
    std::vector<real> V_reset(n_neurons), b_arr(n_neurons);
    for (index_t i = 0; i < n_neurons; ++i) {
        const bool exc = i < cfg.n_exc;
        const real Vr = exc ? bio.e_V_reset : bio.i_V_reset;
        const real bb = exc ? bio.e_b : bio.i_b;
        V_reset[i] = draw_scaled(Vr, bio.randomize_neuron, bio.rn_lower, bio.rn_upper, bio.rn_var, rng);
        b_arr[i] = draw_scaled(bb, bio.randomize_neuron, bio.rn_lower, bio.rn_upper, bio.rn_var, rng);
    }

    AstroConstants ac{};
    ac.Kd_IP3_1 = bio.Kd_IP3_1; ac.Kd_IP3_2 = bio.Kd_IP3_2; ac.Kd_act = bio.Kd_act; ac.Kd_inh = bio.Kd_inh;
    ac.Km_SERCA = bio.Km_SERCA; ac.k_IP3R = bio.k_IP3R; ac.rate_IP3R = bio.rate_IP3R; ac.rate_L = bio.rate_L;
    ac.rate_SERCA = bio.rate_SERCA; ac.ratio_ER_cyt = bio.ratio_ER_cyt; ac.SIC_scale = bio.SIC_scale; ac.SIC_th = bio.SIC_th;

    NeuronConstants ne{}, ni{};
    ne.C_m = bio.e_C_m; ne.g_L = bio.e_g_L; ne.E_L = bio.e_E_L; ne.V_th = bio.e_V_th; ne.Delta_T = bio.e_Delta_T;
    ne.a = bio.e_a; ne.tau_w = bio.e_tau_w; ne.V_peak = bio.e_V_peak; ne.t_ref = bio.e_t_ref; ne.E_ex = bio.e_E_ex;
    ne.E_in = bio.e_E_in; ne.tau_syn_ex = bio.e_tau_syn_ex; ne.tau_syn_in = bio.e_tau_syn_in; ne.I_e = 0.0;
    ne.psc_init_ex = 2.7182818284590452353602874713527 / ne.tau_syn_ex;
    ne.psc_init_in = 2.7182818284590452353602874713527 / ne.tau_syn_in;

    ni.C_m = bio.i_C_m; ni.g_L = bio.i_g_L; ni.E_L = bio.i_E_L; ni.V_th = bio.i_V_th; ni.Delta_T = bio.i_Delta_T;
    ni.a = bio.i_a; ni.tau_w = bio.i_tau_w; ni.V_peak = bio.i_V_peak; ni.t_ref = bio.i_t_ref; ni.E_ex = bio.i_E_ex;
    ni.E_in = bio.i_E_in; ni.tau_syn_ex = bio.i_tau_syn_ex; ni.tau_syn_in = bio.i_tau_syn_in; ni.I_e = 0.0;
    ni.psc_init_ex = 2.7182818284590452353602874713527 / ni.tau_syn_ex;
    ni.psc_init_in = 2.7182818284590452353602874713527 / ni.tau_syn_in;

    // ---- delays / ring depth ----
    const int d_e_steps = delay_to_steps(bio.d_e, dt);
    const int d_i_steps = delay_to_steps(bio.d_i, dt);
    const int d_a2n_steps = delay_to_steps(bio.d_a2n, dt);
    const int ring_slots = std::max({d_e_steps, d_i_steps, d_a2n_steps}) + 1;

    // ---- device allocation ----
    DeviceState s{};
    s.n_astro = cfg.n_astro;
    s.n_neurons = n_neurons; s.n_exc = cfg.n_exc; s.n_inh = cfg.n_inh;

    auto alloc_copy_real = [&](std::vector<real>& host) {
        real* d = nullptr;
        CUDA_CHECK(cudaMalloc(&d, host.size() * sizeof(real)));
        CUDA_CHECK(cudaMemcpy(d, host.data(), host.size() * sizeof(real), cudaMemcpyHostToDevice));
        return d;
    };
    auto alloc_fill_real = [&](index_t n, real value) {
        std::vector<real> h(n, value);
        return alloc_copy_real(h);
    };
    auto alloc_zero_real = [&](std::size_t n) {
        real* d = nullptr;
        CUDA_CHECK(cudaMalloc(&d, n * sizeof(real)));
        CUDA_CHECK(cudaMemset(d, 0, n * sizeof(real)));
        return d;
    };

    s.Ca = alloc_fill_real(cfg.n_astro, bio.Ca_init);
    s.IP3 = alloc_fill_real(cfg.n_astro, bio.IP3_init);
    s.h = alloc_fill_real(cfg.n_astro, bio.h_init);
    s.Ca_tot = alloc_copy_real(Ca_tot);
    s.IP3_0 = alloc_copy_real(IP3_0);
    s.tau_IP3 = alloc_copy_real(tau_IP3);
    s.delta_IP3 = alloc_copy_real(delta_IP3);

    {
        std::vector<real> V0(n_neurons);
        for (index_t i = 0; i < n_neurons; ++i) V0[i] = (i < cfg.n_exc) ? bio.e_V_m : bio.i_V_m;
        s.V = alloc_copy_real(V0);
    }
    s.w = alloc_zero_real(n_neurons);
    s.g_ex = alloc_zero_real(n_neurons); s.dg_ex = alloc_zero_real(n_neurons);
    s.g_in = alloc_zero_real(n_neurons); s.dg_in = alloc_zero_real(n_neurons);
    s.I_sic_hold = alloc_zero_real(n_neurons);
    s.V_reset = alloc_copy_real(V_reset);
    s.b = alloc_copy_real(b_arr);
    CUDA_CHECK(cudaMalloc(&s.refractory_steps, n_neurons * sizeof(int)));
    CUDA_CHECK(cudaMemset(s.refractory_steps, 0, n_neurons * sizeof(int)));

    s.stp_x = alloc_fill_real(n_neurons, 1.0);
    s.stp_u = alloc_fill_real(n_neurons, bio.stp_U);
    s.stp_t_last = alloc_fill_real(n_neurons, -1e12);

    s.ring_neuron = alloc_zero_real(static_cast<std::size_t>(ring_slots) * n_neurons);
    s.ring_sic = alloc_zero_real(static_cast<std::size_t>(ring_slots) * cfg.n_astro);

    s.inc_exc = upload_csr(conn.inc_exc);
    s.inc_inh = upload_csr(conn.inc_inh);
    s.inc_sic = upload_csr(conn.inc_sic);
    s.inc_e2a = upload_csr(conn.inc_e2a);

    CUDA_CHECK(cudaMalloc(&s.spike_counter, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(s.spike_counter, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&s.ever_active, cfg.n_astro * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemset(s.ever_active, 0, cfg.n_astro * sizeof(unsigned char)));

    // ---- time loop ----
    const std::int64_t pre_steps = static_cast<std::int64_t>(pre_ms / dt + 0.5);
    const std::int64_t sim_steps = static_cast<std::int64_t>(sim_ms / dt + 0.5);
    const std::int64_t total_steps = pre_steps + sim_steps;
    const real h_step = dt / static_cast<real>(substeps);

    const int grid_astro = (cfg.n_astro + block - 1) / block;
    const int grid_neuron = (n_neurons + block - 1) / block;

    if (print_every_steps <= 0) print_every_steps = static_cast<int>(std::max<std::int64_t>(1, total_steps / 20));

    unsigned long long spikes_before_measured = 0;
    unsigned long long spikes_at_last_checkpoint = 0;
    std::int64_t steps_at_last_checkpoint = pre_steps;
    // ~10 checkpoints across the recorded window, so a runaway/settling
    // trend (firing rate climbing over time, e.g. towards a synchronized
    // regime) is visible instead of hidden inside one end-of-run average.
    const std::int64_t checkpoint_every =
        std::max<std::int64_t>(1, sim_steps / 10);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t_run0 = std::chrono::steady_clock::now();

    for (std::int64_t step = 0; step < total_steps; ++step) {
        const int steps_per_noise = std::max(1, static_cast<int>(std::llround(bio.noise_dt / dt)));
        const std::uint64_t noise_idx = static_cast<std::uint64_t>(step / steps_per_noise);
        const bool sic_refresh_now =
            (step - d_a2n_steps >= 0) && ((step - d_a2n_steps) % bio.sic_interval == 0);

        real shared_noise_astro = 0.0;
        if (bio.astro_noise_std > 0.0 && !bio.independent_noise) {
            shared_noise_astro = bio.astro_noise_std * rng_normal(seed ^ 0x9E3779B97F4A7C15ULL, noise_idx);
        }
        real shared_noise_exc = 0.0, shared_noise_inh = 0.0;  // independent_noise=true for exc/inh in this config

        astro_step_kernel<<<grid_astro, block>>>(s, ac, h_step, substeps, dt, step, ring_slots,
                                                 d_e_steps, bio.w_n2a, bio.astro_poiss_rate,
                                                 bio.astro_poiss_weight, bio.astro_noise_std,
                                                 bio.independent_noise, shared_noise_astro, seed);

        neuron_step_kernel<<<grid_neuron, block>>>(
            s, ne, ni, h_step, substeps, dt, step, ring_slots, d_e_steps, d_i_steps, d_a2n_steps,
            bio.w_e, bio.w_i, bio.w_a2n, sic_refresh_now, bio.exc_poiss_rate, bio.exc_poiss_weight,
            bio.inh_poiss_rate, bio.inh_poiss_weight, bio.exc_noise_std, bio.independent_noise,
            shared_noise_exc, bio.inh_noise_std, bio.independent_noise, shared_noise_inh, noise_idx,
            noise_idx, bio.stp_U, bio.stp_tau_rec, bio.stp_tau_fac, bio.stp_enabled, seed);

        if (step == pre_steps - 1 || (pre_steps == 0 && step == 0)) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&spikes_before_measured, s.spike_counter, sizeof(unsigned long long),
                                  cudaMemcpyDeviceToHost));
            spikes_at_last_checkpoint = spikes_before_measured;
            // ever_active should reflect the recorded window only, matching
            // the CPU reference's calcium analysis (which only sees
            // recorded-window astrocyte samples) -- discard any threshold
            // crossings seen during the discarded transient.
            CUDA_CHECK(cudaMemset(s.ever_active, 0, cfg.n_astro * sizeof(unsigned char)));
        }

        // Firing rate over each ~10% slice of the recorded window: cheap
        // (about 10 syncs total, not per step) and shows whether any excess
        // rate is present from the start or builds up over time.
        if (step >= pre_steps && (step - pre_steps) % checkpoint_every == 0 && step > steps_at_last_checkpoint) {
            unsigned long long spikes_now = 0;
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&spikes_now, s.spike_counter, sizeof(unsigned long long),
                                  cudaMemcpyDeviceToHost));
            const std::int64_t window_steps = step - steps_at_last_checkpoint;
            const double window_s = window_steps * dt * 1e-3;
            const double window_hz = static_cast<double>(spikes_now - spikes_at_last_checkpoint) /
                                     (static_cast<double>(n_neurons) * window_s);
            std::printf("  step %8lld / %8lld (%5.1f%%)  windowed rate: %.4f Hz\n",
                       static_cast<long long>(step), static_cast<long long>(total_steps),
                       100.0 * step / total_steps, window_hz);
            std::fflush(stdout);
            spikes_at_last_checkpoint = spikes_now;
            steps_at_last_checkpoint = step;
        } else if (step % print_every_steps == 0) {
            CUDA_CHECK(cudaGetLastError());
            std::printf("  step %8lld / %8lld (%5.1f%%)\r", static_cast<long long>(step),
                       static_cast<long long>(total_steps), 100.0 * step / total_steps);
            std::fflush(stdout);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t_run1 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaGetLastError());

    unsigned long long spikes_total = 0;
    CUDA_CHECK(cudaMemcpy(&spikes_total, s.spike_counter, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    const unsigned long long spikes_measured = spikes_total - spikes_before_measured;

    std::vector<real> Ca_final(cfg.n_astro);
    CUDA_CHECK(cudaMemcpy(Ca_final.data(), s.Ca, cfg.n_astro * sizeof(real), cudaMemcpyDeviceToHost));
    real mean_ca = 0.0;
    index_t active_astro = 0;
    for (real ca : Ca_final) {
        mean_ca += ca;
        if ((ca - bio.SIC_th) * 1000.0 > 1.0) ++active_astro;
    }
    mean_ca /= std::max<index_t>(1, cfg.n_astro);

    std::vector<unsigned char> ever_active_host(cfg.n_astro);
    CUDA_CHECK(cudaMemcpy(ever_active_host.data(), s.ever_active, cfg.n_astro * sizeof(unsigned char),
                          cudaMemcpyDeviceToHost));
    index_t ever_active_count = 0;
    for (unsigned char v : ever_active_host) ever_active_count += v;

    const double wall_s = std::chrono::duration<double>(t_run1 - t_run0).count();
    const double mean_rate_hz =
        static_cast<double>(spikes_measured) / (static_cast<double>(n_neurons) * (sim_ms * 1e-3));

    std::printf("\n=== done ===\n");
    std::printf("wall time            : %.3f s for %lld steps (%.2f us/step, both kernels)\n", wall_s,
               static_cast<long long>(total_steps), wall_s * 1e6 / static_cast<double>(total_steps));
    std::printf("spikes (recorded win): %llu\n", spikes_measured);
    std::printf("mean firing rate     : %.4f Hz\n", mean_rate_hz);
    std::printf("mean calcium (final) : %.6g uM  (SIC_th=%.6g uM)\n", mean_ca, bio.SIC_th);
    std::printf("astrocytes active    : %u / %u (Ca above SIC threshold at the final step only)\n",
               active_astro, cfg.n_astro);
    std::printf("astrocytes engaged   : %u / %u (Ca crossed SIC threshold at ANY point in the "
               "recorded window -- comparable to astrosimgpu's 'astrocytes with transients')\n",
               ever_active_count, cfg.n_astro);

    return 0;
}
