# Pre-upsampling for Optimization Stability (v2.0.1)

## 1. Background and Motivation
Seismic velocity monitoring using the stretching method relies on interpolating cross-correlation functions (CCFs). While the v2.0 implementation using linear interpolation (`jnp.interp`) was fast, it suffered from "jagged" likelihood surfaces because the interpolation nodes caused discontinuous gradients. This often led to excessive iterations in the L-BFGS-B optimizer or failure in line searches.

## 2. The Solution: 4x Pre-upsampling
Instead of changing the interpolation algorithm itself (which would increase computational complexity), we introduced a pre-processing step to upsample the reference waveforms.

### Staged Approach
1.  **FFT-based Upsampling**: The reference CCF and its derivative are upsampled 4x (from 1024 to 4096 points) using `scipy.signal.resample` before the EKF loop starts.
2.  **Separate Outlier Detection**: To avoid shape mismatch errors (1024 vs 4096), the EKF engine now accepts an additional argument `ccf_r_orig` (original resolution). Outlier rejection uses this original array, while the core stretching/stretching sensitivity calculation uses the upsampled 4096-point arrays.
3.  **Linear Interpolation Consistency**: The EKF engine continues to use JAX's fast `jnp.interp`, but the finer grid effectively smoothes the likelihood surface.

## 3. Results and Performance (v2.0.1 vs v2.0.0)
Benchmark tests on 2983 days of data showed significant improvements:

| Metric | v2.0.0 (Baseline) | v2.0.1 (4x Upsampled) | Improvement |
| :--- | :--- | :--- | :--- |
| **Total Optimization Time** | 330.4s | **238.9s** | **~28% faster** |
| **Step 2 (Physical Fit) Time** | 147.4s | **80.9s** | **~45% faster** |
| **Convergence Stability** | Standard | **High (Smooth gradients)** | - |

### Summary
By increasing the grid density by a factor of 4, we achieved a smoother likelihood surface that allows the L-BFGS-B optimizer to converge in fewer steps. The staged implementation ensures numerical stability without increasing the per-iteration computational overhead significantly.

## 4. Usage
The upsampling is handled automatically in `plot_tradeoff_jax.py`. The EKF engine in `KalmanFilter_jax.py` has been slightly extended to accept the upsampled derivative and the original reference separately.
