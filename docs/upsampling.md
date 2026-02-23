# Pre-upsampling for Optimization Stability (v2.0.1)

## 1. Background and Motivation
Seismic velocity monitoring using the stretching method relies on interpolating cross-correlation functions (CCFs). While the v2.0 implementation using linear interpolation (`jnp.interp`) was fast, it suffered from "jagged" likelihood surfaces because the interpolation nodes caused discontinuous gradients. This often led to excessive iterations in the L-BFGS-B optimizer or failure in line searches.

## 2. The Solution: 4x Pre-upsampling
Instead of changing the interpolation algorithm itself (which would increase computational complexity), we introduced a pre-processing step to upsample the reference waveforms.

### Staged Approach
1.  **FFT-based Upsampling**: The reference CCF and its derivative are upsampled 4x (from 1024 to 4096 points) using `scipy.signal.resample` before the EKF loop starts.
2.  **Separate Outlier Detection**: To avoid shape mismatch errors (1024 vs 4096), the EKF engine now accepts an additional argument `ccf_r_orig` (original resolution). Outlier rejection uses this original array, while the core stretching/stretching sensitivity calculation uses the upsampled 4096-point arrays.
3.  **Linear Interpolation Consistency**: The EKF engine continues to use JAX's fast `jnp.interp`, but the finer grid effectively smoothes the likelihood surface.

## 3. Results and Performance (v2.0.2 vs v2.0.0)
Benchmark tests on 2983 days of data showed significant cumulative improvements:

| Metric | v2.0.0 (Baseline) | v2.0.2 (Final Config) | Improvement |
| :--- | :--- | :--- | :--- |
| **Total Optimization Time** | 330.4s | **225.9s** | **~32% faster** |
| **Step 1 (Noise Baseline)** | 183.0s | **95.4s** | **~48% faster** |
| **Peak Memory Usage** | ~15.4GB (Hessian) | **~11.5GB** | **~25% reduced** |

## 4. Technical Refinements in v2.0.2
### Hybrid Diagonal Hessian (AD + FD)
To avoid the massive memory overhead of JAX's 2nd-order automatic differentiation, we implemented a hybrid approach:
- Use 1st-order AD (`jax.grad`) for base gradients.
- Use Finite Differences (FD) to estimate the diagonal Hessian elements ($H_{ii}$) for parameter scaling.
This provides the benefits of adaptive scaling while keeping memory usage at 1st-order levels (~11.5GB).

### Parameter Stabilization
- **Fixed at0_alpha**: Setting the initial velocity state to exactly **0.0** (with high initial uncertainty $P_{t|0}=1E-2$) prevents overfitting to early data artifacts.
- **Fixed Delay_rain**: The precipitation delay is fixed to **0.0**, reducing optimization dimensionality and improving numerical stability.
