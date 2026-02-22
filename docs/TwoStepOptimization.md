# Two-Step Optimization for JAX-based Kalman Filter

This document describes the two-step optimization strategy implemented to stabilize the estimation of process noise ($Q_0$) and physical response parameters in seismic velocity change ($dv/v$) analysis.

## 1. Motivation: The $Q_0$ Collapse Problem
In a simultaneous 9-parameter optimization, the optimizer tends to drive the velocity process noise ($Q_{0, \alpha}$) toward zero. This happens because the deterministic explanatory models (Precipitation and Earthquake recovery) have high degrees of freedom and can "absorb" the variations that should physically belong to stochastic process noise. Statistically, the filter prefers a deterministic fit to avoid the determinant penalty ($\ln |S_t|$) associated with increased uncertainty.

## 2. The Two-Step Strategy

### Step 1: Noise Baseline Extraction
- **Optimized Parameters**: $Q_{0, amp}$, $Q_{0, \alpha}$, $h_0$ (observation noise), and $at_{0, \alpha}$ (initial velocity state).
- **Fixed Parameters**: All explanatory variables (Rain and EQ amplitudes) are fixed to **exactly 0.0**.
- **Physical Meaning**: By disabling physical models, we force the Kalman Filter to explain all temporal variations using only process noise. This determines the "raw potential" for change in the data, preventing the physical models from underestimating the baseline noise level.

### Step 2: Physical Model Fitting
- **Optimized Parameters**: $	au_{rain}$, $Amp_{rain}$, $	au_{eq}$, $Amp_{eq}$.
- **Fixed Parameters**: $Q_0$ and $h_0$ values obtained in Step 1 are **fixed**.
- **Physical Meaning**: With the filter's "sensitivity" (noise levels) pre-determined, the physical models are fitted to explain the specific patterns (seasonal cycles and seismic recovery) within the allowed stochastic framework. This ensures that the extracted physical constants are robust and not overfitted to noise.

## 3. Algorithmic Enhancements

### Hessian-based Auto-scaling
Before starting Step 2, the algorithm calculates the diagonal of the Hessian matrix at the initial point. 
- **The Problem**: Parameters like $	au$ (time constant) and $Amp$ (amplitude) have vastly different sensitivities, creating "long, narrow valleys" in the likelihood surface.
- **The Solution**: Each parameter's search step is normalized by $1/\sqrt{|H_{ii}|}$ (the inverse square root of the curvature). This "sphericalizes" the likelihood surface, allowing the L-BFGS-B optimizer to converge much faster and more reliably.

### Numerical Stability Fixes
- **Woodbury Symmetry**: Updated the covariance matrix ($P_t$) update to maintain symmetry and positive-definiteness.
- **Eigenvalue Precision**: Switched to calculating the secondary eigenvalue as $\lambda_2 = \det(M) / \lambda_1$ to prevent numerical underflow/overflow.
- **Interpolation Boundary**: Set `jnp.interp` to use zero-padding (`left=0, right=0`) to eliminate jumps caused by data shifting at the edges of the observation period.

## 4. Current Performance (Benchmark)
- **Stability**: $Q_{0, \alpha}$ now recovers to a stable, non-zero value ($\sim 2 	imes 10^{-8}$).
- **Speed**: Optimization completes in approximately 5-6 minutes per pair, significantly faster than the previous simultaneous approach which often timed out.
- **Physical Plausibility**: $	au_{rain}$ and $	au_{eq}$ converge to values (e.g., 37 days and 11 days) that align with hydrological and seismological expectations.
