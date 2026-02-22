# Technical Improvements in eKlf v2.0

This document outlines the major algorithmic and numerical enhancements introduced in the latest version of eKlf.

## 1. Stretching Methodology: Interpolation-Based Sensitivity
- **Legacy**: Used a 3rd-order Taylor series to approximate waveform stretching. While computationally efficient, it lacked precision for large velocity changes.
- **Improved**: Uses `jnp.interp` to sample the reference waveform and its derivative directly at the stretched coordinates. This ensures mathematical rigor and stability across a much wider range of $dv/v$.

## 2. Robust Likelihood Calculation
- **Dynamic Degrees of Freedom**: The likelihood calculation now dynamically accounts for the number of active components after outlier rejection. This prevents the optimizer from "cheating" by masking poorly fitting data to artificially inflate the likelihood.
- **Sensitivity Accuracy**: Removed redundant scaling factors in the Jacobian ($Z_1$) to align strictly with the partial derivative $\partial y / \partial \alpha$, leading to more stable estimation of the process noise $Q_0$.

## 3. Numerical Stability and Precision
- **Covariance Updates**: Implemented the Woodbury matrix identity with explicit symmetry enforcement to ensure the model covariance matrix ($P_t$) remains positive-definite.
- **Eigenvalue Precision**: Adopted a robust calculation for the secondary eigenvalue ($\lambda_2 = \det / \lambda_1$), preventing numerical underflow that previously caused instability in the likelihood surface when process noise was small.

## 4. Reference CCF Remastering
- **Iterative "Pull-Back"**: Implemented a two-pass approach where the reference CCF is updated by shifting observed data back to the zero-stretch coordinate system using preliminary velocity change estimates.
- **Natural Scaling**: Removed forced peak normalization during the update process. This allows the reference to inherit the actual amplitude scale of the data, naturally stabilizing the amplitude factor $A$ around 1.0.

## 5. Two-Step Optimization Workflow
To resolve the statistical competition between stochastic process noise ($Q_0$) and deterministic models (Rain/EQ), we introduced a staged optimization:
1. **Baseline Phase**: Determine the "raw" variability of the data using only $Q_0$ and $h_0$.
2. **Physical Phase**: Fit deterministic models within the allowed stochastic framework.
This prevents physical parameters from absorbing the background variability, ensuring that $Q_{0, \alpha}$ remains physically meaningful.
