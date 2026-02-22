
import jax.numpy as jnp
import numpy as np

def standard_likelihood(Zt, Pt, h0, vt):
    """
    Standard Kalman Filter Likelihood Calculation (Naive Implementation)
    
    Likelihood L(t) ~ N(vt, St)
    St = Zt @ Pt @ Zt.T + h0 * I
    lnL = -0.5 * (ln|St| + vt.T @ St^-1 @ vt)
    """
    N = vt.shape[0] # Dimension of observation (Lag * 2)
    # 1. Innovation Covariance St
    St = Zt @ Pt @ Zt.T + h0 * jnp.eye(N)
    
    # 2. Determinant term: ln|St|
    sign, logdet = jnp.linalg.slogdet(St)
    ln_det_St = logdet 
    
    # 3. Residual term: vt.T @ St^-1 @ vt
    St_inv = jnp.linalg.inv(St)
    mahalanobis = vt.T @ St_inv @ vt
    
    # 4. Total Log Likelihood
    lnL = -0.5 * (ln_det_St + mahalanobis)
    
    return lnL, ln_det_St, mahalanobis

def current_jax_likelihood(Zt, Pt, h0, vt):
    """
    Current JAX Implementation (Optimized with Woodbury Identity?)
    """
    N = vt.shape[0] # Dimension of observation
    # Z2 = Zt.T @ Zt (2x2 matrix)
    Z2 = Zt.T @ Zt 
    
    # v2 = vt.T @ vt (scalar)
    v2 = jnp.sum(vt**2)
    
    # gamma = Zt.T @ vt (2x1 vector)
    gamma = Zt.T @ vt
    
    inv_Pt = jnp.linalg.inv(Pt)
    mid_inv = jnp.linalg.inv(Z2/h0 + inv_Pt)
    
    M = Z2 @ Pt
    tr = M[0,0] + M[1,1]
    det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    disc = jnp.maximum(tr**2 - 4*det, 0.0)
    lam1 = (tr + jnp.sqrt(disc)) / 2.0
    lam2 = (tr - jnp.sqrt(disc)) / 2.0
    
    # lnL1 (Determinant Term Approximation?)
    # Is (N-2)*log(h0) correct? 
    # St has eigenvalues: lam1+h0, lam2+h0, and (N-2) eigenvalues of h0
    lnL1 = jnp.log(lam1 + h0) + jnp.log(lam2 + h0) + (N - 2) * jnp.log(h0)
    
    # lnL2 (Residual Term Approximation?)
    lnL2 = h0 * v2 - gamma.T @ mid_inv @ gamma
    
    # Total Log Likelihood
    lnL_step = -(lnL1 + lnL2 * (h0**-2)) / 2.0
    
    return lnL_step, lnL1, lnL2 * (h0**-2)

def run_test():
    # Setup standard test parameters
    N = 50 # Dimension of observation (e.g. 25 lags * 2 components)
    M = 2  # Dimension of state (alpha, beta)
    
    np.random.seed(42)
    Zt = np.random.randn(N, M)
    Pt = np.eye(M) * 1e-5
    h0 = 0.1 # Observation noise variance
    vt = np.random.randn(N) * np.sqrt(h0) # Residuals consistent with h0
    
    # Convert to JAX arrays
    Zt_j = jnp.array(Zt)
    Pt_j = jnp.array(Pt)
    vt_j = jnp.array(vt)
    
    print(f"Test Parameters:")
    print(f"  Obs Dim (N): {N}")
    print(f"  State Dim (M): {M}")
    print(f"  h0: {h0}")
    print("-" * 30)
    
    # 1. Run Standard Calculation
    lnL_std, det_std, mah_std = standard_likelihood(Zt_j, Pt_j, h0, vt_j)
    print(f"Standard Implementation:")
    print(f"  ln|St| term: {det_std:.6f}")
    print(f"  Mahalanobis term: {mah_std:.6f}")
    print(f"  Total lnL: {lnL_std:.6f}")
    print("-" * 30)
    
    # 2. Run Current JAX Calculation
    lnL_cur, det_cur, mah_cur = current_jax_likelihood(Zt_j, Pt_j, h0, vt_j)
    print(f"Current JAX Implementation:")
    print(f"  lnL1 term: {det_cur:.6f}")
    print(f"  lnL2 * h0^-2 term: {mah_cur:.6f}")
    print(f"  Total lnL: {lnL_cur:.6f}")
    print("-" * 30)
    
    # 3. Compare
    diff_val = lnL_cur - lnL_std
    print(f"Difference (Current - Standard): {diff_val:.6e}")
    if abs(diff_val) < 1e-4:
        print(">> MATCH: The implementations are mathematically equivalent.")
    else:
        print(">> MISMATCH: The implementations differ significantly.")

if __name__ == "__main__":
    run_test()
