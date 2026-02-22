import numpy as np
from scipy import signal

def deri_numpy(data, n, delta):
    """Calculate the n-th derivative of the data using FFT."""
    if n > 0:
        dw = 1.0 / (data.shape[-1] * delta) * np.pi * 2.0
        spctrm = np.fft.rfft(data)
        freq_idxs = np.arange(spctrm.shape[-1])
        spctrm = spctrm * (1j * freq_idxs * dw)**n
        return np.fft.irfft(spctrm, n=data.shape[-1])
    return data

def T_numpy(t, tau):
    """Exponential response function."""
    return np.exp(-t / tau)

def est_h0_numpy(ccfs1, ccf_r_unused, ts, te):
    """Initial estimation of observation noise h0."""
    idx = np.arange(1024)
    mask0 = ((idx >= ts + 512) & (idx < te + 512)) | ((idx > 512 - te) & (idx <= 512 - ts))
    
    h0_sum, count_total = 0.0, 0.0
    ccf_r_final = np.zeros((3, 3, 1024))
    
    for row in range(3):
        for col in range(3):
            ccfs = ccfs1[row, col]
            # Replace NaNs with 0 for mean calculation
            mccfs = np.where(np.isnan(ccfs), 0.0, ccfs)
            valid_days = np.sum(~np.isnan(ccfs), axis=0)
            ref1 = np.sum(mccfs, axis=0) / np.where(valid_days > 0, valid_days, 1.0)
            
            # Simple scaling to align components
            dot_num = np.sum(np.where(mask0, ccfs * ref1, 0.0), axis=-1)
            dot_den = np.sum(np.where(mask0, ref1**2, 0.0))
            scale = dot_num / np.where(dot_den > 0, dot_den, 1.0)
            
            mask = (scale > 0.5) & (scale < 5.0)
            diff_sq = np.where(mask0, (ccfs - ref1)**2, 0.0)
            
            h0_sum += np.sum(np.where(mask[:, np.newaxis], diff_sq, 0.0))
            count_total += np.sum(mask) * np.sum(mask0)
            
            scaled_ccfs = np.where(mask[:, np.newaxis], ccfs / scale[:, np.newaxis], np.nan)
            ccf_r_final[row, col] = np.nanmean(scaled_ccfs, axis=0)
            
    return h0_sum / np.where(count_total > 0, count_total, 1.0), ccf_r_final

def eKlf_numpy(ccfs_all, ccf_r, delta, ts, te, beta, h0, Q0, Pt_init, at_init):
    """
    Implementation of the Extended Kalman Filter and Smoother using NumPy.
    Logic is mathematically equivalent to the JAX version.
    """
    idx = np.arange(1024)
    mask0 = ((idx >= ts + 512) & (idx < te + 512)) | ((idx > 512 - te) & (idx <= 512 - ts))
    lag_time_full = (np.arange(1024) - 512) * delta
    num_lags_valid = np.sum(mask0)
    
    ccf_deri = deri_numpy(ccf_r, 1, delta)
    ms_r = np.sum(np.where(mask0, ccf_r**2, 0.0), axis=-1)
    Qt = np.diag(np.array([Q0[0], Q0[1]]))
    
    # ccfs_all: (3, 3, days, 1024) -> yt_all: (days, 3, 3, 1024)
    yt_all = np.transpose(ccfs_all, (2, 0, 1, 3))
    num_days = yt_all.shape[0]
    
    # Histories for smoothing
    att_history = np.zeros((num_days, 2))
    Ptt_history = np.zeros((num_days, 2, 2))
    mask_history = np.zeros(num_days, dtype=bool)
    
    at = at_init.copy()
    Pt = Pt_init.copy()
    lnL = 0.0
    
    # Forward Filter
    for t in range(num_days):
        yt = yt_all[t]
        beta_val = beta[t]
        A, alpha = at[0], at[1]
        
        t_shifted = lag_time_full * (1.0 + alpha + beta_val)
        
        # Stretching via Interpolation
        Z0 = np.zeros((3, 3, 1024))
        Z1_base = np.zeros((3, 3, 1024))
        for r in range(3):
            for c in range(3):
                Z0[r, c] = np.interp(t_shifted, lag_time_full, ccf_r[r, c])
                Z1_base[r, c] = np.interp(t_shifted, lag_time_full, ccf_deri[r, c])
        
        Z1 = A * Z1_base * lag_time_full
        
        # Outlier Rejection
        dot_product = np.sum(np.where(mask0, yt * ccf_r, 0.0), axis=-1)
        scale = dot_product / np.where(ms_r > 0.0, ms_r, 1.0)
        outlier_mask = (scale > 0.5 * A) & (scale < 2.0 * A)
        comp_count = np.sum(outlier_mask)
        day_effective_mask = mask0[np.newaxis, np.newaxis, :] & outlier_mask[:, :, np.newaxis]
        
        has_data = comp_count > 0
        is_missing = np.all(np.isnan(yt))
        
        if has_data:
            # Construct sensitivity matrix Zt (N x 2)
            # Flatten only the valid points
            Zt = np.stack([Z0[day_effective_mask], Z1[day_effective_mask]], axis=-1)
            vt = yt[day_effective_mask] - A * Z0[day_effective_mask]
            
            Z2 = Zt.T @ Zt
            gamma = Zt.T @ vt
            v2 = np.sum(vt**2)
            
            # Woodbury-based Update (Stable)
            X = Z2 / h0
            inner_mat = np.identity(2) + X @ Pt
            inner_inv = np.linalg.inv(inner_mat)
            mid_inv_raw = Pt @ inner_inv
            mid_inv = (mid_inv_raw + mid_inv_raw.T) / 2.0
            
            # Likelihood Step
            M = Z2 @ Pt
            tr, det = np.trace(M), np.linalg.det(M)
            disc = max(tr**2 - 4*det, 1e-12)
            lam1 = (tr + np.sqrt(disc)) / 2.0
            lam2 = det / max(lam1, 1e-15)
            
            n_obs = num_lags_valid * comp_count
            lnL1 = np.log(lam1 + h0) + np.log(lam2 + h0) + max(n_obs - 2, 0.0) * np.log(h0)
            lnL2 = h0 * v2 - gamma.T @ mid_inv @ gamma
            lnL_const = (n_obs / 2.0) * np.log(2 * np.pi)
            
            if not is_missing:
                lnL += -(lnL1 + lnL2 * (h0**-2)) / 2.0 - lnL_const
            
            Xi = mid_inv / h0
            at = at + Xi @ gamma
            Pt = mid_inv
            
        att_history[t] = at
        Ptt_history[t] = Pt
        mask_history[t] = has_data
        
        # Predict for next step
        Pt = Pt + Qt
        
    # Backward Smoother
    alphat_all = np.zeros_like(att_history)
    Vt_all = np.zeros_like(Ptt_history)
    
    alphat_all[-1] = att_history[-1]
    Vt_all[-1] = Ptt_history[-1]
    
    for t in reversed(range(num_days - 1)):
        Pt_pred = Ptt_history[t] + Qt
        Jt = Ptt_history[t] @ np.linalg.inv(Pt_pred)
        alphat_all[t] = att_history[t] + Jt @ (alphat_all[t+1] - att_history[t])
        Vt_all[t] = Ptt_history[t] + Jt @ (Vt_all[t+1] - Pt_pred) @ Jt.T
        
    return alphat_all, Vt_all, lnL, mask_history

def cal_ref_numpy(ccfs_all, at_all, beta_all, delta):
    """Update reference CCF using estimated velocity changes (Pull-back)."""
    lag_time_full = (np.arange(1024) - 512) * delta
    yt_all = np.transpose(ccfs_all, (2, 0, 1, 3))
    
    num_days = yt_all.shape[0]
    Z_pulled_sum = np.zeros((3, 3, 1024))
    count = np.zeros((3, 3, 1024))
    
    alpha_total_all = at_all[:, 1] + beta_all
    
    for t in range(num_days):
        yt = yt_all[t]
        alpha_total = alpha_total_all[t]
        t_pulled = lag_time_full / (1.0 + alpha_total)
        
        for r in range(3):
            for c in range(3):
                if not np.all(np.isnan(yt[r, c])):
                    pulled = np.interp(t_pulled, lag_time_full, yt[r, c])
                    Z_pulled_sum[r, c] += pulled
                    count[r, c] += 1
                    
    new_ref = Z_pulled_sum / np.where(count > 0, count, 1.0)
    # Natural scaling: No peak normalization to preserve data amplitude
    return new_ref
