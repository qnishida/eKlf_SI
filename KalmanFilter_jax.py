import jax
import jax.numpy as jnp
from jax import lax, jit, vmap
from functools import partial
import numpy as np

def deri_jax(data, n, delta):
    if n > 0:
        dω = 1./(data.shape[-1] * delta) * jnp.pi * 2.
        spctrm = jnp.fft.rfft(data)
        freq_idxs = jnp.arange(spctrm.shape[-1])
        spctrm = spctrm * (1j * freq_idxs * dω)**n
        return jnp.fft.irfft(spctrm, n=data.shape[-1])
    return data

def T_jax(t, τ):
    return jnp.exp(-t / τ)

def est_h0_jax(ccfs1, ccf_r_unused, ts, te):
    idx = jnp.arange(1024)
    mask0 = ((idx >= ts + 512) & (idx < te + 512)) | ((idx > 512 - te) & (idx <= 512 - ts))
    ccf_r_init = jnp.nanmean(ccfs1, axis=2)
    h0_sum, count_total = 0.0, 0.0
    ccf_r_final = jnp.zeros((3, 3, 1024))
    for row in range(3):
        for col in range(3):
            ccfs = ccfs1[row, col]
            mccfs = jnp.where(jnp.isnan(ccfs), 0.0, ccfs)
            valid_days = jnp.sum(~jnp.isnan(ccfs), axis=0)
            ref1 = jnp.sum(mccfs, axis=0) / jnp.where(valid_days > 0, valid_days, 1.0)
            dot_num = jnp.sum(jnp.where(mask0, ccfs * ref1, 0.0), axis=-1)
            dot_den = jnp.sum(jnp.where(mask0, ref1**2, 0.0))
            scale = dot_num / dot_den
            mask = (scale > 0.5) & (scale < 5.0)
            diff_sq = jnp.where(mask0, (ccfs - ref1)**2, 0.0)
            h0_sum += jnp.sum(jnp.where(mask[:, None], diff_sq, 0.0))
            count_total += jnp.sum(mask) * jnp.sum(mask0)
            scaled_ccfs = jnp.where(mask[:, None], ccfs / scale[:, None], jnp.nan)
            ccf_r_final = ccf_r_final.at[row, col].set(jnp.nanmean(scaled_ccfs, axis=0))
    return h0_sum / jnp.where(count_total > 0, count_total, 1.0), ccf_r_final

@partial(jit, static_argnums=())
def batch_interp(t_points, xp, fp_3x3):
    return vmap(vmap(lambda f: jnp.interp(t_points, xp, f)))(fp_3x3)

@jit
def eKlf_jax(ccfs_all, ccf_r, delta, ts, te, β, h0, Q0, Pt_init, at_init):
    idx = jnp.arange(1024)
    mask0 = ((idx >= ts + 512) & (idx < te + 512)) | ((idx > 512 - te) & (idx <= 512 - ts))
    lag_time_full = (jnp.arange(1024) - 512) * delta
    num_lags_valid = jnp.sum(mask0)
    ccf_deri = deri_jax(ccf_r, 1, delta)
    ms_r = jnp.sum(jnp.where(mask0, ccf_r**2, 0.0), axis=-1)
    Qt = jnp.diag(jnp.array([Q0[0], Q0[1]]))
    yt_all = jnp.transpose(ccfs_all, (2, 0, 1, 3))

    def scan_fn(carry, inputs):
        at, Pt, lnL_acc = carry
        yt, β_val = inputs
        A, α = at[0], at[1]
        t_shifted = lag_time_full * (1.0 + α + β_val)
        
        # Robust Interpolation-based stretching
        Z0 = batch_interp(t_shifted, lag_time_full, ccf_r)
        Z1_base = batch_interp(t_shifted, lag_time_full, ccf_deri)
        # Coordinate-corrected sensitivity (Case B)
        Z1 = A * Z1_base * lag_time_full
        
        # Outlier Rejection
        dot_product = jnp.sum(jnp.where(mask0, yt * ccf_r, 0.0), axis=-1)
        scale = dot_product / jnp.where(ms_r > 0.0, ms_r, 1.0)
        outlier_mask = (scale > 0.5) & (scale < 2.0)
        comp_count = jnp.sum(outlier_mask)
        day_effective_mask = mask0[None, None, :] & outlier_mask[:, :, None]

        Zt = jnp.stack([Z0, Z1], axis=-1)
        Z2 = jnp.einsum('ijkl,ijkm,ijk->lm', Zt, Zt, day_effective_mask)
        vt_comps = yt - A * Z0
        γ = jnp.einsum('ijkl,ijk,ijk->l', Zt, vt_comps, day_effective_mask)
        v2 = jnp.sum(jnp.where(day_effective_mask, vt_comps**2, 0.0))
        
        has_data = comp_count > 0
        is_missing = jnp.all(jnp.isnan(yt))
        
        Z2_reg = jnp.where(has_data, Z2, 0.0)
        X = Z2_reg / h0
        inner_mat = jnp.identity(2) + X @ Pt
        inner_inv = jnp.linalg.inv(inner_mat)
        mid_inv_raw = Pt @ inner_inv
        mid_inv = (mid_inv_raw + mid_inv_raw.T) / 2.0
        
        M = Z2_reg @ Pt
        tr, det = M[0,0] + M[1,1], M[0,0]*M[1,1] - M[0,1]*M[1,0]
        disc = jnp.maximum(tr**2 - 4*det, 1e-12)
        lam1 = (tr + jnp.sqrt(disc)) / 2.0
        lam2 = det / jnp.maximum(lam1, 1e-15)
        
        # Dynamic Degrees of Freedom based on survivor components
        # This prevents 'cheating' by making the likelihood penalize data loss
        n_obs = num_lags_valid * comp_count
        lnL1 = jnp.log(lam1 + h0) + jnp.log(lam2 + h0) + jnp.maximum(n_obs - 2, 0.0) * jnp.log(h0)
        lnL2 = h0 * v2 - γ.T @ mid_inv @ γ
        lnL_const = (n_obs / 2.0) * jnp.log(2 * jnp.pi)
        
        Xi = mid_inv / h0
        
        lnL_valid = -(lnL1 + lnL2 * (h0**-2)) / 2.0 - lnL_const
        lnL_step = jnp.where(is_missing, 0.0, lnL_valid)
        
        att = jnp.where(has_data, at + Xi @ γ, at)
        Ptt = jnp.where(has_data, mid_inv, Pt)
        return (att, Ptt + Qt, lnL_acc + lnL_step), (att, Ptt, has_data)

    init_carry = (at_init, Pt_init, 0.0)
    (at_final, Pt_final, lnL), (att_history, Ptt_history, mask_history) = lax.scan(scan_fn, init_carry, (yt_all, β))
    
    def smoother_scan(carry, inputs):
        α_next, V_next = carry
        att, Ptt, Pt_pred = inputs
        At = Ptt @ jnp.linalg.inv(Pt_pred)
        αt = att + At @ (α_next - att)
        Vt = Ptt + At @ (V_next - Pt_pred) @ At.T
        return (αt, Vt), (αt, Vt)

    Pt_preds = Ptt_history + Qt
    init_smoother = (att_history[-1], Ptt_history[-1])
    _, (αt_smooth, Vt_smooth) = lax.scan(smoother_scan, init_smoother, (att_history[:-1], Ptt_history[:-1], Pt_preds[:-1]), reverse=True)
    
    αt_all = jnp.concatenate([αt_smooth, att_history[-1:]], axis=0)
    Vt_all = jnp.concatenate([Vt_smooth, Ptt_history[-1:]], axis=0)
    return αt_all, Vt_all, lnL, mask_history

@jit
def cal_ref_jax(ccfs_all, at_all, beta_all, delta):
    lag_time_full = (jnp.arange(1024) - 512) * delta
    yt_all = jnp.transpose(ccfs_all, (2, 0, 1, 3))
    def pull_back(yt, alpha_total, A):
        t_pulled = lag_time_full / (1.0 + alpha_total)
        Z_pulled = batch_interp(t_pulled, lag_time_full, yt)
        return Z_pulled
    alpha_total_all = at_all[:, 1] + beta_all
    A_all = at_all[:, 0]
    Z_all = vmap(pull_back)(yt_all, alpha_total_all, A_all)
    new_ref = jnp.nanmean(Z_all, axis=0)
    
    return new_ref
