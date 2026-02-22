import jax
import jax.numpy as jnp
from jax import jit, vmap, config, value_and_grad, lax
from functools import partial
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import signal, optimize
import os
import time
import datetime
from KalmanFilter_jax import eKlf_jax, T_jax, est_h0_jax, cal_ref_jax

# Enable float64
config.update("jax_enable_x64", True)

class KlfTradeoffV17:
    def __init__(self, precipitation_path="./Data/gwl_ebino_seikei.dat"):
        with open(precipitation_path, "r") as f:
            lines = f.readlines()
        self.precip_raw = jnp.array([float(line.split()[4]) for line in lines])
        self.flags = jnp.array([int(line.split()[5]) for line in lines])
        self.num0 = int(jnp.where(self.flags == 1)[0][0])
        self.precipitation = self.precip_raw - jnp.mean(self.precip_raw)
        self.t_kmt = 2177 
        self.q_ref = jnp.array([1E-6, 4E-9, 100., 1E-7, -5.0, 1E-3, 1E-3, 30.0])
        self.base_date = datetime.datetime(2010, 5, 1)

    def get_beta_extended(self, q_params, num_days):
        T0 = T_jax(jnp.arange(len(self.precipitation), dtype=jnp.float64), q_params[2])
        # Correct non-circular shift: positive delay means rain effect arrives later
        x_orig = jnp.arange(len(self.precipitation), dtype=jnp.float64)
        y2 = jnp.interp(x_orig - q_params[4], x_orig, self.precipitation, left=0, right=0)
        full_conv = jax.scipy.signal.convolve(y2, T0, mode='full')
        beta = lax.dynamic_slice(full_conv, (self.num0,), (num_days,))
        x = jnp.arange(num_days, dtype=jnp.float64)
        x_m, y_m = jnp.mean(x), jnp.mean(beta)
        slope = jnp.sum((x - x_m) * (beta - y_m)) / jnp.sum((x - x_m)**2)
        beta_clean = q_params[3] * (beta - y_m - slope * (x - x_m))
        eq_term = jnp.where(x >= self.t_kmt, q_params[6] * jnp.exp(-(x - self.t_kmt) / jnp.maximum(q_params[7], 1.0)), 0.0)
        return beta_clean + eq_term

    def initial_guess_physics(self, ccfs1, ccf_r, delta, ts, te, h0, num_days):
        print("Obtaining preliminary physics-based guess (Robust Cross-Correlation LSQ)...")
        alpha_t, _, _, mask = eKlf_jax(ccfs1, ccf_r, delta, ts, te, jnp.zeros(num_days), h0, [1E-5, 1E-8], jnp.diag(jnp.array([1E-5, 1E-8])), jnp.array([1.0, 0.0]))
        raw_alpha = alpha_t[:, 1]
        u0 = signal.detrend(np.array(raw_alpha))
        u0[np.logical_not(np.array(mask))] = 0
        def rain_cost_with_delay(tau):
            T0 = T_jax(jnp.arange(len(self.precipitation), dtype=jnp.float64), tau)
            conv = jax.scipy.signal.convolve(self.precipitation, T0, mode='full')[self.num0:self.num0+num_days]
            u1 = signal.detrend(np.array(conv))
            u1[np.logical_not(np.array(mask))] = 0
            corr = signal.correlate(u0, u1, mode='full')
            lags = signal.correlation_lags(len(u0), len(u1), mode='full')
            dt_max = 30
            search_mask = (lags >= -dt_max) & (lags <= dt_max)
            search_corr = corr[search_mask]
            search_lags = lags[search_mask]
            best_idx = np.argmin(search_corr)
            best_delay = -search_lags[best_idx]
            min_corr = search_corr[best_idx]
            u1_sq = np.sum(u1**2)
            if u1_sq == 0: return 1e20, 0.0, 0.0
            gamma = min_corr / u1_sq
            cost = -2*min_corr*gamma + gamma*gamma*u1_sq + np.sum(u0**2)
            return cost, gamma, float(best_delay)
        tau_grid = [20., 50., 100., 200., 500.]
        results = [rain_cost_with_delay(t) for t in tau_grid]
        best_idx = np.argmin([r[0] for r in results])
        best_tau = tau_grid[best_idx]
        best_cost, best_gamma, best_delay = results[best_idx]
        best_A_eq, best_tau_eq = 0.0, 30.0
        if num_days > self.t_kmt:
            alpha_eq = u0[self.t_kmt:]
            mask_eq = mask[self.t_kmt:]
            tx_eq = np.arange(len(alpha_eq))
            def eq_cost(tau_eq):
                decay = np.exp(-tx_eq / tau_eq)
                A = jnp.sum(mask_eq * decay * alpha_eq) / jnp.maximum(jnp.sum(mask_eq * decay**2), 1e-10)
                return jnp.sum(mask_eq * (alpha_eq - A * decay)**2), A
            tau_eq_grid = [10., 30., 50., 100., 300.]
            eq_results = [eq_cost(t) for t in tau_eq_grid]
            best_eq_idx = np.argmin([r[0] for r in eq_results])
            best_tau_eq = tau_eq_grid[best_eq_idx]
            _, best_A_eq = eq_results[best_eq_idx]
        new_q_ref = self.q_ref.at[2].set(best_tau).at[3].set(best_gamma).at[4].set(best_delay).at[6].set(best_A_eq).at[7].set(best_tau_eq)
        return new_q_ref

def main():
    freq_band = "0.15to0.90"
    file_path = "./Data/long-STS-2_bp" + freq_band + "white.h5"
    klf_inst = KlfTradeoffV17()
    
    with h5py.File(file_path, 'r') as h5file:
        delta = (int)(h5file.attrs['delta'] * 1000 + 0.5) / 1000
        ts, te = int(20 / delta), int(100 / delta)
        sta_pair = ("04", "03") 
        num_days = min(h5file[sta_pair[0]][sta_pair[1]]["ccfs"].shape[2], len(klf_inst.precipitation) - klf_inst.num0)
        print(f"Using full period: {num_days} days")
        ccfs1 = jnp.array(h5file[sta_pair[0]][sta_pair[1]]["ccfs"][:, :, :num_days, :])
        win2 = signal.windows.tukey(te - ts, 0.1)
        win = np.zeros(1024)
        for i in range(ts, te): win[i + 512], win[512 - i] = win2[i - ts], win2[i - ts]
        ccfs1 = ccfs1 * jnp.array(win)
        
        h0_initial, ccf_r = est_h0_jax(ccfs1, None, ts, te)
        
        print("Preliminary KLF pass for renormalization...")
        q_init_lsq = klf_inst.initial_guess_physics(ccfs1, ccf_r, delta, ts, te, h0_initial, num_days)
        beta_init = klf_inst.get_beta_extended(q_init_lsq, num_days)
        at_pre, _, _, _ = eKlf_jax(ccfs1, ccf_r, delta, ts, te, beta_init, h0_initial, [q_init_lsq[0], q_init_lsq[1]], jnp.diag(jnp.array([q_init_lsq[0], q_init_lsq[1]])), jnp.array([1.0, 0.0]))
        
        print("Updating reference CCF using preliminary results...")
        ccf_r = cal_ref_jax(ccfs1, at_pre, beta_init, delta)
        h0_initial, _ = est_h0_jax(ccfs1, ccf_r, ts, te)
        
        q_physics_init = klf_inst.initial_guess_physics(ccfs1, ccf_r, delta, ts, te, h0_initial, num_days)
        print(f"Final Initial Guess:\n{q_physics_init}\nh0: {h0_initial:.4e}")

        @jit
        def calc_lnL(q_phys, h0_val):
            beta = klf_inst.get_beta_extended(q_phys, num_days)
            Pt_init = jnp.diag(jnp.array([q_phys[0], q_phys[1]]))
            at_init = jnp.array([1.0, q_phys[5]])
            _, _, lnL, _ = eKlf_jax(ccfs1, ccf_r, delta, ts, te, beta, h0_val, [q_phys[0], q_phys[1]], Pt_init, at_init)
            return lnL

        baseline_lnL = calc_lnL(q_physics_init, h0_initial)

        # --- TWO-STEP OPTIMIZATION (OFFICIAL LOGIC) ---
        print("\n--- Starting Two-Step Optimization ---")
        overall_start = time.time()
        
        # Step 1: Noise Baseline (Explanatory variables FIXED to 0)
        @jit
        def phi_to_q_step1(phi):
            qr = q_physics_init
            q_out = jnp.array([
                qr[0]*jnp.exp(phi[0]), qr[1]*jnp.exp(phi[1]), 
                qr[2], 0.0, 0.0, qr[5]+phi[3]*1E-3, 0.0, qr[7]
            ])
            h0_out = h0_initial * jnp.exp(phi[2])
            return q_out, h0_out
        
        @jit
        def objective_step1(phi):
            q_p, h0_v = phi_to_q_step1(phi)
            return -(calc_lnL(q_p, h0_v) - baseline_lnL)

        print("Step 1: Determining Noise Baseline...")
        phi_init_s1 = np.zeros(4)
        bounds_s1 = [(-15.0, 5.0), (-15.0, 5.0), (-2.0, 2.0), (-10.0, 10.0)]
        res_s1 = optimize.fmin_l_bfgs_b(lambda p: (float(objective_step1(p)), np.array(jax.grad(objective_step1)(p))), 
                                       phi_init_s1, bounds=bounds_s1, pgtol=1e-10)
        q_noise, h0_fixed = phi_to_q_step1(res_s1[0])

        # Step 2: Physical Responses (Q0 and h0 FIXED from Step 1)
        @jit
        def phi_to_q_step2(phi):
            qr = q_physics_init
            # Delay_rain is FIXED to 0.0
            q_out = jnp.array([
                q_noise[0], q_noise[1], 
                qr[2]*phi[0], phi[1]*1.0E-5, 0.0, 
                q_noise[5], qr[6]*phi[2], qr[7]*phi[3]
            ])
            return q_out, h0_fixed

        @jit
        def objective_step2(phi):
            q_p, h0_v = phi_to_q_step2(phi)
            return -(calc_lnL(q_p, h0_v) - baseline_lnL)

        print("Step 2: Fitting Physical Models (Rain, EQ) [Delay fixed to 0]...")
        qr = q_physics_init
        phi_init_s2 = jnp.array([1.0, 0.0, 1.0, 0.3])
        bounds_s2 = [(20.0/qr[2], 200.0/qr[2]), (-10.0, 10.0), (0.01, 100.0), (0.1, 30.0/qr[7])]
        
        # --- Hessian-based Auto-scaling ---
        print("  Calculating Hessian for auto-scaling...")
        # Compute diagonal of Hessian at the starting point to estimate sensitivity
        h_fn = jax.hessian(objective_step2)
        H_diag = jnp.abs(jnp.diag(h_fn(phi_init_s2)))
        # Scaling factor: 1/sqrt(curvature). Prevents sensitivity imbalance.
        scales = jnp.clip(1.0 / jnp.sqrt(H_diag + 1e-12), 1e-3, 1e3)
        print(f"  Auto-scales applied: {scales}")
        
        def objective_step2_scaled(theta):
            phi = theta * scales
            val, grad = jax.value_and_grad(objective_step2)(phi)
            # Chain rule: dL/dtheta = dL/dphi * dphi/dtheta = grad * scales
            return float(val), np.array(grad * scales)

        theta_init = phi_init_s2 / scales
        bounds_theta = [(b[0]/s, b[1]/s) for b, s in zip(bounds_s2, scales)]
        
        res_s2 = optimize.fmin_l_bfgs_b(objective_step2_scaled, 
                                       np.array(theta_init), bounds=bounds_theta, pgtol=1e-10)
        q_best, h0_best = phi_to_q_step2(res_s2[0] * scales)
        
        total_time = time.time() - overall_start
        print(f"Optimization Complete in {total_time:.2f} seconds.")
        print("-" * 40)
        print(f"Final Params: {q_best}")
        print(f"Final h0: {h0_best:.4e}")

        # --- Trade-off Sweep for All parameters ---
        num_steps = 31
        steps = jnp.linspace(-1.0, 1.0, num_steps)
        param_names = ["Q0_amp", "Q0_alpha", "Tau_rain", "Amp_rain", "Delay_rain", "at0_alpha", "Amp_eq", "Tau_eq", "h0"]
        baseline_lnL = calc_lnL(q_best, h0_best)
        results = []
        
        def get_sweep_q_h0(idx, step_val):
            q_tmp = jnp.array(q_best)
            h0_tmp = h0_best
            if idx in [0, 1]: q_tmp = q_tmp.at[idx].multiply(jnp.power(10.0, step_val * 2))
            elif idx in [3, 4, 5]: q_tmp = q_tmp.at[idx].add(step_val * 5.0 * (1e-5 if idx==3 else 1.0))
            elif idx == 8: h0_tmp = h0_tmp * jnp.power(10.0, step_val)
            else: q_tmp = q_tmp.at[idx].multiply(jnp.power(10.0, step_val))
            return q_tmp, h0_tmp

        for i in range(9):
            lnLs = vmap(lambda s: calc_lnL(*get_sweep_q_h0(i, s)))(steps)
            results.append(lnLs - baseline_lnL)
        results = jnp.stack(results)
        
        fig = plt.figure(figsize=(18, 32))
        gs = fig.add_gridspec(5, 3) 
        axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(3)]
        for i in range(9):
            val_best = q_best[i] if i < 8 else h0_best
            if i in [0, 1, 2, 6, 7, 8]:
                sweep_vals = val_best * jnp.power(10.0, steps * (2 if i < 2 else 1))
                axes[i].plot(sweep_vals, results[i], lw=3)
                axes[i].set_xscale('log')
                axes[i].set_xlabel("Value")
            else:
                axes[i].plot(val_best + steps * 5.0 * (1e-5 if i==3 else 1.0), results[i], lw=3)
                axes[i].set_xlabel("Additive")
            
            axes[i].set_title(f"{param_names[i]}\nBest: {val_best:.2e}")
            axes[i].axvline(val_best, color='red', ls='--')
            axes[i].grid(True, alpha=0.3)
            
        dates = [klf_inst.base_date + datetime.timedelta(days=int(i)) for i in range(num_days)]
        ax_amp = fig.add_subplot(gs[3, :])
        beta = klf_inst.get_beta_extended(q_best, num_days)
        alpha_t, _, _, _ = eKlf_jax(ccfs1, ccf_r, delta, ts, te, beta, h0_best, [q_best[0], q_best[1]], jnp.diag(jnp.array([q_best[0], q_best[1]])), jnp.array([1.0, q_best[5]]))
        mean_A = float(jnp.nanmean(alpha_t[:, 0]))
        ax_amp.plot(dates, alpha_t[:, 0], color='C0', label='Amplitude (A)', lw=1.0)
        ax_amp.set_ylabel(f'Amplitude (A) [Mean: {mean_A:.3f}]')
        ax_amp.grid(True, alpha=0.3)
        ax_amp.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        ax_dv = fig.add_subplot(gs[4, :])
        ax_dv.plot(dates, alpha_t[:, 1], label='Residual alpha', color='C1', lw=1.5)
        ax_dv.plot(dates, beta, label='Rain Model beta', color='C2', ls='--', lw=1.5)
        ax_dv.plot(dates, alpha_t[:, 1] + beta, label='Total alpha+beta', color='C3', ls=':', lw=2)
        ax_dv.set_ylabel('Velocity Change (dv/v)')
        ax_dv.grid(True, alpha=0.3)
        ax_dv.legend()
        ax_dv.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        plt.tight_layout()
        plt.savefig("Trade_off_jax_REVERTED_STABLE_v3.png")
        print("Success: Final plots saved.")

if __name__ == "__main__":
    main()
