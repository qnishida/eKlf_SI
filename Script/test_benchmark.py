import jax
import jax.numpy as jnp
from jax import jit, vmap, config, value_and_grad, lax
import numpy as np
import h5py
from scipy import signal, optimize
import os
import time
import datetime
import psutil
from KalmanFilter_jax import eKlf_jax, T_jax, est_h0_jax, cal_ref_jax, deri_jax

# Enable float64
config.update("jax_enable_x64", True)

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Return in MB

class KlfBenchmark:
    def __init__(self, precipitation_path="./Data/gwl_ebino_seikei.dat"):
        with open(precipitation_path, "r") as f:
            lines = f.readlines()
        self.precip_raw = jnp.array([float(line.split()[4]) for line in lines])
        self.flags = jnp.array([int(line.split()[5]) for line in lines])
        self.num0 = int(jnp.where(self.flags == 1)[0][0])
        self.precipitation = self.precip_raw - jnp.mean(self.precip_raw)
        self.t_kmt = 2177 
        self.q_ref = jnp.array([1E-6, 4E-9, 100., 1E-7, 0.0, 0.0, 1E-3, 30.0])
        self.base_date = datetime.datetime(2010, 5, 1)

    def get_beta_extended(self, q_params, num_days):
        T0 = T_jax(jnp.arange(len(self.precipitation), dtype=jnp.float64), q_params[2])
        full_conv = jax.scipy.signal.convolve(self.precipitation, T0, mode='full')
        beta = lax.dynamic_slice(full_conv, (self.num0,), (num_days,))
        x = jnp.arange(num_days, dtype=jnp.float64)
        x_m, y_m = jnp.mean(x), jnp.mean(beta)
        slope = jnp.sum((x - x_m) * (beta - y_m)) / jnp.sum((x - x_m)**2)
        beta_clean = q_params[3] * (beta - y_m - slope * (x - x_m))
        eq_term = jnp.where(x >= self.t_kmt, q_params[6] * jnp.exp(-(x - self.t_kmt) / jnp.maximum(q_params[7], 1.0)), 0.0)
        return beta_clean + eq_term

    def initial_guess_physics(self, ccfs1, ccf_r_dense, ccf_deri_dense, ccf_r_orig, delta, ts, te, h0, num_days):
        pt_init_mat = jnp.diag(jnp.array([1E-2, 1E-2]))
        alpha_t, _, _, mask = eKlf_jax(ccfs1, ccf_r_dense, ccf_deri_dense, ccf_r_orig, delta, ts, te, jnp.zeros(num_days), h0, [1E-5, 1E-8], pt_init_mat, jnp.array([1.0, 0.0]))
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
            zero_lag_idx = np.where(lags == 0)[0][0]
            min_corr = corr[zero_lag_idx]
            u1_sq = np.sum(u1**2)
            if u1_sq == 0: return 1e20, 0.0
            gamma = min_corr / u1_sq
            cost = -2*min_corr*gamma + gamma*gamma*u1_sq + np.sum(u0**2)
            return cost, gamma
        tau_grid = [20., 50., 100., 200., 500.]
        results = [rain_cost_with_delay(t) for t in tau_grid]
        best_idx = np.argmin([r[0] for r in results])
        best_tau = tau_grid[best_idx]
        _, best_gamma = results[best_idx]
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
        new_q_ref = self.q_ref.at[2].set(best_tau).at[3].set(best_gamma).at[6].set(best_A_eq).at[7].set(best_tau_eq)
        return new_q_ref

def main():
    file_path = "./Data/long-STS-2_bp0.15to0.90white.h5"
    benchmark_inst = KlfBenchmark()
    
    with h5py.File(file_path, 'r') as h5file:
        delta = (int)(h5file.attrs['delta'] * 1000 + 0.5) / 1000
        ts, te = int(20 / delta), int(100 / delta)
        sta_pair = ("04", "03") 
        num_days = min(h5file[sta_pair[0]][sta_pair[1]]["ccfs"].shape[2], len(benchmark_inst.precipitation) - benchmark_inst.num0)
        ccfs1 = jnp.array(h5file[sta_pair[0]][sta_pair[1]]["ccfs"][:, :, :num_days, :])
        win2 = signal.windows.tukey(te - ts, 0.1)
        win = np.zeros(1024)
        for i in range(ts, te): win[i + 512], win[512 - i] = win2[i - ts], win2[i - ts]
        ccfs1 = ccfs1 * jnp.array(win)
        
        h0_initial, ccf_r_orig = est_h0_jax(ccfs1, None, ts, te)
        
        def upsample_reference(ref_ccf, delta_val):
            factor = 4
            n_orig = ref_ccf.shape[-1]
            n_new = n_orig * factor
            ccf_dense = signal.resample(ref_ccf, n_new, axis=-1).astype(np.float64)
            deri_orig = deri_jax(ref_ccf, 1, delta_val)
            deri_dense = signal.resample(deri_orig, n_new, axis=-1).astype(np.float64)
            return ccf_dense, deri_dense

        ccf_r_dense, ccf_deri_dense = upsample_reference(ccf_r_orig, delta)

        q_init_lsq = benchmark_inst.initial_guess_physics(ccfs1, ccf_r_dense, ccf_deri_dense, ccf_r_orig, delta, ts, te, h0_initial, num_days)
        beta_init = benchmark_inst.get_beta_extended(q_init_lsq, num_days)
        at_pre, _, _, _ = eKlf_jax(ccfs1, ccf_r_dense, ccf_deri_dense, ccf_r_orig, delta, ts, te, beta_init, h0_initial, [q_init_lsq[0], q_init_lsq[1]], jnp.diag(jnp.array([1E-2, 1E-2])), jnp.array([1.0, 0.0]))
        ccf_r_orig = cal_ref_jax(ccfs1, at_pre, beta_init, delta)
        ccf_r_dense, ccf_deri_dense = upsample_reference(ccf_r_orig, delta)
        h0_initial, _ = est_h0_jax(ccfs1, ccf_r_orig, ts, te)
        
        q_physics_init = benchmark_inst.initial_guess_physics(ccfs1, ccf_r_dense, ccf_deri_dense, ccf_r_orig, delta, ts, te, h0_initial, num_days)

        @jit
        def calc_lnL(q_phys, h0_val):
            beta = benchmark_inst.get_beta_extended(q_phys, num_days)
            Pt_init = jnp.diag(jnp.array([1E-2, 1E-2]))
            at_init = jnp.array([1.0, 0.0])
            _, _, lnL, _ = eKlf_jax(ccfs1, ccf_r_dense, ccf_deri_dense, ccf_r_orig, delta, ts, te, beta, h0_val, [q_phys[0], q_phys[1]], Pt_init, at_init)
            return lnL

        baseline_lnL = calc_lnL(q_physics_init, h0_initial)

        report = []
        report.append(f"Benchmark Date: {datetime.datetime.now()}")
        report.append(f"Model: v2.0.1 Final Configuration (FIXED at0=0.0, Delay=0.0, Pt_init=1E-2)")
        report.append(f"Data: {sta_pair}, {num_days} days")
        report.append("-" * 40)

        # Step 1: Noise Baseline
        print("Executing Step 1...")
        @jit
        def phi_to_q_step1(phi):
            qr = q_physics_init
            q_out = jnp.array([qr[0]*jnp.exp(phi[0]), qr[1]*jnp.exp(phi[1]), qr[2], 0.0, 0.0, 0.0, 0.0, qr[7]])
            h0_out = h0_initial * jnp.exp(phi[2])
            return q_out, h0_out
        
        @jit
        def objective_step1(phi):
            q_p, h0_v = phi_to_q_step1(phi)
            return -(calc_lnL(q_p, h0_v) - baseline_lnL)

        start_s1 = time.time()
        phi_init_s1 = np.zeros(3)
        bounds_s1 = [(-15.0, 5.0), (-15.0, 5.0), (-2.0, 2.0)]
        x_s1, f_s1, d_s1 = optimize.fmin_l_bfgs_b(lambda p: (float(objective_step1(p)), np.array(jax.grad(objective_step1)(p))), 
                                                 phi_init_s1, bounds=bounds_s1, pgtol=1e-10)
        time_s1 = time.time() - start_s1
        mem_s1 = get_memory_usage()
        q_noise, h0_fixed = phi_to_q_step1(x_s1)

        report.append("Step 1: Noise Baseline")
        report.append(f"  Runtime: {time_s1:.4f} seconds")
        report.append(f"  Memory Usage: {mem_s1:.2f} MB")
        report.append(f"  Iterations: {d_s1['nit']}")
        report.append(f"  Function Calls: {d_s1['funcalls']}")
        report.append(f"  Status: {d_s1['warnflag']} ({d_s1['task']})")
        report.append("")

        # Step 2: Physical Responses
        print("Executing Step 2...")
        @jit
        def phi_to_q_step2(phi):
            qr = q_physics_init
            q_out = jnp.array([q_noise[0], q_noise[1], qr[2]*phi[0], phi[1]*1.0E-5, 0.0, 0.0, qr[6]*phi[2], qr[7]*phi[3]])
            return q_out, h0_fixed

        @jit
        def objective_step2(phi):
            q_p, h0_v = phi_to_q_step2(phi)
            return -(calc_lnL(q_p, h0_v) - baseline_lnL)

        start_s2 = time.time()
        phi_init_s2 = jnp.array([1.0, 0.0, 1.0, 0.3])
        bounds_s2 = [(20.0/q_physics_init[2], 200.0/q_physics_init[2]), (-10.0, 10.0), (0.01, 100.0), (0.1, 30.0/q_physics_init[7])]
        
        grad_fn = jax.grad(objective_step2)
        g_base = grad_fn(phi_init_s2)
        eps = 1e-4
        H_diag_list = []
        for i in range(len(phi_init_s2)):
            phi_p = phi_init_s2.at[i].add(eps)
            g_p = grad_fn(phi_p)
            h_ii = (g_p[i] - g_base[i]) / eps
            H_diag_list.append(h_ii)
        H_diag = jnp.abs(jnp.array(H_diag_list))
        scales = jnp.clip(1.0 / jnp.sqrt(H_diag + 1e-12), 1e-3, 1e3)
        
        def objective_step2_scaled(theta):
            phi = theta * scales
            val, grad = jax.value_and_grad(objective_step2)(phi)
            return float(val), np.array(grad * scales)

        theta_init = phi_init_s2 / scales
        bounds_theta = [(b[0]/s, b[1]/s) for b, s in zip(bounds_s2, scales)]
        
        x_s2_scaled, f_s2, d_s2 = optimize.fmin_l_bfgs_b(objective_step2_scaled, 
                                                        np.array(theta_init), bounds=bounds_theta, pgtol=1e-10)
        time_s2 = time.time() - start_s2
        mem_s2 = get_memory_usage()
        q_best, h0_best = phi_to_q_step2(x_s2_scaled * scales)

        report.append("Step 2: Physical Responses")
        report.append(f"  Runtime: {time_s2:.4f} seconds")
        report.append(f"  Memory Usage: {mem_s2:.2f} MB")
        report.append(f"  Iterations: {d_s2['nit']}")
        report.append(f"  Function Calls: {d_s2['funcalls']}")
        report.append(f"  Status: {d_s2['warnflag']} ({d_s2['task']})")
        report.append("-" * 40)
        
        peak_mem = get_memory_usage()
        report.append(f"Peak Process Memory: {peak_mem:.2f} MB")
        report.append(f"Total Optimization Time: {time_s1 + time_s2:.4f} seconds")

        os.makedirs("log", exist_ok=True)
        with open("log/benchmark_wo_at0.txt", "w") as f:
            f.write("\n".join(report))
        print("Benchmark (with full stats) complete.")

if __name__ == "__main__":
    main()
