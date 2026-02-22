import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, config
import numpy as np
import h5py
import time
import datetime
import pickle
import os
from scipy import signal, fftpack
from KalmanFilter_jax import eKlf_jax, T_jax

# Enable float64 based on environment variable, default to False (float32)
if os.environ.get("JAX_ENABLE_X64", "False").lower() == "true":
    config.update("jax_enable_x64", True)
    print("JAX: Double precision (float64) enabled.")
else:
    config.update("jax_enable_x64", False)
    print("JAX: Single precision (float32) enabled.")

# We'll use a JAX-compatible optimizer. 
# If jaxopt is not available, we can use scipy.optimize with JAX gradients.
try:
    import jaxopt
    HAS_JAXOPT = True
except ImportError:
    HAS_JAXOPT = False
    from scipy import optimize

class KlfJax:
    def __init__(self, precipitation_path="./gwl_ebino_seikei.dat"):
        with open(precipitation_path, "r") as f:
            lines = f.readlines()
        self.precipitation = jnp.array([float(line.split()[4]) for line in lines])
        self.num0 = 0
        for i in range(len(lines)):
            if int(lines[i].split()[5]) == 1:
                self.num0 = i
        self.precipitation -= jnp.mean(self.precipitation)
        self.q_ref = jnp.array([1E-5, 4E-9, 100., 6.726449064255413e-07, -5., 1E-3, -1E-3, 30.])

    def get_beta(self, q_params):
        # q0[2] = τ, q0[4] = delay
        T0 = T_jax(jnp.arange(len(self.precipitation)), q_params[2])
        
        # JAX version of the delay logic
        Y = jnp.fft.fft(self.precipitation)
        freqs = jnp.fft.fftfreq(len(Y))
        Y2 = Y * jnp.exp(2j * jnp.pi * freqs * (-q_params[4]))
        y2 = jnp.real(jnp.fft.ifft(Y2))
        
        # Convolution in JAX
        # jax.scipy.signal.convolve is available
        beta = jax.scipy.signal.convolve(y2, T0, mode='full')[self.num0:len(y2) + self.num0]
        # Trim or pad beta to match length if necessary, here we follow original slicing
        beta = beta[:len(y2) - self.num0]
        
        beta = beta - jnp.mean(beta)
        # Detrending in JAX (simple linear regression subtraction)
        x = jnp.arange(len(beta))
        coeffs = jnp.polyfit(x, beta, 1)
        beta = beta - (coeffs[0] * x + coeffs[1])
        
        return beta

def create_likelihood_fn(klf_jax_inst, ccfs1, ccf_r, delta, ts, te, h0):
    @jit
    def likelihood(qm):
        q0 = qm * klf_jax_inst.q_ref
        beta = klf_jax_inst.get_beta(q0)
        
        # q0[0]: Q0[0], q0[1]: Q0[1], q0[3]: beta_scale, q0[5]: at0[0], q0[6]: at0[1]
        # Original: Pt=np.array([[q0[0],0],[0,q0[1]]]), at0 = [q0[5],q0[6]]
        Pt_init = jnp.diag(jnp.array([q0[0], q0[1]]))
        at_init = jnp.array([q0[5], q0[6]])
        
        _, _, lnL, _ = eKlf_jax(
            ccfs1, ccf_r, delta, ts, te, 
            q0[3] * beta, h0, [q0[0], q0[1]], 
            Pt_init, at_init
        )
        return -lnL
    
    return likelihood

def main():
    date0 = datetime.datetime(2010, 5, 1)
    date1 = datetime.datetime(2018, 9, 1)
    
    freq_band = "0.15to0.90"
    file_path = "./long-STS-2_bp" + freq_band + "white.h5"
    
    klf_jax_inst = KlfJax()
    
    results = {}

    with h5py.File(file_path, 'r') as h5file:
        delta = (int)(h5file.attrs['delta'] * 1000 + 0.5) / 1000
        ts = int(20 / delta)
        te = int(100 / delta)
        
        station0 = ["00", "01", "02", "03", "04", "06", "07", "09"]
        sta_pairs = []
        for i in range(len(station0)):
            for j in range(i+1, len(station0)):
                sta_pairs.append((station0[j], station0[i]))

        # Tukey window setup
        idx = jnp.arange(1024)
        mask_tm = ((idx >= ts + 512) & (idx < te + 512)) | ((idx > 512 - te) & (idx <= 512 - ts))
        win2 = signal.windows.tukey(te - ts, 0.1)
        win = np.zeros(1024)
        for i in range(ts, te):
            win[i + 512], win[512 - i] = win2[i - ts], win2[i - ts]
        win = jnp.array(win)

        for sta_pair in sta_pairs:
            print(f"Processing {sta_pair}...")
            ccfs1 = jnp.array(h5file[sta_pair[0]][sta_pair[1]]["ccfs"][:, :, :, :]) * win
            
            # Initial reference CCF and h0
            # For simplicity using numpy for initial h0 and ref if they don't need gradients
            mccfs = np.ma.masked_array(np.array(ccfs1), np.isnan(ccfs1))
            ccf_r = jnp.array(mccfs.mean(axis=2)) # (3, 3, 1024)
            
            # Use the JAX version of h0 estimation
            from KalmanFilter_jax import est_h0_jax
            h0 = est_h0_jax(ccfs1, ccf_r, ts, te)
            print(f"h0: {h0}")

            likelihood_fn = create_likelihood_fn(klf_jax_inst, ccfs1, ccf_r, delta, ts, te, h0)
            
            # Initial parameters
            qm_init = jnp.array([1., 1., 1., 1., 1., 1., 0.5])
            bounds = (jnp.array([5E-2, 1E-3, 0.1, 0.6, 0., 0.5, -100.]),
                      jnp.array([2E1, 2E1, 2.0, 1.3, 2.0, 2.0, 100.]))

            if HAS_JAXOPT:
                print("Using jaxopt.LBFGSB")
                # L-BFGS-B with bounds
                pgv = jaxopt.LBFGSB(fun=likelihood_fn, tol=1e-6)
                res = pgv.run(qm_init, bounds=bounds)
                q_est = res.params
            else:
                print("Using scipy.optimize with JAX gradients")
                # Use scipy with JAX's value_and_grad
                def scipy_obj(qm):
                    val, grad = value_and_grad(likelihood_fn)(qm)
                    return float(val), np.array(grad)
                
                # Reshape bounds for scipy
                scipy_bounds = list(zip(bounds[0].tolist(), bounds[1].tolist()))
                q_est, f_min, info = optimize.fmin_l_bfgs_b(
                    scipy_obj, np.array(qm_init), bounds=scipy_bounds, 
                    pgtol=1e-6, epsilon=1e-8
                )
            
            print(f"Optimized q: {q_est * klf_jax_inst.q_ref}")
            
            # Final run to get smoothed results
            q0 = q_est * klf_jax_inst.q_ref
            beta = klf_jax_inst.get_beta(q0)
            Pt_init = jnp.diag(jnp.array([q0[0], q0[1]]))
            at_init = jnp.array([q0[5], q0[6]])
            αt, Vt, lnL, mask_param = eKlf_jax(
                ccfs1, ccf_r, delta, ts, te, 
                q0[3] * beta, h0, [q0[0], q0[1]], 
                Pt_init, at_init
            )
            
            results[sta_pair] = {
                "αt": np.array(αt),
                "Vt": np.array(Vt),
                "β": np.array(beta),
                "mask": np.array(mask_param),
                "qm": np.array(q0)
            }

    with open(f'out_jax_{datetime.date.today()}.pickle', mode='wb') as fo:
        pickle.dump(results, fo)

if __name__ == "__main__":
    main()
