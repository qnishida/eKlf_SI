# An Implementation of an extended Kalman filter/smoother for time-lapse monitoring of seismic velocity

## Description
An Implementation of an extended Kalman filter/smoother for time-lapse monitoring of seismic velocity using the stretching method. See Nishida et al. (2020) for details.

* Nishida, K., Mizutani, Y., Ichihara, M., & Aoki, Y. (2020). Time-lapse monitoring of seismic velocity associated with 2011 Shinmoe-dake eruption using seismic interferometry: An extended Kalman filter approach. Journal of Geophysical Research: Solid Earth, 125, e2020JB020180. https://doi.org/10.1029/2020JB020180

---

## New Version (v2.0.1): JAX-Optimized EKF
A new version optimized with [JAX](https://github.com/google/jax) is now available. This version provides significant speedups, improved numerical stability, and a more robust optimization strategy.

### Key Improvements (v2.0.1)
- **FFT Pre-upsampling**: Implements 4x upsampling of reference waveforms to smooth the likelihood surface and stabilize L-BFGS-B convergence.
- **Hybrid Diagonal Hessian**: Combines JAX automatic differentiation with finite differences to estimate parameter sensitivity efficiently (~25% memory reduction vs. full Hessian).
- **JAX Acceleration**: Leverages XLA compilation for high-speed hyperparameter optimization.
- **Two-Step Optimization**: Implements a staged workflow that separately determines noise baselines and physical response parameters to prevent overfitting.

### Installation
```bash
pip install jax jaxlib numpy scipy matplotlib h5py
```
*Note: For optimal precision, enabling 64-bit support in JAX is highly recommended:*
```python
from jax import config
config.update("jax_enable_x64", True)
```

### Usage (Two-Step Optimization)
To maintain physical integrity, we recommend the following two-step process:
1. **Step 1 (Noise Baseline)**: Estimate process noise ($Q_0$) and observation noise ($h_0$) with physical models disabled.
2. **Step 2 (Physical Parameters)**: Fit physical response models (Precipitation, EQ recovery) while keeping noise parameters fixed.

Refer to `plot_tradeoff_jax.py` for a complete example of this workflow.

---

## Files 
* `KalmanFilter_jax.py`: The latest JAX-optimized EKF engine.
* `KalmanFilter_numpy.py`: Equivalent interpolation-based logic implemented using standard NumPy (no JAX dependency).
* `plot_tradeoff_jax.py`: A comprehensive sample script demonstrating the two-step optimization and result visualization.
* `Script/test_benchmark.py`: A performance benchmarking script to measure execution time and memory usage.
* `KalmanFilter.py`: Legacy Taylor-series based implementation (archived in `legacy/`).
* `gwl_ebino_seikei.dat`: The precipitation data (AMeDAS) at Ebino station. 
* `est_param_with_quake.py`: Sample code for the legacy implementation.

## Data format of the precipitation data at Ebino (gwl_ebino_seikei.dat)
YY/MM/DD, YYYY, MM, DD, Precipitation(mm), Days from 4/30 2010 (1 means the 1st day of seismic data on 5/1 2010)

In situ precipitation observations were obtained from the Automated Meteorological Data Acquisition System (AMeDAS) of the Japan Meteorological Agency (JMA) are available at http://www.data.jma.go.jp/obd/stats/etrn/index.php (in Japanese).

## Data format of cross-correlation functions (CCFs)
The CCFs are stored in HDF5 format (6Gb). The header includes the station locations, date, station elevation, the sampling interval, number of points, and the components. The file is available at zenodo (https://doi.org/10.5281/zenodo.2539824). 
* long-STS-2_bp0.10to0.40white.h5

## Typo of eq. 34 (Nishida et al. 2020)
* Nishida, K., Mizutani, Y., Ichihara, M., & Aoki, Y. (2020). Time-lapse monitoring of seismic velocity associated with 2011 Shinmoe-dake eruption using seismic interferometry: An extended Kalman filter approach. Journal of Geophysical Research: Solid Earth, 125, e2020JB020180. https://doi.org/10.1029/2020JB020180

### Error
![image](https://user-images.githubusercontent.com/10939329/209293501-78138321-8265-4576-9992-581f3b02f117.png)
### Correction 
![image](https://user-images.githubusercontent.com/10939329/209293546-cc895e07-582e-4a67-a4c9-3918371e3867.png)
