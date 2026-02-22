# An Implementation of an extended Kalman filter/smoother for time-lapse monitoring of seismic velocity

## Description
An Implementation of an extended Kalman filter/smoother for time-lapse monitoring of seismic velocity using the stretching method. See Nishida et al. (2020) for details.

* Time-lapse monitoring of seismic velocity associated with 2011 Shinmoe-dake eruption using seismic interferometry, Nishida, Kiwamu, Mizutani, Yuta, Ichihara, Mie, Aoki, Yosuke, ESSOAar, https://doi.org/10.1002/essoar.10503078.1 (2020)

---

## New Version (v2.0): JAX-Optimized EKF
A new version optimized with [JAX](https://github.com/google/jax) is now available. This version provides significant speedups, improved numerical stability, and a more robust optimization strategy.

### Key Improvements
- **JAX Acceleration**: Leverages automatic differentiation and XLA compilation for high-speed hyperparameter optimization.
- **Two-Step Optimization**: Implements a staged workflow that separately determines noise baselines and physical response parameters to prevent overfitting.
- **High-Precision Stretching**: Replaces Taylor-series approximations with direct linear interpolation for more accurate waveform stretching.
- **Enhanced Stability**: Features Woodbury matrix identity updates and precise eigenvalue calculations.

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
* Time-lapse monitoring of seismic velocity associated with 2011 Shinmoe-dake eruption using seismic interferometry, Nishida, Kiwamu, Mizutani, Yuta, Ichihara, Mie, Aoki, Yosuke, ESSOAar, https://doi.org/10.1002/essoar.10503078.1 (2020)

### Error
![image](https://user-images.githubusercontent.com/10939329/209293501-78138321-8265-4576-9992-581f3b02f117.png)
### Correction 
![image](https://user-images.githubusercontent.com/10939329/209293546-cc895e07-582e-4a67-a4c9-3918371e3867.png)
