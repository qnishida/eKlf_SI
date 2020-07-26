# An Implementation of an extended Kalman filter/smoother for time-lapse monitoring of seismic velocity

## Description
An Implementation of an extended Kalman filter/smoother for time-lapse monitoring of seismic velocity using the stretching method. See Nishida et al. (2020) for details.

*Time-lapse monitoring of seismic velocity associated with 2011 Shinmoe-dake eruption using seismic interferometry, Nishida, Kiwamu, Mizutani, Yuta, Ichihara, Mie, Aoki, Yosuke, ESSOAar, https://doi.org/10.1002/essoar.10503078.1 (2020)

## Files 
* Kalmanfilter.py: An implementation of the extended Kalman filter/smoother
* gwl_ebino_seikei.dat: The precipitation data (AMeDAS) at Ebino station. 
* est_param_with_quake.py: A sample python code for estimating temporal variations of the seismic velocities. This code also estimates the hyper-parameters using Maximum Likelihood Method.

## Data format of the precipitation data at Ebino (gwl_ebino_seikei.dat)
YY/MM/DD, YYYY, MM, DD, Precipitation(mm), Days from 4/30 2010 (1 means the 1st day of seismic data on 5/1 2010)

In situ precipitation observations were obtained from the Automated Meteorological Data Acquisition System (AMeDAS) of the Japan Meteorological Agency (JMA) are available at http://www.data.jma.go.jp/obd/stats/etrn/index.php (in Japanese).

## Data format of cross-correlation functions (CCFs)
The CCFs are stored in HDF5 format. The header includes the station locations, date, station elevation, the sampling interval, number of points, and the components. The file will be available at zenodo (6GB). 
* long-STS-2_bp0.10to0.40white.h5


