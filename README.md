# Kirishima
# Imprementation of Extended Kalman filter for the streching method

* Kalmanfilter.py: An implementation of the extended Kalman filter
* gwl_ebino_seikei.dat: The precipitation data (AMeDAS) at Ebino station. 
* long-STS-2_bp0.10to0.40white.h5: The data set of daily cross-correlation functions for all the pair of stations.
* est_param_with_quake.py: A sample python code for estimating temporal variations of the seismic velocities. 

## Data format of the precipitation data at Ebino (gwl_ebino_seikei.dat)
YY/MM/DD, YYYY, MM, DD, Precipitation(mm), Days from 4/30 2010 (1 means the 1st day of seismic data on 5/1 2010)

In situ precipitation observations were obtained from the Automated Meteorological Data Acquisition System (AMeDAS) of the Japan Meteorological Agency (JMA) are available at http://www.data.jma.go.jp/obd/stats/etrn/index.php (in Japanese).

## Data format of cross-correlation functions (CCFs)
The CCFs are stored in HDF5 format. The header includes the station locations, date, station elevation, the sampling interval, number of points, and the components. 


