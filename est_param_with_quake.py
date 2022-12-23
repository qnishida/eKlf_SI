from scipy import signal
from scipy import optimize
from scipy import fftpack
import time
import datetime
import numpy as np
import h5py
import pickle

FLAG_PLOT = False
if FLAG_PLOT: 
    import matplotlib.pyplot as plt

from KalmanFilter import Klf
stnm = {"00":"EvKVO","01":"EvSMW","02":"EvSMN","03":"EvTKS","04":"EvTKW","05":"NTKOF","06":"SMJNH","07":"SKRMV","08":"BKRMV","09":"SKRHV","10":"BKRHV"}
date_kumamoto = datetime.datetime(2016,4,16)
date0 = datetime.datetime(2010,5,1)
date1 = datetime.datetime(2018,9,1)
Δ     = datetime.timedelta(days=1)
t_kmt = (date_kumamoto-date0).days

counter = 0

#def likelihood(qm, flag=0,ref=0): #q0[0]: amp, q0[1] = σm^2, q0[2] = τ, q0[3] = amp, q0[4] = delay, q0[5] = Pt[0], q0[6] = Pt[1], q0[7] = at0[1]
def likelihood(qm, flag=0,ref=0): #q0[0]: amp, q0[1] = σm^2, q0[2] = τ, q0[3] = amp, q0[4] = delay, q0[5] = at0[0], q0[6] = at0[1]
    q0 = qm * Klf.q_ref
    #T0 = np.exp(-np.arange(len(Klf.precipitation))/np.exp(q0[2]))
    T0 = Klf.T(np.arange(len(Klf.precipitation)),q0[2])
    Y = fftpack.fft(Klf.precipitation)
    Y2= Y*np.exp(2j*np.pi*fftpack.fftfreq(len(Y))*(-q0[4]))
    y2 = np.real(fftpack.ifft(Y2))
    β = signal.convolve(y2, T0)[Klf.num0:len(y2)]
    β = β-β.mean()
    β = q0[3]*signal.detrend(β)
    tx = np.arange(0,len(β),1)
    
    if FLAG_PLOT:
        global counter
        if counter %45 == 0: plt.plot(β)
    β[t_kmt:] += (q0[6]*np.exp(-(tx[t_kmt:]-t_kmt)/q0[7]))
    if FLAG_PLOT:
        if counter %45== 0:plt.plot(β)

    αt,Vt,lnL, mask_param = Klf.eKlf(ccfs1,ccf_r,delta,ts,te,β,h0,[q0[0],q0[1]], Pt=np.array([[q0[0],0],[0,q0[1]]]),at0 = [1,q0[5]])
    if FLAG_PLOT:
        if counter %45 == 0:plt.plot(αt[0:len(β),1])
        if counter %45 == 0:plt.show()
        counter += 1
    
    if flag == 0:
        np.set_printoptions(precision=2,linewidth=200)
        print(q0,lnL-ref,lnL, flush=True)
        np.set_printoptions(precision=8)
        return(-lnL+ref)
    if flag == 1:
        print(q0,lnL)
        return(αt,Vt,lnL, β, mask_param )



freq_band = "0.15to0.90"
file = "./long-STS-2_bp"+freq_band+"white.h5"

Klf.init()
αt_all = {}
Vt_all = {}
mask_all = {}
q_all = {}
β_all = {}

with h5py.File(file,'r') as h5file:
    delta= (int)(h5file.attrs['delta']*1000+0.5)/1000
    ts = int(20/delta)
    te = int(100/delta)
    start = time.time()
    mask_tm = [False]*1024
    for i in range(ts,te): mask_tm[i+512] , mask_tm[512-i] = True, True
    win =  np.zeros(1024)
    win2= signal.tukey(te-ts,.1)
    for i in range(ts,te): win[i+512] , win[512-i] = win2[i-ts], win2[i-ts]
    ########
    npts = h5file.attrs['npts']
    #station0 = ["00","01","02","03","04","06","07","09"]
    station0 = ["00","01","02","03","04","05","06","07","08","09","10"]
    sta_pairs = []
    for i in range(len(station0)):
        for j in range(i+1,len(station0)):
            sta_pairs.append((station0[j],station0[i]))
    
    for sta_pair in sorted(list(set(sta_pairs))): #[("04","03")]: [("07","04")]:#[("06","03")]: #[("02","01")]: #
        print("#",sta_pair,"Time:{0}".format(time.time() - start))
        ccfs1 = h5file[sta_pair[0]][sta_pair[1]]["ccfs"][:,:,:,:]*win
        ave = np.zeros([ccfs1.shape[0],ccfs1.shape[1],ccfs1.shape[2]])
        ccf_r = np.zeros([ccfs1.shape[0],ccfs1.shape[1],ccfs1.shape[3]])
        β  = np.zeros(len(ccfs1[0][0])+1)
        ave= np.zeros(len(ccfs1[0][0])+1)
        
        h0 = Klf.est_h0(ccfs1,ccf_r,ts,te)
        print(h0)
        β.fill(0)
        #αt,Vt,lnL, mask_param = Klf.eKlf(ccfs1,ccf_r,delta,ts,te,β,h0,[1E-6,1E-9])
        #αt,Vt,lnL, mask_param = Klf.eKlf(ccfs1,ccf_r,delta,ts,te,β,h0,[2E-5,4E-8])
        αt,Vt,lnL, mask_param = Klf.eKlf(ccfs1,ccf_r,delta,ts,te,β,h0,[1E-5,4E-9])
        print("Likelihood = ",lnL)
        #Search for the optimized paramters

        mask2 = [np.count_nonzero(tmp)==9 for tmp in mask_param] #mask_param.copy()
        mask2[2100:-1] = [False]*(len(mask2)-2100-1) #Cut after Kumamoto earthquake

        ##For precipitation
        Klf.q_ref[2] = 100. #days[idx]
        β2,ave,res, amp, γ  = Klf.cal_gwl(αt,Klf.q_ref[2],mask2)
        β[0:len(β2)] = β2
        Klf.q_ref[3] = γ
        print("gamma = ", γ)
        
        #For Kumamoto earthauake return res,y_ref,X,ye
        Klf.q_ref[7] = 30.
        res,A,y_ref,ye  = Klf.cal_kumamoto(t_kmt, αt,Klf.q_ref[7],mask2)
        Klf.q_ref[6] = A
        #Klf.q_ref[9] = y_ref

        print("#")
        q_init = [1.,1.,1.,1.,1.,1.,1.,1.]        
        bounds = [[5E-2,2E1],[1E-3,2E1],[0.1,2.],[0.1,2.],[0,2],[-10,10],[1E-3,1E1],[0.1,10]]
        q_est, min, log=optimize.fmin_l_bfgs_b(lambda q_tmp:likelihood([q_tmp[0],q_tmp[1],q_tmp[2],q_tmp[3],q_tmp[4],q_tmp[5],q_tmp[6],q_tmp[7]],ref=lnL)\
                                               ,q_init,bounds=bounds, approx_grad = True,epsilon=1E-5,iprint=99,disp=1,factr=100000000)#pgtol=1E-6,epsilon=1E-8,iprint=-1
        print("#",log)
        αt,Vt,lnL, β, mask_param = likelihood(q_est,flag=1)
        αt_all[sta_pair] = αt
        Vt_all[sta_pair] = Vt
        mask_all[sta_pair] = mask_param
        q_all[sta_pair] = q_est*Klf.q_ref
        β_all[sta_pair] = β
data_all = {"αt":αt_all, "Vt":Vt_all,"β":β_all, "mask":mask_all, "qm":q_all}

with open('out_weq'+str(datetime.date.today())+'.pickle', mode='wb') as fo:
    pickle.dump(data_all, fo)
