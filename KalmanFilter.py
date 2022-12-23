import numpy as np
import math 
from scipy import signal
from scipy import interpolate
from scipy import fftpack
import datetime

# The class of the extended Kalman filter
class Klf:  
    def init():#Initialize precipitation data
        with open("./gwl_ebino_seikei.dat","r") as f: lines = f.readlines()
        Klf.precipitation = np.array([float(line.split()[4]) for line in lines])
        for i in range(len(lines)): # The search for the 1st day of seismic data
            if int(lines[i].split()[5]) == 1: 
                Klf.num0 = i
        Klf.precipitation -= Klf.precipitation.mean()
        #Klf.q_ref = np.array([1E-6,1E-9,100.,6.726449064255413e-07,-5.,1E-3,-1E-3,30.])
        #Klf.q_ref = np.array([1E-4,4E-8,100.,6.726449064255413e-07,-5.,1E-3,-1E-3,30.])
        Klf.q_ref = np.array([1E-5,4E-9,100.,6.726449064255413e-07,-5.,1E-3,-1E-3,30.])
        
    # The initial guess for the explanatory variable of the Kumamoto earthquake.
    # We assumed that an exponential type response.
    #
    # t_kmt: Date of the Kumamoto earthquake
    # at: the intial estimation of seismic velocity changes without the explanatory variables
    # τ: The decay time of the repsonse
    # mask_param: Data mask
    def cal_kumamoto(t_kmt, at, τ, mask_param): 
        dv = at[:,1].copy() 
        t0 = len(dv[:])
        tx = np.arange(0,t0,1)
        if len((dv[t_kmt-100:t_kmt-1])[mask_param[t_kmt-100:t_kmt-1]]) > 0:
            y_ref = (dv[t_kmt-100:t_kmt-1])[mask_param[t_kmt-100:t_kmt-1]].mean()
        else:
            y_ref = 0
        ye = np.zeros(t0) + y_ref

        cross = 0.
        ms = 0.
        for i in range(t_kmt,t0):
            η = np.exp(-(tx[i]-t_kmt)/τ)
            cross += (η*(dv[i]-y_ref))
            ms += (η*η)
        A = cross/ms
        #print(A,y_ref)
        ye[t_kmt:] = A*np.exp(-(tx[t_kmt:]-t_kmt)/τ) + y_ref
        res = ((ye[t_kmt:t0]-dv[t_kmt:t0])**2).mean()
        return res,A,y_ref,ye
    
    # The initial guess for the explanatory variable of the ground water level.
    # We assumed that an exponential type response.
    #
    # at: the intial estimation of seismic velocity changes without the explanatory variables
    # τ: The decay time of the repsonse
    # mask_param: Data mask
    def cal_gwl(at, τ, mask_param):
        T0 = Klf.T(np.arange(len(Klf.precipitation)),τ) #np.exp(-np.arange(len(Klf.precipitation))/τ)
        β = signal.convolve(Klf.precipitation, T0)[Klf.num0:len(Klf.precipitation)]
        ave = at[:,1].copy() #/at[:,0]*1E2
        ave[np.isnan(ave)] = 0
        β = β-β.mean()
        number = min(len(ave),len(β))
        u0 = signal.detrend(ave[0:number]) 
        u1 = signal.detrend(β[0:number]) 
        u0[np.logical_not(mask_param[0:number])]=0
        u1[np.logical_not(mask_param[0:number])]=0
        ccc = u0 @ u1

        γ = ccc / (u1@u1)
        res = -2*ccc*γ + γ*γ*u1@u1 + u0@u0
        #β = (gwl[num:number+num]).copy()
    
        β = β-β.mean()
        #print(len(ccf_gwl[number-1-int(τ/2):number-1+int(τ/2)]))
        number = min(len(ave),len(β))
        ave = ave[0:number]
        amp = (np.max(β)- np.min(β))/2
        β = β * γ
        return β, ave, res, amp, γ

    # Estimation of a prior data covariance h0.  h_0 id estimated from the time average 
    # of the squared difference between observed CCFs and the reference (see the text). 
    # 
    # [Input] ccfs1: observed CCFs
    #         ccf_r: the reference CCF 
    # [Output] a prior data covariance h0
    def est_h0(ccfs1,ccf_r,ts,te):
        mask0 = [False]*1024
        for i in range(ts,te): mask0[i+512] , mask0[512-i] = True, True
        h0 = 0.
        for row in range(3):
            for col in range(3):
                ccfs = ccfs1[row][col]
                #Calc. of reference the CCF
                mccfs = np.ma.masked_array(ccfs,np.isnan(ccfs))
                ccf_r[row][col] = mccfs.mean(axis=0)
                scale = ccfs[:,mask0].dot(ccf_r[row][col][mask0])/ccf_r[row][col][mask0].dot(ccf_r[row][col][mask0])
                mask = (scale>0.5) & (scale <5.)
                h0 += np.mean(((mccfs[mask])[:,mask0]- ccf_r[row][col][mask0])**2)
                mccfs[mask] = np.diag(1./scale[mask]) @  np.array(mccfs[mask])
                ccf_r[row][col] = mccfs[mask].mean(axis=0)
                #h0 += np.mean(((mccfs[mask])[:,mask0]- ccf_r[row][col][mask0])**2)
        h0 /= 9.
        return(h0)
   
    # An implementation of the extended Kalman filter and smoother
    #
    # [Input] ccfs_all: observed CCFs
    #         ccf_r: the reference CCF
    #         delta: the sampling interval [s]
    #         ts: the start of lag time [index number]
    #         te: the end of the lag time [index number]
    #         β: the temporal changes estimated by the explanatory variables
    #         h0: a prior data covariance
    #         Q0: a prior model covariance
    #         Pt: A priori model covariance for the initial value
    #         at0: A priori initial amplitude
    # [Output] αt: Estimated conditional mean values
    #          Vt: Estimated model covariance 
    #          mask_param: the mask of estimated values
    def eKlf(ccfs_all,ccf_r,delta,ts,te,β,h0,Q0,Pt=np.array([[1E-5,0.],[0.,1E-8]]),at0=np.array([1.,0.])):
        mask0 = [False]*1024
        for i in range(ts,te): mask0[i+512] , mask0[512-i] = True, True
        #h0 = 1.5**2 #0.08
        #Ht = h0*np.identity((te-ts)*2)
        lag_time = ((np.arange(1024)-512)*delta)[mask0]
        
        lnL = 0
        #mask_param = [True]*(len(ccfs_all[0][0])+1)
        att = np.zeros([len(ccfs_all[0][0]),2])
        αt = np.zeros([len(ccfs_all[0][0]),2])
        Ptt = np.zeros([len(ccfs_all[0][0]),2,2])
        Vt = np.zeros([len(ccfs_all[0][0]),2,2])
        Qt = np.zeros([2,2])
        mask_param = np.full([len(ccfs_all[0][0]),3,3], False, dtype=np.bool) 
            
        itime = 0

        at = at0.copy() #Initial value
        Qt = np.array([[Q0[0],0],[0,Q0[1]]]) #Pt*ϵ #[[1E-6,0],[0,1E-9]] #Initial value
        Tayler_Series = np.zeros([3,3,5,(te-ts)*2]) # Up to 5th order
        ms_r = np.zeros([3,3])

        for cmp1 in range(3):
            for cmp2 in range(3):
                Tayler_Series[cmp1][cmp2] = [Klf.deri(ccf_r[cmp1][cmp2],i,delta)[mask0] for i in range(5)]
                ms_r[cmp1][cmp2] = np.sum((ccf_r[cmp1][cmp2][mask0]**2))
                #Zt = np.array(ccf_r[cmp1][cmp2].T,Klf.deri(ccf_r[cmp1][cmp2],1,delta)*lag_time]).T])
        for iday in range(ccfs_all.shape[2]):
        #for y_tmp in ccfs_all:
            yt = ccfs_all[:,:,iday,mask0]
            Z2 = np.zeros([2,2])
            γ = np.zeros(2)
            v2 = 0.
            count = 0
            for cmp1 in range(3):
                for cmp2 in range(3):
                    Σ = np.zeros([3,(te-ts)*2])
                    scale = yt[cmp1][cmp2].dot(ccf_r[cmp1][cmp2][mask0])/ms_r[cmp1][cmp2]
                    Z0, Z1  = np.zeros((te-ts)*2), np.zeros((te-ts)*2)
                    A, α = at[0], at[1]#/at[0]
                    if scale > 0.5 and scale < 2:
                        for i in range(3):
                            Σ[i] = ((α+β[itime])*lag_time)**i
                        for i in range(3):
                            Z0 += (Tayler_Series[cmp1][cmp2][i]/math.factorial(i)*Σ[i])
                        Z1 = A*(Tayler_Series[cmp1][cmp2][1])
                        for i in range(1,3):
                            Z1 += A*(Tayler_Series[cmp1][cmp2][i+1]/math.factorial(i)*Σ[i])
                            Z1 += A*(Tayler_Series[cmp1][cmp2][i]/math.factorial(i-1)*Σ[i-1])
                        Z1 *= lag_time
                        Zt = np.array([Z0.T,Z1.T]).T
                        ###
                        Z2 += Zt.T @ Zt
                        vt = yt[cmp1][cmp2] - A*Z0 #Zt @ at[itime][cmp1][cmp2] #Reduction of β
                        γ += Zt.T @ vt
                        v2 += vt.T @ vt
                        mask_param[itime][cmp1][cmp2] = True
                        count += 1
                    #else: print(iday,A,α,scale)
            Ξ = Pt/h0 - (Pt @ Z2 @ np.linalg.inv(Z2/h0 +np.linalg.inv(Pt)) /(h0**2))
            #Likelihood
            λ = np.linalg.eig(Z2 @ Pt)[0]
            lnL1 = np.log(λ[0]+h0)+np.log(λ[1]+h0)+((te-ts)*2*9-2)*np.log(h0)
            lnL2 = h0*v2-γ.T @ (np.linalg.inv(Z2/h0 +np.linalg.inv(Pt))) @ γ
            #print("Likelihood ",lnL1,lnL2,lnL2*h0**-2)
            lnL += -(lnL1+lnL2*h0**-2)/2
            att[itime] = at + Ξ @ γ # at + Kt @ vt
            Ptt[itime] = Pt - Ξ @ (Z2 @ Pt @ Z2 + h0 * Z2) @ Ξ.T #Pt - Kt@Ft@Kt.T
            at[:] = att[itime]      #Tt=1
            Pt[:] = Ptt[itime] + Qt    #Tt and Rt = 1 
            itime += 1
        lnL -= (te-ts)*2*count/2.*np.log(2*np.pi)
        #Kalman Smoother
        itmax = itime-1
        αt[itmax] = att[itmax]
        Vt[itmax] = Ptt[itmax]
        for itime in reversed(range(itmax)):
            Pt1 = Ptt[itime]+Qt
            At = Ptt[itime]*np.linalg.inv(Pt1)
            αt[itime] = att[itime]+ At @ (αt[itime+1]-att[itime+1])
            Vt[itime] = Ptt[itime]+ At @ (Vt[itime+1]-Pt1) @ At.T
                
        return αt, Vt, lnL, mask_param

    # Calclation of the reference CCF
    # [Input] ccfs_all: Observed CCFs
    #         at: The initial guess of the stretching factor
    #         delra: sampling interval [s]
    # [Output] ccf_r2: The reference CCF
    def cal_ref(ccfs_all,at,delta):
        ccf_r2 = np.zeros([ccfs_all.shape[0],ccfs_all.shape[1],ccfs_all.shape[3]])
        lag_time = (np.arange(1024)-512)*delta

        for itime in range(ccfs_all.shape[2]):
            α0 = 0
            α0 = at[itime][1]#/at[itime][0]
            for cmp1 in range(3):
                for cmp2 in range(3):
                    y_tmp = ccfs_all[cmp1][cmp2][itime]
                    lag_time2 = (np.arange(1024)-512)*delta*(1+α0)
                    f2 = interpolate.interp1d(lag_time, y_tmp, kind="quadratic", fill_value='extrapolate')
                    ccf_r2[cmp1][cmp2] += f2(lag_time2)
        for cmp1 in range(3):
            for cmp2 in range(3):
                ccf_r2[cmp1][cmp2]  /= ccfs_all.shape[2]
        return ccf_r2
    
    # Evaluation of the response function for precipitation data
    # [Input] t: Days
    #         τ: Decay time [days]
    #         flag: The response type
    # [Output] Res: The response
    def T(t,τ,flag = "Exp"):
        if flag == "Talwani": #Talwani
            Res = (t+1E-4)**-1.5*np.exp(-1.5*τ/(t+1E-4))
            Res *= np.sqrt(3*τ/2/np.pi)*1E3*9.8*1E-3 #ρ=1E3, g=9.8
        elif flag == "Exp": #Exponential
            Res = np.exp(-t/τ)
        return Res

    # Calculation of derivative of data using FFT
    # [Input] data: an array
    #         n: the order of the derivative
    #         delta: sampling interval
    # [Output] data: the derivative
    def deri(data,n,delta):
        if n > 0:
            dω = 1./(len(data)*delta)*np.pi*2.
            spctrm = np.fft.rfft(data)
            for i in range(len(spctrm)): spctrm[i] = spctrm[i]*(1j*(i)*dω)**n 
            return np.fft.irfft(spctrm)
        elif n == 0: return(data)
        
        
    # Estimation of the precipitation effects by a simple data fitting
    # [Input] at: Initial guess of seismic velocity changes
    #         τ: The decay time of the response
    #         p: the power, which determines the nonlinearity. If p = 1, we neglect the nonlinearity.
    #         thresh: Threshold of the precipitation
    # [Output] β: Estimated seismic velcity changes by the precipitation model
    #          ave: at*1E2 with exclusions of the outliers of at. This is just for the plot.
    #          res: Residual between the data and the model
    #          amp = (max(β)- min(β))/2
    #          γ: Amplitudes of the response
    #          idx0: Delay time
    def cal_gwl_LSQ(at,τ,p,thresh,mask_param):
        with open("./gwl_ebino_seikei.dat","r") as f:
            lines = f.readlines()
        gwl = np.array([float(line.split()[4]) for line in lines])

        T0 =Klf.T(np.arange(len(gwl)),τ)
        T0[0]=0

        for idx in range(len(gwl)):
            if gwl[idx] > thresh : gwl[idx] = thresh
        gwl = signal.convolve(gwl**p, T0)[:len(gwl)]

        for i in range(len(lines)):
            if int(lines[i].split()[5]) == 1: num = i
                
        β = (gwl[num:]).copy()
        ave = at[:,1]*1E2 #/at[:,0]*1E2
        ave[np.isnan(ave)] = 0
        β = β-β.mean()
        number = min(len(ave),len(β))
        u0 = signal.detrend(ave[0:number]) 
        u1 = signal.detrend(β[0:number]) 
        u0[np.logical_not(mask_param[0:number])]=0
        
        ccf_gwl = signal.correlate(u0,u1)
        #idx1 = np.argmin(ccf_gwl)-(number-1)
        Δτ = int(τ/2)
        idx0 = np.argmin(ccf_gwl[number-1-Δτ:number-1+Δτ])-Δτ
        γ = np.min(ccf_gwl[number-1-Δτ:number-1+Δτ])/u1[mask_param[0:number]].dot(u1[mask_param[0:number]])
        res = -2*np.min(ccf_gwl[number-1-Δτ:number-1+Δτ])*γ + γ*γ*u1[mask_param[0:number]].dot(u1[mask_param[0:number]]) + u0[mask_param[0:number]].dot(u0[mask_param[0:number]])
        #print("##",idx0)
        β = (gwl[num-idx0:number+num-idx0]).copy()
        #print(len(β),num+idx0,number+num+idx0,len(gwl))
        β = β-β.mean()
        #print(len(ccf_gwl[number-1-int(τ/2):number-1+int(τ/2)]))
        number = min(len(ave),len(β))
        ave = ave[0:number]
        amp = (np.max(β)- np.min(β))/2
        β = β * γ
        return β, ave, res, amp, γ*1E-2,idx0

