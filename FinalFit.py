from RAcalcforlmfit import Rp, Rs, p_residual, RAp_eff, RAs_eff, t_residual
from lorentzfit import params, background
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import LorentzianModel
from lmfit import Parameters, minimize
from waterprep import Water_inter, Water_inter_single, Water_R0s, Water_R0p

#plotting style
#matplotlib style
plt.rcParams.update({'font.size': 14}) #gen text size
plt.rc('xtick', labelsize=12) #axes font
plt.rc('ytick', labelsize=12)

#Flach data
#fig 3 data
p35 = np.loadtxt('./Data/fig3_H20/p35.csv', float, delimiter=',') 
p40 = np.loadtxt('./Data/fig3_H20/p40.csv', float, delimiter=',') 
p45 = np.loadtxt('./Data/fig3_H20/p45.csv', float, delimiter=',') 
p50 = np.loadtxt('./Data/fig3_H20/p50.csv', float, delimiter=',') 


#fig 2 data
s35 = np.loadtxt('./Data/fig2_H2O/s35.csv', float, delimiter=',') 
s40 = np.loadtxt('./Data/fig2_H2O/s40.csv', float, delimiter=',') 
s45 = np.loadtxt('./Data/fig2_H2O/s45.csv', float, delimiter=',') 
s50 = np.loadtxt('./Data/fig2_H2O/s50.csv', float, delimiter=',') 


#angles
datasets = [p35, p40, p45, p50, s35, s40, s45, s50]
angles_deg = [35, 40, 45, 50]
angles = []


##### test #####
for a in angles_deg:
    a_rad = a/180 * np.pi
    angles.append(a_rad)
##### test #####


##measured
n =  1.41 #ellipsometry 
d = 3.06 * 1e-7 #cm ; ellipsometry or molecular geom
alpha_deg = 90 #in degrees; guess or indp var (orientation)
alpha = alpha_deg/180 * np.pi

#decide on relevant range
v_max = 2950
v_min = 2890
points = 300

fwhh = []
v0 = []
offset = []

real_peaks = []
fit_peaks = []
i = 0
#fit for k(v)
for a in datasets:
    #get right data to fit
    mask = (a[:,0] < v_max) & (a[:,0] > v_min)
    rA_data = a[mask,1] * 1e-3
    v = a[mask,0]
    #guess init values
    guess = [2915, 6, -0.6, 0.04] #center, sigma, amplitude, offset
    
    #first fit
    fwhh_val, v0_val, offset_val, amplitude = params(guess, v, rA_data)
    fwhh.append(fwhh_val)
    v0.append(v0_val)
    offset.append(offset_val)

    #fit of fit to get baseline - will make it bigger and bigger at each step
    v_min -= 100
    v_max += 100
    v = np.linspace(v_min, v_max, points)
    fitband = LorentzianModel().eval(x=v, center=v0_val, sigma=fwhh_val/2, amplitude=amplitude)
    bkg, peak = background(guess, v, fitband)
    
    v_min += 100
    v_max -= 100
    #print(f' fwhh: {fwhh_val}, v0: {v0_val}, offset: {offset_val}')
    fit_peaks.append(np.min(fitband- bkg))
    real_peaks.append(np.min(rA_data) - np.max(rA_data))


#one fwhh and v0 for all angles
fwhh = np.median(fwhh)
v0 = np.median(v0)


#water data
cn2 = Water_inter_single(v0)

kws_fixed = {
    "alpha": alpha, #in radians
    "nmax": n,
    "d": d, #cm ; ellipsometry or molecular geom
    "fwhh": fwhh,
    "v0": v0,
    'cn2': cn2,
}

#set params
##init guess params
kmax_0 = 1.00 #1.07?
Gamma_pol_0 = 0.014 #0.014
tilt_0 = 0.0 #information about orientation; in RADIANS!!!

#pass pars on to lmfit
pars = Parameters()

pars.add('Gamma_pol', value=Gamma_pol_0, min=0, max=0.05, vary=False)
pars.add('tilt', value=tilt_0, min=-np.pi/2, max=np.pi/2, vary=True)
pars.add('kmax', value=kmax_0, min=1, max=2, vary=True)

#make data dataset
theta1 = np.array(angles)
yvals = np.array(fit_peaks)


result = minimize(t_residual, pars, args=(theta1,), kws={"data": yvals, **kws_fixed}, method='leastsq', ftol=1e-12, xtol=1e-15)
print(result.redchi)

fit_params = result.params

rA_fit = t_residual(fit_params, theta1,None, **kws_fixed)

print(fit_params)
#print(rA_fit)
#print(yvals)


#plot fits
#fig 10

tilt = fit_params['tilt'].value
kmax = fit_params['kmax'].value
Gamma_pol = fit_params['Gamma_pol'].value


v_max = 2950
v_min = 2890
points = 300
v_min -= 50
v_max += 50
v, cn2 = Water_inter(vmax=v_max, vmin=v_min, num=300)

cont_angles = np.linspace(30, 52, 100)
peaks = []
peaks_s = []

for a in cont_angles:
    a_rad = a/180 * np.pi
    rp_fit = Rp(a_rad, cn2, d,  n, v,alpha, fwhh, v0, tilt, kmax)
    rs_fit = Rs(a_rad, cn2, d, n, v,alpha, fwhh, v0, tilt, kmax)

    #water reflection
    R0p = Water_R0p(a_rad, cn2)
    R0s = Water_R0s(a_rad, cn2)

    rAp_eff = RAp_eff(rp_fit, rs_fit, R0p, R0s, Gamma_pol) # simulation

    #baseline code 
    bkg, peak  = background(guess, v, rAp_eff)
    y_corrected = rAp_eff - bkg

    #s pol
    rAs_eff = RAs_eff(rp_fit, rs_fit, R0p, R0s, Gamma_pol) # simulation
    bkg2, peak_s  = background(guess, v, rAs_eff)
    y_corrected_s = rAs_eff - bkg2

    #plt.plot(v, rAp_eff.real - np.max(rAp_eff.real), label=f'{a} deg') #account for baseline offset? - np.max(rAp_eff.real)
    #plt.plot(v, y_corrected.real, label=f'{a} deg (baseline corrected)')

    peak = np.min(y_corrected.real)
    peak_s = np.min(y_corrected_s.real)
    peaks.append(peak)
    peaks_s.append(peak_s)

manual_res = t_residual(pars, theta1, data=yvals, **kws_fixed)
fit_res = t_residual(result.params, theta1, data=yvals, **kws_fixed)
print(t_residual(pars, theta1, data=yvals, **kws_fixed))
print(result.message)  # human readable convergence info
print(result.nfev)     # number of function evaluations
print("Fit residuals stats:", np.mean(fit_res), np.std(fit_res))
print("Manual residuals stats:", np.mean(manual_res), np.std(manual_res))
#my data
#plt.scatter(angles, fit_peaks, color='red', label='My peaks')


##### test #####
cont_angles /= 180
cont_angles *= np.pi
##### test #####


plt.plot(cont_angles, peaks, label = "p-pol", linewidth=1.5)
plt.plot(cont_angles, peaks_s, label = "s-pol", linewidth=1.5)

plt.plot(theta1, yvals[0:4], 'o', label='_p-pol', color='blue')
plt.plot(theta1, yvals[4:9], 'o', label='_s-pol', color='orange')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) #scientific notation
plt.ylim([-0.07,0.0])
plt.xlabel('Angle (rad)')
plt.ylabel('Peak R-A')
plt.title('Simulated Peak R-A for -CH2- Asym. Stretch')
plt.legend()
#plt.plot(theta1, rA_fit, label='fit')
#plt.savefig('./plots/GRC26_sim_poster.png', dpi=300)
plt.show()

'''
next steps: 
- still looks like a fit problem.... 
- treat s and p sperately for the first Lorentzian fit
'''