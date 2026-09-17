from RAcalc import Rs, Rp, RAp_eff
from lorentzfit import params, background
from waterprep import Water_inter, Water_R0s, Water_R0p
import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import LorentzianModel

#read data
p35 = np.loadtxt('./Data/ch2_stretch_p_35.txt', float, skiprows=3) 
p40 = np.loadtxt('./Data/ch2_stretch_p_40.txt', float, skiprows=3) 
p45 = np.loadtxt('./Data/ch2_stretch_p_45.txt', float, skiprows=3) 
p50 = np.loadtxt('./Data/fig3_H20/p50.csv', float, delimiter=',') 

#angles
datasets = [p35, p40, p45, p50]
angles = [35, 40, 45, 50]

#set params

##would be fit
kmax = 1.07
Gamma_pol = 0.014 
tilt = 0 #information about orientation; in RADIANS!!!

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


#demonstration for one angle 
mask = (p50[:,0] < v_max) & (p50[:,0] > v_min)
rA_data = p50[mask,1] * 1e-3
v = p50[mask,0]
#guess init values
guess = [2915, 6, -0.6, 0.04] #center, sigma, amplitude, offset

#first fit
fwhh_val, v0_val, offset_val, amplitude = params(guess, v, rA_data)
fwhh.append(fwhh_val)
v0.append(v0_val)
offset.append(offset_val)

#fit of fit to get baseline
v_min -= 10
v_max += 10
v = np.linspace(v_min, v_max, points)
fitband = LorentzianModel().eval(x=v, center=v0_val, sigma=fwhh_val/2, amplitude=amplitude)
bkg, peak = background(guess, v, fitband)

#print(f' fwhh: {fwhh_val}, v0: {v0_val}, offset: {offset_val}')
fit_peaks.append(np.min(fitband- bkg))
real_peaks.append(np.min(rA_data) - np.max(rA_data))

print(peak.fit_report())

#plt.plot(v, rA_data - offset_val, 'o', label=f'{a} deg')
plt.plot(p50[:,0], p50[:,1] * 1e-3 - offset_val, '-', label='data', alpha = 0.5)
plt.plot(v, fitband - bkg, "--", label='fit')
plt.legend()
plt.title('Lorentzian Fit of k(v) for 50 deg')
plt.xlabel('wavenumber (cm$^{-1}$)')
plt.ylabel('reflection-absorption')

#plt.plot(v, peak, label=f'{angles[i]} deg peak')
plt.show()



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

    #fit of fit to get baseline
    v_min -= 10
    v_max += 10
    v = np.linspace(v_min, v_max, points)
    fitband = LorentzianModel().eval(x=v, center=v0_val, sigma=fwhh_val/2, amplitude=amplitude)
    bkg, peak = background(guess, v, fitband)
    
    #print(f' fwhh: {fwhh_val}, v0: {v0_val}, offset: {offset_val}')
    fit_peaks.append(np.min(fitband- bkg))
    real_peaks.append(np.min(rA_data) - np.max(rA_data))

    #plt.plot(v, rA_data - offset_val, 'o', label=f'{a} deg')
    plt.plot(v, fitband - bkg, "--", label=f'{angles[i]} deg fit')
    plt.plot(v, peak, label=f'{angles[i]} deg peak')
    i += 1

print(fit_peaks)
print(real_peaks)
plt.legend()
plt.show()

#one fwhh and v0 for all angles
fwhh = np.median(fwhh)
v0 = np.median(v0)
