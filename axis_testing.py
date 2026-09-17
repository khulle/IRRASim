from RAcalc import Rs, Rp, RAp_eff
from lorentzfit import params, background
from waterprep import Water_inter, Water_R0s, Water_R0p
import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import LorentzianModel

#Flach data
#fig 3 data
p35 = np.loadtxt('./Data/ch2_stretch_p_35.txt', float, skiprows=3) 
p40 = np.loadtxt('./Data/ch2_stretch_p_40.txt', float, skiprows=3) 
p45 = np.loadtxt('./Data/ch2_stretch_p_45.txt', float, skiprows=3) 
p50 = np.loadtxt('./Data/ch2_stretch_p_50.txt', float, skiprows=3) 

p35_new = np.loadtxt('./Data/fig3_H20/p35.csv', float, delimiter=',')
p40_new = np.loadtxt('./Data/fig3_H20/p40.csv', float, delimiter=',')
p45_new = np.loadtxt('./Data/fig3_H20/p45.csv', float, delimiter=',')
p50_new = np.loadtxt('./Data/fig3_H20/p50.csv', float, delimiter=',')

#fig 10 data
fig10_simulation = np.loadtxt('./Data/simulation.csv', float, delimiter=',')
fig10_data = np.loadtxt('./Data/datapoints.csv', float, delimiter=',')

#angles
datasets = [p35_new, p40_new, p45_new, p50_new]
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

    #plt.plot(a[mask,0], rA_data - offset_val, '-', label=f'{angles[i]} deg data')
    plt.plot(v, fitband - bkg, '-')
    plt.scatter(v0[i], fig10_data[i,1])
    
    i += 1
    
#plt.scatter(v0[0:4], fig10_data[:,1])
#plt.xlim([2890, 2950])


print(fit_peaks)
print(real_peaks)
plt.show()

#one fwhh and v0 for all angles
fwhh = np.median(fwhh)
v0 = np.median(v0)


'''
Problems/improvements:
-try reselecting data? some artifacts in lorentzian fit (35,40,50)

- maybe add second peak first? there is some necessary convolution around 2890

-changed comp of arcsin to be more numerically stable (supposedly)
    - switch to sin/cos formula? 
    - what do i expecpt R0p to look like? 

- try with water data they used rather than Hale
'''