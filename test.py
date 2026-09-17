from RAcalc import Rs, Rp, RAp_eff
from lorentzfit import params
from waterprep import Water_inter, Water_R0s, Water_R0p
import matplotlib.pyplot as plt
import numpy as np


#read data
p35 = np.loadtxt('./Data/ch2_stretch_p_35.txt', float, skiprows=3) 
p40 = np.loadtxt('./Data/ch2_stretch_p_40.txt', float, skiprows=3) 
p45 = np.loadtxt('./Data/ch2_stretch_p_45.txt', float, skiprows=3) 
p50 = np.loadtxt('./Data/ch2_stretch_p_50.txt', float, skiprows=3) 

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

#fit for k(v)
for a in datasets:
    #get right data to fit
    mask = (a[:,0] < v_max) & (a[:,0] > v_min)
    rA_data = a[mask,1] * 1e-3
    v = a[mask,0]
    #guess init values
    guess = [2915, 6, -0.6, 0.04] #center, sigma, amplitude, offset
    #do the fitting
    fwhh_val, v0_val, offset_val = params(guess, v, rA_data)
    fwhh.append(fwhh_val)
    v0.append(v0_val)
    offset.append(offset_val)
    #print(f' fwhh: {fwhh_val}, v0: {v0_val}, offset: {offset_val}')
    real_peaks.append(np.min(rA_data) - np.max(rA_data))

#one fwhh and v0 for all angles
fwhh = np.median(fwhh)
v0 = np.median(v0)

#get water data
v, cn2 = Water_inter(vmax=v_max, vmin=v_min, num=300)

idx = 0
#irras simulation
for a in angles:
    a_rad = a/180 * np.pi
    #monolayer
    rp_fit = Rp(a_rad, cn2, d,  n, v,alpha, fwhh, v0, 0, tilt, kmax) #offset[idx]
    rs_fit = Rs(a_rad, cn2, d, n, v,alpha, fwhh, v0, 0, tilt, kmax)

    #water values
    R0p = Water_R0p(a_rad, cn2)

    #effective RA
    rAp_eff = RAp_eff(rp_fit, rs_fit, R0p, Gamma_pol) # simulation
    rAp_eff -= np.max(rAp_eff) #account for baseline offset? - np.max(rAp_eff)

    #baseline corrections
    mask = (v > v_min) | (v < v0-fwhh/2) #left
    coeffs = np.polyfit(v[mask], rAp_eff.real[mask], 1)  # degree 1 = linear
    left_baseline = np.polyval(coeffs, v)

    mask = (v < v_max) | (v > v0-fwhh/2) #right
    coeffs = np.polyfit(v[mask], rAp_eff.real[mask], 1)  # degree 1 = linear
    right_baseline = np.polyval(coeffs, v)
    # subtract baseline
    y_corrected = rAp_eff -  right_baseline

    #plt.plot(v, rAp_eff.real - np.max(rAp_eff.real), label=f'{a} deg') #account for baseline offset? - np.max(rAp_eff.real)
    plt.plot(v, y_corrected.real, label=f'{a} deg')
    idx += 1

plt.legend()
plt.show()


#fig 10
cont_angles = np.linspace(30, 52, 100)
peaks = []

for a in cont_angles:
    a_rad = a/180 * np.pi
    rp_fit = Rp(a_rad, cn2, d,  n, v,alpha, fwhh, v0, 0, tilt, kmax)
    rs_fit = Rs(a_rad, cn2, d, n, v,alpha, fwhh, v0, 0, tilt, kmax)

    #water reflection
    R0p = Water_R0p(a_rad, cn2)

    rAp_eff = RAp_eff(rp_fit, rs_fit, R0p, Gamma_pol) # simulation

    # choose baseline regions (e.g., edges)
    mask = (v > v_min) | (v < v0-fwhh/2)

    # fit line to baseline regions
    coeffs = np.polyfit(v[mask], rAp_eff.real[mask], 1)  # degree 1 = linear
    left_baseline = np.polyval(coeffs, v)
    # subtract baseline
    y_corrected = rAp_eff - left_baseline

    #plt.plot(v, rAp_eff.real - np.max(rAp_eff.real), label=f'{a} deg') #account for baseline offset? - np.max(rAp_eff.real)
    #plt.plot(v, y_corrected.real, label=f'{a} deg (baseline corrected)')

    peak = np.min(y_corrected.real) - np.max(y_corrected.real) 
    peaks.append(peak)

plt.plot(cont_angles, peaks)
plt.scatter(angles, real_peaks, color='red', label='real peaks')
plt.show()

'''
Problems/improvements:
- should use height at exact chosen v0
- deal with offset/lobes -> perhaps the issue? 
'''