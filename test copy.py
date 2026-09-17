from RAcalc import Rs, Rp, RAp_eff
from lorentzfit import params, background
from waterprep import Water_inter, Water_R0s, Water_R0p
import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import LorentzianModel

#Flach data
#fig 3 data
p35 = np.loadtxt('./Data/fig3_H20/p35.csv', float, delimiter=',')
p40 = np.loadtxt('./Data/fig3_H20/p40.csv', float, delimiter=',')
p45 = np.loadtxt('./Data/fig3_H20/p45.csv', float, delimiter=',')
p50 = np.loadtxt('./Data/fig3_H20/p50.csv', float, delimiter=',')

#fig 10 data
fig10_simulation = np.loadtxt('./Data/simulation.csv', float, delimiter=',')
fig10_data = np.loadtxt('./Data/datapoints.csv', float, delimiter=',')

#angles
datasets = [p35, p40, p45, p50]
angles = [35, 40, 45, 50]

#set params

##would be fit
kmax = 1.07
Gamma_pol = 0.03
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

    plt.plot(v, fitband - bkg, '-')

print(fit_peaks)
print(real_peaks)
plt.show()

#one fwhh and v0 for all angles
#fwhh = np.median(fwhh)
#print(fwhh)

#to test effect of fwhm:
fwhh = 15

v0 = np.median(v0)

#get water data
v_min -= 50
v_max += 50
v, cn2 = Water_inter(vmax=v_max, vmin=v_min, num=300)

#irras simulation
for a in angles:
    a_rad = a/180 * np.pi
    #monolayer
    rp_fit = Rp(a_rad, cn2, d,  n, v,alpha, fwhh, v0, tilt, kmax) 
    rs_fit = Rs(a_rad, cn2, d, n, v,alpha, fwhh, v0, tilt, kmax)

    #water values
    R0p = Water_R0p(a_rad, cn2)
    R0s = Water_R0s(a_rad, cn2)

    #effective RA
    rAp_eff = RAp_eff(rp_fit, rs_fit, R0p, R0s, Gamma_pol) # simulation
    #rAp_eff -= np.max(rAp_eff) #but why is it so offset though?

    #baseline code 
    #knots_xvals = np.random.uniform(v_min, v0-fwhh/2, size=10) #5 random knots on left
    #knots_xvals = np.append(knots_xvals, np.random.uniform(v0+fwhh/2, v_max, size=10)) #5 random knots on right
    #mask = (v > v_min) | (v < v0-fwhh/2)

    bkg, peak  = background(guess, v, rAp_eff)
    #plt.plot(v, R0p, "--", label=f'{a} deg Rs')
    #plt.plot(v, rAp_eff.real - bkg, label=f'{a} deg') 
    #plt.plot(v, y_corrected.real, "-.",label=f'{a} deg')
    plt.plot(v, rAp_eff.real, label=f'{a} deg (no baseline correction)')
    #plt.plot(v, bkg, "--", label=f'{a} deg baseline')
    #plt.plot(v, peak, "-.", label=f'{a} deg peak')


#plt.legend()
plt.show()


#fig 10
cont_angles = np.linspace(30, 52, 100)
peaks = []

water_peaks = []
for a in cont_angles:
    a_rad = a/180 * np.pi
    rp_fit = Rp(a_rad, cn2, d,  n, v,alpha, fwhh, v0, tilt, kmax)
    rs_fit = Rs(a_rad, cn2, d, n, v,alpha, fwhh, v0, tilt, kmax)

    #water reflection
    R0p = Water_R0p(a_rad, cn2)
    R0s = Water_R0s(a_rad, cn2)
    water_peaks.append(np.max(R0p))

    rAp_eff = RAp_eff(rp_fit, rs_fit, R0p, R0s, Gamma_pol) # simulation

    #baseline code 
    bkg, peak  = background(guess, v, rAp_eff)
    y_corrected = rAp_eff - bkg

    #plt.plot(v, rAp_eff.real - np.max(rAp_eff.real), label=f'{a} deg') #account for baseline offset? - np.max(rAp_eff.real)
    #plt.plot(v, y_corrected.real, label=f'{a} deg (baseline corrected)')

    peak = np.min(y_corrected.real)
    peaks.append(peak)
    
#my data
plt.scatter(angles, fit_peaks, color='red', label="Kyle's peaks")
plt.plot(cont_angles, peaks, c = 'r', label = "Kyle's simulation")

#Flach data
plt.plot(fig10_simulation[:,0], fig10_simulation[:,1], c = 'k', label='Flach simulation', alpha = 0.7)
plt.scatter(fig10_data[:,0], fig10_data[:,1], color='k', label='Flach data')
plt.legend()
plt.title("Simulated and Measured Peak Abs")
plt.xlabel('Angle (degrees)')
plt.ylabel('Peak Absorption')
plt.ylim([-0.07,0.0])
plt.show()

'''
Problems/improvements:

- maybe add second peak first? there is some necessary convolution around 2890

'''