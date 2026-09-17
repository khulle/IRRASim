from flach_v2 import RAp, RAs, k_params, RAp_eff, cn2
import numpy as np
import matplotlib.pyplot as plt

#params
#constants 
##air
n1 = 1 #real refractive index
k1 = 0 #complex

##random constants
c = 299792458 * 1e2 #speed of light in cm/s

##experiment defined constants
theta1_deg = np.linspace(30, 52, 100) #angle of incidence
theta1 = theta1_deg/180 * np.pi

###Flach figure 10: asymmetric CH2 mode of behenic acid methyl ester - 2926cm^-1
###note symm is the lower wavenumber one
###fitted/guessed from multiple angles - using what they state in fig 10
kmax = 1.07
Gamma_pol = 0.014 
tilt = 0 #information about orientation; in RADIANS!!!

###measureable
n =  1.41 #ellipsometry 
d = 3.06 * 1e-7 #cm ; ellipsometry or molecular geom
alpha_deg = 90 #in degrees; guess or indp var (orientation)
alpha = alpha_deg/180 * np.pi

###calculated constants
####anisotropic constants
f = (3 * (np.cos(tilt))**2 - 1)/2
kxmax = (f * (np.sin(alpha)**2)/2 + (1-f)/3)* kmax
kzmax = (f * (np.cos(alpha)**2)/2 + (1-f)/3)* kmax

#import params
fwhh, v0 = k_params()
offset = 0

#get v range (should match interpolated range)
v_max = 2950
v_min = 2890
v = np.linspace(v_min, v_max, 300)

#simulate it!
min_abs = []
for a in theta1:
    rAp_fit = RAp(a, cn2, d, kzmax, kxmax, n, v,fwhh, v0, offset)
    rAs_fit = RAs(a, cn2, d, kzmax, kxmax, n, v,fwhh, v0, offset)
    rAp_eff = RAp_eff(rAp_fit, rAs_fit, Gamma_pol) # simulation
    min_abs.append(min(rAp_eff.real)) #store min abs (peak height)
#plots

plt.plot(theta1_deg, min_abs, label='simulated RAp eff')
plt.xlabel('Angle of Incidence (degrees)')
plt.ylabel('RA (arb. units)')
plt.title(f'Flach Model Simulation CH$_2$ Asymmetric Stretch')
plt.legend()
plt.show()


