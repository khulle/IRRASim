#libraries
import numpy as np
from math import pi
from waterprep import Theta2

#constants 
##air
n1 = 1 #real refractive index
k1 = 0 #complex

##random constants
c = 299792458 * 1e2 #speed of light in cm/s 


#actual RA 
def Rs(theta1, cn2, d, nmax, v,alpha, fwhh, v0, tilt, kmax):
    ####anisotropic constants
    f = (3 * (np.cos(tilt))**2 - 1)/2
    kxmax = (f * (np.sin(alpha)**2)/2 + (1-f)/3)* kmax
    kzmax = (f * (np.cos(alpha)**2) + (1-f)/3)* kmax


    #water surface ("Fresnel")
    #ctheta2 = np.arcsin(np.sin(theta1) / (cn2)) 
    ctheta2 = Theta2(theta1, cn2)


    Delta = 2 * pi * c * (v - v0) #is v supposed to be wavenumbers?
    gamma = 2 * pi * c * fwhh 
    
    #thin-film refractive index - Lorentzian

    #Lorentzian
    kx = kxmax* gamma**2 /(4 * Delta**2 + gamma**2) #k(v) 
    kz = kzmax* gamma**2 /(4 * Delta**2 + gamma**2) #k(v) 
    nx = nmax - 2 * Delta * kxmax * gamma /(4 * Delta**2 + gamma**2)
    nz = nmax - 2 * Delta * kzmax * gamma /(4 * Delta**2 + gamma**2)
    cnx = nx + 1j * kx

    I1 = (cnx**2 - cn2**2)*d

    k0 = 2 * pi / (1/v) #walength in cm

    #calc rs
    A = np.sin(theta1 - ctheta2) 
    B = -1j * k0 / cn2 * np.sin(theta1) * I1
    C = np.sin(theta1 + ctheta2)
    D = -1j * k0 / cn2.real * np.sin(theta1) * I1

    rs = -(A+B)/(C+D)

    #calc RA!
    Rs = (rs * np.conj(rs)).real
    #return -1 * np.log10(Rs/R0s) + offset
    return Rs

def Rp(theta1, cn2, d, nmax, v,alpha, fwhh, v0, tilt, kmax):
    ####anisotropic constants
    f = (3 * (np.cos(tilt))**2 - 1)/2
    kxmax = (f * (np.sin(alpha)**2)/2 + (1-f)/3)* kmax
    kzmax = (f * (np.cos(alpha)**2) + (1-f)/3)* kmax


    #water surface ("Fresnel")
    #ctheta2 = np.arcsin(np.sin(theta1) / (cn2)) #c bc this is a complex value
    ctheta2 = Theta2(theta1, cn2)


    Delta = 2 * pi * c * (v - v0) 
    gamma = 2 * pi * c * fwhh 
    
    #thin-film refractive index - Lorentzian

    #let's try editing s.t. Gaussian rather than Lorentzian
    kx = kxmax* gamma**2 /(4 * Delta**2 + gamma**2) #k(v) 
    kz = kzmax* gamma**2 /(4 * Delta**2 + gamma**2) #k(v) 
    nx = nmax - 2 * Delta * kxmax * gamma /(4 * Delta**2 + gamma**2)
    nz = nmax - 2 * Delta * kzmax * gamma /(4 * Delta**2 + gamma**2)
    cnx = nx + 1j * kx
    cnz = nz + 1j * kz

    I1 = (cnx**2 - cn2**2)*d
    I2 = (cnz**2 - cn2**2)*d / cnz**2

    k0 = 2 * pi / (1/v) #1/v = walength in cm

    #calc rp
    A = np.sin(theta1 - ctheta2) * np.cos(theta1 + ctheta2) 
    B = (- 1j * k0 / cn2) * np.sin(theta1) * (I1 * np.cos(theta1) * np.cos(ctheta2) - I2 * np.sin(theta1)* np.sin(ctheta2))
    C = np.sin(theta1 + ctheta2) * np.cos(theta1 - ctheta2)
    D =  (- 1j * k0 / cn2) * np.sin(theta1) * (I1 * np.cos(theta1) * np.cos(ctheta2) + I2 * np.sin(theta1)* np.sin(ctheta2))
    rp = (A+B)/(C+D)

    #calc RA!
    Rp = abs(rp)**2
    #return -1 * np.log10(Rp/R0p) + offset #note log10 not ln
    return Rp

def RAp_eff(Rp, Rs, R0p, R0s, Gamma):
    Rp_eff = Rp + Gamma * (Rs - Rp)
    R0p_eff = R0p + Gamma * (R0s - R0p) #water
    return -1 * np.log10(Rp_eff/R0p_eff)


