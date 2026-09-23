import numpy as np
import importlib.resources as resources

#old data source - not used
##refractiveindex.info
###tG. M. Hale and M. R. Querry. Optical constants of water in the 200-nm to 200-µm wavelength region. Appl. Opt. 12, 555-563 (1973)
#wl_n2 = np.loadtxt('./Bertie_1989/Hale_n.txt', float) # make sure you are in Flach folder
#wl_k2 = np.loadtxt('./Hale_1973/Hale_k.txt', float)
#wn_all = 1e4/wl_n2[:,0] #convert from wavelength to wavenumber
#cn2_all = wl_n2[:,1] + 1j * wl_k2[:,1] #combine real and complex parts to get complex refractive index

## ref 28 in Flach
## Bertie, J.E. & M. Khalique Ahmed J. Phys. Chem. 1989, 93, 2210-2218
##digitzed with adobe OCR + excel
water_data = np.loadtxt(resources.files("irrasim.data") / "Bertie_9000_1250.txt", float, skiprows=1) #wavenumber 1/cm, k, n 

wn_all = water_data[:,0]
cn2_all = water_data[:,2] + 1j * water_data[:,1]


def Water_inter(vmax,vmin, num): #vmax, vmin, number of points; should be in wavenumbers (cm^-1)
    v = np.linspace(vmin, vmax, num) #evenly spaced wavenumber axis for fitting/plotting/maths
    #water sorting
    idx = np.argsort(wn_all)
    wn_sorted = wn_all[idx]
    cn2_sorted = cn2_all[idx]

    #interpolate data
    cn2 = np.interp(v, wn_sorted, cn2_sorted)
    return v, cn2

def Water_inter_data(data): #vmax, vmin, number of points; should be in wavenumbers (cm^-1)
    v = data[:,0]
    #water sorting
    idx = np.argsort(wn_all)
    wn_sorted = wn_all[idx]
    cn2_sorted = cn2_all[idx]

    #interpolate data
    cn2 = np.interp(v, wn_sorted, cn2_sorted)
    return v, cn2

def Water_inter_single(v):
    #water sorting
    idx = np.argsort(wn_all)
    wn_sorted = wn_all[idx]
    cn2_sorted = cn2_all[idx]

    #interpolate data
    cn2 = np.interp(v, wn_sorted, cn2_sorted)
    return cn2

def Theta2(theta1, cn2):
    sin_theta2 = (1 / cn2) * np.sin(theta1)
    cos_theta2 = np.sqrt(1 - sin_theta2**2)

    if np.isscalar(cn2):
        cn2 = np.full_like(theta1, cn2, dtype = complex)

    for i in range(len(cn2)):
        if np.imag(cn2[i] * cos_theta2[i]) < 0:
            cos_theta2[i] = -cos_theta2[i]

    ctheta2 = np.arccos(cos_theta2)
    return ctheta2

def Water_R0s(theta1, cn2):
    #water surface ("Fresnel")
    ctheta2 = Theta2(theta1, cn2)
    r0s = np.sin(theta1 - ctheta2) / np.sin(theta1 + ctheta2) 
    R0s = abs(r0s)**2 #actual reflectivity from pure water surface
    return R0s

def Water_R0p(theta1, cn2): #may be picking wrong root?
    #water surface ("Fresnel")
    #ctheta2 = np.arcsin(np.sin(theta1) / (cn2)) #c bc this is a complex value
    ctheta2 = Theta2(theta1, cn2)

    r0p = np.tan(theta1 - ctheta2) / np.tan(theta1 + ctheta2)
    R0p = abs(r0p)**2
    return R0p
