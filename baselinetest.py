import numpy as np
from RAcalcforlmfit import Rp, Rs, RAp_eff, RAs_eff
import matplotlib.pyplot as plt
from waterprep import Water_inter, Water_inter_data, Water_R0s, Water_R0p
from scipy.optimize import curve_fit

#set k, tilt, and Gamma_pol to 0 for baseline test

#fit pars
kmax = 0
tilt = 0
Gamma_pol = 0.014 #from paper, prob diff in our setup but no effect

#experimental constants
theta1 = 60/180 * np.pi #radians

#constants
#set for an ideal transparent thin film
n = 1.43
d = 18.2 * 1e-8 #3.06 * 1e-7 cm
alpha_deg = 90 #in degrees;
alpha = alpha_deg/180 * np.pi

fwhh = 5
v0 = 2900

#full range
v, cn2 = Water_inter(vmax=4000, vmin=1000, num=10000)


rS = Rs(theta1, cn2, d=d, nmax=n, v=v, alpha=alpha, fwhh=fwhh, v0=v0, tilt=tilt, kmax=kmax)
rP = Rp(theta1, cn2, d=d, nmax=n, v=v, alpha=alpha, fwhh=fwhh, v0=v0, tilt=tilt, kmax=kmax)
r0s = Water_R0s(theta1, cn2)
r0p = Water_R0p(theta1, cn2)

rA_s = RAs_eff(rP, rS, r0p, r0s, Gamma_pol)
rA_p = RAp_eff(rP, rS, r0p, r0s, Gamma_pol)


######## comparison to SA data ########
#data = np.loadtxt("../ds-env/Data/IRRAS/20260814/SA-avg-40uL.dpt")
data = np.loadtxt("../ds-env/Data/IRRAS/20260914/260914SA-1-Av.dpt")
v, cn2 = Water_inter_data(data) #interpolate water data to match SA data


def RA_s_wrap(v, d):
    rS = Rs(theta1, cn2, d=d, nmax=n, v=v, alpha=alpha, fwhh=fwhh, v0=v0, tilt=tilt, kmax=kmax)
    rP = Rp(theta1, cn2, d=d, nmax=n, v=v, alpha=alpha, fwhh=fwhh, v0=v0, tilt=tilt, kmax=kmax)
    r0s = Water_R0s(theta1, cn2)
    r0p = Water_R0p(theta1, cn2)

    rA_s = RAs_eff(rP, rS, r0p, r0s, Gamma_pol)
    return rA_s


popt, pcov = curve_fit(RA_s_wrap, data[:,0], data[:,1], p0=[d], bounds=(1e-10, 1e-6))
print("Fitted d:", popt[0], "+/-", np.sqrt(np.diag(pcov))[0])
d_fit = popt[0]

fit_rA_s = RA_s_wrap(data[:,0], d_fit)

plt.plot(data[:,0], data[:,1], label = 'SA', alpha = 0.8, lw=0.7)
plt.plot(v, fit_rA_s, label='ideal film', lw = 1, alpha = 1)
#plt.plot(v, rA_p, label='p-pol')
#plt.plot(v, cn2.real-1.33, label='n', alpha=0.3)
#plt.plot(v, cn2.imag, label='k', alpha=0.3)
#plt.axvline(x=1650, c="r", ls="--", alpha=0.5)

#plt.ylim([0.1, 0.2])
#plt.xlim([3000,2800])


plt.xlabel('Wavenumber $(cm^{-1})$')
plt.ylabel('RA')
plt.title('Baseline Test: Ideal Transparent Thin Film vs SA')
plt.legend()
plt.gca().invert_xaxis()
#plt.savefig("../ds-env/Data/IRRAS/20260814/codevsSA.png", dpi=300)
plt.show()

################ SoL data #############
data = np.loadtxt("../ds-env/Data/IRRAS/20260904/260904SoL-46mN.dpt")
v, cn2 = Water_inter_data(data) #interpolate water data to match SA data

def RA_s_wrap(v, d):
    rS = Rs(theta1, cn2, d=d, nmax=n, v=v, alpha=alpha, fwhh=fwhh, v0=v0, tilt=tilt, kmax=kmax)
    rP = Rp(theta1, cn2, d=d, nmax=n, v=v, alpha=alpha, fwhh=fwhh, v0=v0, tilt=tilt, kmax=kmax)
    r0s = Water_R0s(theta1, cn2)
    r0p = Water_R0p(theta1, cn2)

    rA_s = RAs_eff(rP, rS, r0p, r0s, Gamma_pol)
    return rA_s

#mask for baseline
x = data[:,0]
y = data[:,1]

x_min, x_max = 2600, 2700
mask = (x > x_min) & (x < x_max)
y_masked = y[mask]
baseY = y - y_masked.mean()


popt, pcov = curve_fit(RA_s_wrap, x, baseY, p0=[d], bounds=(1e-10, 1e-6))
print("Fitted d:", popt[0], "+/-", np.sqrt(np.diag(pcov))[0])
d_fit = popt[0]

fit_rA_s = RA_s_wrap(data[:,0], d_fit)

plt.plot(x, baseY, label = 'SoL', alpha = 0.8, lw=0.7)
plt.plot(v, fit_rA_s, label='ideal film', lw = 1, alpha = 1)
#plt.plot(v, rA_p, label='p-pol')
#plt.plot(v, cn2.real-1.33, label='n', alpha=0.3)
#plt.plot(v, cn2.imag, label='k', alpha=0.3)
#plt.axvline(x=1650, c="r", ls="--", alpha=0.5)

#plt.ylim([0.1, 0.2])
#plt.xlim([3000,2800])


plt.xlabel('Wavenumber $(cm^{-1})$')
plt.ylabel('RA')
plt.title('Baseline Modelling: Ideal Transparent Thin Film vs SoL')
plt.legend()
plt.gca().invert_xaxis()
plt.tight_layout()
#plt.savefig("../ds-env/Data/IRRAS/20260904/baseline-SoL-46mN.png", dpi=300)
plt.show()
