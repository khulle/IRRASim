import numpy as np
from lmfit.models import LorentzianModel, ConstantModel, LinearModel

#fit RA data - lmfit!
def params(guess, v, rA_data):
    c, s, a, o = guess
    peak = LorentzianModel()
    background = ConstantModel()
    mod = background + peak
    pars = background.make_params(c=o) #initial offset guess
    pars += peak.make_params(center=c, sigma=s, amplitude=a) #initial peak guesses

    out = mod.fit(rA_data, pars, x=v)
    #print(out.fit_report())

    fwhh = out.params['sigma'].value * 2 #convert sigma to fwhh
    v0 = out.params['center'].value
    offset = out.params['c'].value
    amplitude = out.params['amplitude'].value

    return fwhh, v0, offset, amplitude

'''
def baseline(guess, fixed, v, rA_data): #feed simulated rA_data
    #pick "knots" -> outside of peak region 
    fwhh, v0, v_min, v_max = fixed
    knots_xvals = np.random.uniform(v_min, v0-fwhh/2, size=5) #5 random knots on left
    knots_xvals = np.append(knots_xvals, np.random.uniform(v0+fwhh/2, v_max, size=5)) #5 random knots on right
    #mask = (v > v_min) | (v < v0-fwhh/2)

    bkg = SplineModel(prefix='bkg_', xknots=knot_xvals)
    params.update(bkg.guess(y, x))

    #lorentzian + spline model
    c, s, a, o = guess
    peak = LorentzianModel()
    mod = bkg + peak
    pars = bkg.make_params(c=o) #initial offset guess
    pars += peak.make_params(center=c, sigma=s, amplitude=a) #initial peak guesses

    out = mod.fit(rA_data, pars, x=v)

    fwhh = out.params['sigma'].value * 2 #convert sigma to fwhh
    v0 = out.params['center'].value
    offset = out.params['c'].value
'''

def background(guess, v, rA_data):
    c, s, a, o = guess
    peak = LorentzianModel(prefix='peak_')
    background = LinearModel(prefix='bkg_')
    mod = background + peak
    pars = background.guess(rA_data, x=v) #initial linear baseline guess
    pars += peak.make_params(center=c, sigma=s, amplitude=a) #initial peak guesses

    out = mod.fit(rA_data, pars, x=v)
    
    comps = out.eval_components()
    #print(out.fit_report())

    return comps['bkg_'], comps['peak_']


    