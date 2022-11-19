import numpy as np
import matplotlib.pylab as plt

import pymc as pm
import arviz as az
import aesara.tensor as at
import aesara.tensor.extra_ops as ate

import astropy.cosmology as cosmo
import astropy.units as u
import h5py


##########################################################################################################


file = h5py.File('0noise_gaussian_O4.h5','r')
Mcz = np.array(file['det_Mc'])
q = np.array(file['q'])
DL = np.array(file['d_L'])
file.close()


##########################################################################################################


def at_interp(x, xs, ys):
    x = at.as_tensor(x)
    xs = at.as_tensor(xs)
    ys = at.as_tensor(ys)

    ind = ate.searchsorted(xs, x)
    r = (x - xs[ind-1])/(xs[ind]-xs[ind-1])
    return r*ys[ind] + (1-r)*ys[ind-1]


def Ez(z, Om, w, wDM):
    opz = 1 + z
    return at.sqrt(Om*opz**(3*(1+wDM)) + (1-Om)*opz**(3*(1+w)))

def dCs(zs, Om, w, wDM):
    dz = zs[1:] - zs[:-1]
    fz = 1/Ez(zs, Om, w, wDM)
    I = 0.5*dz*(fz[:-1] + fz[1:])
    return at.concatenate([at.as_tensor([0.0]), at.cumsum(I)])

def dLs(zs, dCs):
    return dCs*(1+zs)


##########################################################################################################


Nobs = len(Mcz)
logds = np.log(DL) #log Mpc
Ns = len(DL[0])
wDM = 0

Mczo = np.zeros([Nobs, Ns])

for i in range(Nobs):
    Mczo[i].fill(Mcz[i])
    
print(Mczo, logds)


##########################################################################################################


def make_model(Mczo, logds, zmax=100, Nz=1024):

    zinterp = np.expm1(np.linspace(np.log(1), np.log1p(zmax), Nz))

    with pm.Model() as model:        
        mu_P = pm.Uniform('mu_P', 0.7, 1.7) #in solar mass unit
        sigma_P = pm.Uniform('sigma_P', 0.05, 0.15)

        h = pm.Uniform('h', 0.2, 1.2)
        Om = pm.Uniform('Om', 0.1, 0.5)
        w = pm.Uniform('w', -1.5, -0.5)
        
        dH = pm.Deterministic('dH', 2.99792*(10**3) / h) # Mpc
        
        ds = at.exp(logds)
        dCinterp = dH*dCs(zinterp, Om, w, wDM)
        dLinterp = dLs(zinterp, dCinterp)
        z = at_interp(ds, dLinterp, zinterp)
        z_unit = z/10
        
        var = at.exp(pm.logp(pm.Beta.dist(3, 9), z_unit))
        ddLdz = ds/(1+z)+dH*(1+z)/Ez(z, Om, w, wDM)
        var1 = ds/(10*ddLdz)
        var2 = 1/(1+z)
        Mc = Mczo/(1 + z)
        var3 = at.exp(pm.logp(pm.Normal.dist(mu_P, sigma_P), Mc))
        var4 = at.sum(var*var1*var2*var3, axis=1)
        pm.Potential('pos', at.sum(at.log(var4)))
    return model


tune = 1000
target_accept = 0.99
with make_model(Mczo, logds) as model:
    trace = pm.sample(tune=tune, target_accept=target_accept)
    

##########################################################################################################


with model:
    axes = az.plot_trace(trace, compact=True, var_names=['mu_P', 'sigma_P', 'h', 'Om', 'w'])
    fig = axes.ravel()[0].figure
    fig.savefig('O4.png')
    
    
axes = az.plot_pair(trace, var_names=['mu_P', 'sigma_P', 'h', 'Om', 'w'], marginals=True, kind=['scatter', 'kde'], divergences=True)
fig = axes.ravel()[0].figure
fig.savefig('O4_1.png')