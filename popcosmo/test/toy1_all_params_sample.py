import numpy as np
import matplotlib.pylab as plt

import pymc as pm
import arviz as az
import aesara.tensor as at
import aesara.tensor.extra_ops as ate

import astropy.cosmology as cosmo
import astropy.units as u


Nobs = 1000
z = np.random.beta(3, 9, Nobs)*10
Mc = np.random.normal(1.17, 0.1, Nobs) # in solar mass unit
Mcz = Mc*(1+z)

cp = cosmo.Planck18
DL = cp.luminosity_distance(z).to(u.Gpc).value
DL10 = cp.luminosity_distance(10).to(u.Gpc).value
sigma_logDL = 0.1+0.3*(DL/DL10)
zo = z
logdo = np.log(DL)+sigma_logDL*np.random.randn(Nobs) # in Gpc


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


Ns = 5000
logds = np.zeros([Nobs, Ns])
for i in range(Nobs):
    logds[i] = np.random.normal(logdo[i], sigma_logDL[i], Ns).T
Mczo = np.zeros([Nobs, Ns])
for i in range(Nobs):
    Mczo[i].fill(Mcz[i])


wDM = 0
def make_model(Mczo, logds, zmax=100, Nz=1024):

    zinterp = np.expm1(np.linspace(np.log(1), np.log1p(zmax), Nz))

    with pm.Model() as model:        
        mu_P = pm.Uniform('mu_P', 0.7, 1.7) #in solar mass unit
        sigma_P = pm.Uniform('sigma_P', 0.05, 0.15)

        h = pm.Uniform('h', 0.2, 1.2)
        Om = pm.Uniform('Om', 0.1, 0.5)
        w = pm.Uniform('w', -1.5, -0.5)
        
        dH = pm.Deterministic('dH', 2.99792 / h) # Gpc
        
        ds = pm.Deterministic('ds', at.exp(logds))
        
        dCinterp = dH*dCs(zinterp, Om, w, wDM)
        dLinterp = dLs(zinterp, dCinterp)
        z = pm.Deterministic('z', at_interp(ds, dLinterp, zinterp))
        
        z_unit = pm.Deterministic('z_unit', z/10)
        
        var = pm.Deterministic('var', at.exp(pm.logp(pm.Beta.dist(3, 9), z_unit)))
        var1 = pm.Deterministic('var1', at.sum(var, axis=1))
        pm.Potential('zunitprior', at.sum(at.log(var1)))
        
        ddLdz = ds/(1+z)+dH*(1+z)/Ez(z, Om, w, wDM)
        var2 = pm.Deterministic('var2', at.sum(ds/(10*ddLdz), axis=1))
        pm.Potential('zjac', at.sum(at.log(var2)))
        
        var3 = pm.Deterministic('var3', at.sum(1/(1+z), axis=1))
        pm.Potential('mcjac', at.sum(at.log(var3)))
        
        Mc = pm.Deterministic('Mc', Mczo/(1 + z))
        var4 = pm.Deterministic('var4', at.exp(pm.logp(pm.Normal.dist(mu_P, sigma_P), Mc)))
        var5 = pm.Deterministic('var5', at.sum(var4, axis=1))
        pm.Potential('mcprior', at.sum(at.log(var5)))
    return model

tune = 1000
target_accept = 0.99
with make_model(Mczo, logds) as model:
    trace = pm.sample(tune=tune, target_accept=target_accept)


axes = az.plot_pair(trace, var_names=['mu_P', 'sigma_P', 'h', 'Om', 'w'], marginals=True, kind=['scatter', 'kde'], divergences=True)
fig = axes.ravel()[0].figure
fig.savefig('toy1_all_params_sample.pdf')


with model:
    display(az.summary(trace, var_names=['mu_P', 'sigma_P', 'h', 'Om', 'w']))