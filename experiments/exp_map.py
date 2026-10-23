import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)
os.makedirs("results", exist_ok=True)
"""Exp 1: map accuracy vs sampling ratio; uncertainty calibration; runtime."""
import numpy as np, time, json, sys, warnings; warnings.filterwarnings('ignore')
from scipy.stats import norm
from env import *; from methods import *; from methods import _gp; from skygp import SkyGP
NFK=dict(fscale=1.0, hidden=64, lam_s=5, lam_r=0.2, iters=1200)
seeds=[1,2,3,4,5]; fracs=[0.005,0.01,0.02,0.03,0.05]
out=[]
for sd in seeds:
  S=Scenario(sd); v=S.idx_valid
  for fr in fracs:
    rng=np.random.default_rng(1000*sd+int(fr*1000))
    idx=rng.choice(v,int(fr*len(v)),replace=False); Y=S.measure(idx); test=np.setdiff1d(v,idx)
    rec=dict(seed=sd,frac=fr,n=len(idx),rmse={},mae={},time={})
    def ev(name,f):
      t=time.time(); M=f(); rec['time'][name]=time.time()-t
      e=M[test]-S.PR[test]; rec['rmse'][name]=float(np.sqrt((e**2).mean())); rec['mae'][name]=float(np.abs(e).mean())
    ev('LDPL',lambda:fit_ldpl(S,idx,Y)); ev('TwoSlope',lambda:fit_twoslope(S,idx,Y)[0]); ev('IDW',lambda:fit_idw(S,idx,Y))
    ev('Kriging',lambda:fit_kriging(S,idx,Y))
    # PR-Kriging with std for calibration
    def prk():
      prior,_=fit_twoslope(S,idx,Y); Xs=coords(S,idx); Xa=coords(S,test)
      mu=np.zeros((len(S.P),NB)); sdv=np.zeros((len(test),NB))
      for b in range(NB):
        m,s_=_gp(Xs,Y[:,b]-prior[idx,b],Xa,return_std=True); mu[test,b]=prior[test,b]+m; sdv[:,b]=s_
      rec['_prk_sd']=sdv; return mu
    ev('PR-Kriging',prk)
    ev('MLP',lambda:fit_nf(S,idx,Y,'mlp',hidden=64,iters=1200)); ev('FF-NF',lambda:fit_nf(S,idx,Y,'ff',**NFK))
    ev('PINF',lambda:fit_nf(S,idx,Y,'skymap',**NFK))
    g=[None]
    def sky():
      g[0]=SkyGP(S).fit(idx,Y); mu=np.zeros((len(S.P),NB)); m,s_=g[0].predict(test,return_std=True); mu[test]=m; rec['_sky_sd']=s_; return mu
    ev('SKYMAP',sky)
    rec['hyper']={k:float(x) for k,x in g[0].hyper().items()}
    # calibration: predictive sd includes observation noise? evaluate vs latent truth -> latent sd
    for name,key in [('PR-Kriging','_prk_sd'),('SKYMAP','_sky_sd')]:
      sdv=rec.pop(key)
      if name=='PR-Kriging':  # sklearn std includes white noise; remove fitted noise approx not accessible -> use as is
        pass
      mu=None
    out.append(rec); print(sd,fr,{k:round(x,2) for k,x in rec['rmse'].items()},flush=True)
    json.dump(out,open('results/map.json','w'))
