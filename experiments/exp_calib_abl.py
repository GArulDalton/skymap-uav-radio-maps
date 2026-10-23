import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)
os.makedirs("results", exist_ok=True)
"""Exp 2: ablation + model-mismatch robustness + uncertainty calibration (2% sampling)."""
import numpy as np, json, warnings; warnings.filterwarnings('ignore')
from scipy.stats import norm
from env import *; from methods import *; from methods import _gp; from skygp import SkyGP
out=dict(abl=[],mis=[],cal=[])
def split(S,sd,fr=0.02):
  v=S.idx_valid; rng=np.random.default_rng(77+sd); idx=rng.choice(v,int(fr*len(v)),replace=False)
  return idx,S.measure(idx),np.setdiff1d(v,idx)
for sd in [1,2,3,4,5]:
  S=Scenario(sd); idx,Y,test=split(S,sd)
  r={}
  for name,kw in [('Full',{}),('w/o antenna prior',dict(antenna=False)),('w/o multi-cell',dict(multicell=False)),
                  ('w/o LoS-heteroscedastic',dict(hetero=False)),('w/o all three',dict(antenna=False,multicell=False,hetero=False))]:
    g=SkyGP(S,**kw).fit(idx,Y); M,sdv=g.predict(test,return_std=True)
    r[name]=float(np.sqrt(((M-S.PR[test])**2).mean()))
    if name=='Full':
      sn=g.hyper()['sigma_n']; ytest=S.PR[test]+MEAS_STD*np.random.default_rng(sd).standard_normal(M.shape)
      st=np.sqrt(sdv**2+sn**2); z=(ytest-M)/st
      cal={'SKYMAP':dict(picp95=float((np.abs(z)<1.96).mean()),picp68=float((np.abs(z)<1).mean()),
            nll=float((0.5*z**2+np.log(st)+0.5*np.log(2*np.pi)).mean()), mpiw95=float((2*1.96*st).mean()))}
      # PR-Kriging (sklearn predictive std incl. white noise)
      prior,_=fit_twoslope(S,idx,Y); Xs=coords(S,idx); Xa=coords(S,test); Mp=np.zeros_like(M); sp=np.zeros_like(M)
      for b in range(NB):
        m,s_=_gp(Xs,Y[:,b]-prior[idx,b],Xa,return_std=True); Mp[:,b]=prior[test,b]+m; sp[:,b]=s_
      z=(ytest-Mp)/sp
      cal['PR-Kriging']=dict(picp95=float((np.abs(z)<1.96).mean()),picp68=float((np.abs(z)<1).mean()),
            nll=float((0.5*z**2+np.log(sp)+0.5*np.log(2*np.pi)).mean()), mpiw95=float((2*1.96*sp).mean()))
      # calibration curve data (seed 1)
      if sd==1:
        qs=np.linspace(0.05,0.95,10)
        cal['curve']={'q':qs.tolist(),
          'SKYMAP':[float((np.abs((ytest-M)/st)<norm.ppf(0.5+q/2)).mean()) for q in qs],
          'PR-Kriging':[float((np.abs(z)<norm.ppf(0.5+q/2)).mean()) for q in qs]}
      out['cal'].append(cal)
  out['abl'].append(r); print('abl',sd,{k:round(x,2) for k,x in r.items()},flush=True)
  # model mismatch: no inter-site corr, constant 8 dB shadowing
  S2=Scenario(sd,rho_site=0.0,sigma_const=8.0); idx,Y,test=split(S2,sd)
  m={}
  m['PR-Kriging']=float(np.sqrt(((fit_pr_kriging(S2,idx,Y)[test]-S2.PR[test])**2).mean()))
  g=SkyGP(S2).fit(idx,Y); m['SKYMAP']=float(np.sqrt(((g.predict(test)-S2.PR[test])**2).mean())); m['hyper']={k:float(x) for k,x in g.hyper().items()}
  m['PINF']=float(np.sqrt(((fit_nf(S2,idx,Y,'skymap',fscale=1.0,hidden=64,lam_s=5,lam_r=0.2,iters=1200)[test]-S2.PR[test])**2).mean()))
  out['mis'].append(m); print('mis',sd,m,flush=True)
  json.dump(out,open('results/abl.json','w'))
