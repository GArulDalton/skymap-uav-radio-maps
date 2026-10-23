import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)
os.makedirs("results", exist_ok=True)
"""Exp 3: uncertainty-guided sampling vs random and space-filling."""
import numpy as np, json, warnings; warnings.filterwarnings('ignore')
from scipy.spatial import cKDTree
from env import *; from skygp import SkyGP
F0,STEP,R=0.005,0.005,5; MINSEP=30.0
out=[]
for sd in [1,2,3,4,5]:
  S=Scenario(sd); v=S.idx_valid; N=len(v)
  rng0=np.random.default_rng(500+sd); init=rng0.choice(v,int(F0*N),replace=False)
  res={}
  for strat in ['Random','Space-filling','SKYMAP-UG']:
    rng=np.random.default_rng(900+sd); idx=init.copy(); Y=S.measure(idx); curve=[]
    for rnd in range(R+1):
      g=SkyGP(S).fit(idx,Y); test=np.setdiff1d(v,idx)
      curve.append(float(np.sqrt(((g.predict(test)-S.PR[test])**2).mean())))
      if rnd==R: break
      k=int(STEP*N); pool=np.setdiff1d(v,idx); cand=rng.choice(pool,min(3000,len(pool)),replace=False)
      if strat=='Random': new=rng.choice(pool,k,replace=False)
      else:
        if strat=='Space-filling':
          score=cKDTree(S.P[idx]).query(S.P[cand])[0]
        else:
          _,s_=g.predict(cand,return_std=True); score=(s_**2).sum(1)
        order=np.argsort(-score); new=[]; tree_pts=[]
        for c in order:
          p=S.P[c]
          if all(np.linalg.norm(p-q)>=MINSEP for q in tree_pts): new.append(c); tree_pts.append(p)
          if len(new)==k: break
        if len(new)<k: new+=list(rng.choice(np.setdiff1d(cand,new),k-len(new),replace=False))
        new=np.array(new)
      idx=np.r_[idx,new]; Y=np.r_[Y,S.measure(new)]
    res[strat]=curve; print(sd,strat,[round(c,2) for c in curve],flush=True)
  out.append(res); json.dump(out,open('results/active.json','w'))
