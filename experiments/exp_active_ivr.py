import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)
os.makedirs("results", exist_ok=True)
import numpy as np, json, warnings; warnings.filterwarnings('ignore')
from env import *; from skygp import SkyGP, ivr_scores
ANCHOR = "--anchor" in sys.argv   # pilot: calibrate prior + hyper-parameters on the initial random set only
F0,STEP,R=0.005,0.005,5; MINSEP=30.0
out=[]
for sd in [1,2,3,4,5]:
  S=Scenario(sd); v=S.idx_valid; N=len(v)
  rng0=np.random.default_rng(500+sd); init=rng0.choice(v,int(F0*N),replace=False)
  rng=np.random.default_rng(900+sd); idx=init.copy(); Y=S.measure(idx); curve=[]
  for rnd in range(R+1):
    g=SkyGP(S).fit(idx,Y,n_anchor=len(init) if ANCHOR else None); test=np.setdiff1d(v,idx)
    curve.append(float(np.sqrt(((g.predict(test)-S.PR[test])**2).mean())))
    if rnd==R: break
    k=int(STEP*N); pool=np.setdiff1d(v,idx); cand=rng.choice(pool,1500,replace=False); ref=rng.choice(v,1500,replace=False)
    score=ivr_scores(g,ref,cand); order=np.argsort(-score); new=[]; pts=[]
    for c in order:
      p=S.P[c]
      if all(np.linalg.norm(p-q)>=MINSEP for q in pts): new.append(c); pts.append(p)
      if len(new)==k: break
    if len(new)<k: new+=list(rng.choice(np.setdiff1d(pool,new),k-len(new),replace=False))
    new=np.array(new); idx=np.r_[idx,new]; Y=np.r_[Y,S.measure(new)]
  out.append({'SKYMAP-IVR':curve}); print(sd,[round(c,2) for c in curve],flush=True)
  json.dump(out,open(('results/active_ivr_anchor.json' if ANCHOR else 'results/active_ivr.json'),'w'))
