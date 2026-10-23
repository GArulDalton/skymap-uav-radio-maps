import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)
os.makedirs("results", exist_ok=True)
"""Exp 4: connectivity-aware trajectory planning on 20 m graph."""
import numpy as np, json, warnings; warnings.filterwarnings('ignore')
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from env import *; from methods import fit_pr_kriging; from skygp import SkyGP
TH=0.0; LAM=8.0; NP=40; MC=256
def graph(S):
  i,j,k=S.ijk(np.arange(len(S.P)))
  node=np.where((i%2==0)&(j%2==0)&S.valid)[0]
  pos={n:t for t,n in enumerate(node)}
  E=[]
  for di in (-2,0,2):
    for dj in (-2,0,2):
      for dk in (-1,0,1):
        if di==dj==dk==0: continue
        ii,jj,kk=i[node]+di,j[node]+dj,k[node]+dk
        ok=(ii>=0)&(ii<S.nx)&(jj>=0)&(jj<S.nx)&(kk>=0)&(kk<S.nz)
        nb=S.grid_index(ii[ok],jj[ok],kk[ok]); src=np.arange(len(node))[ok]
        good=S.valid[nb]
        for a,b in zip(src[good],nb[good]):
          if b in pos: E.append((a,pos[b]))
  E=np.array(E); length=np.linalg.norm(S.P[node[E[:,0]]]-S.P[node[E[:,1]]],axis=1)
  return node,E,length
def plan(node,E,length,c,s,t):
  w=length*(1+LAM*0.5*(c[E[:,0]]+c[E[:,1]]))
  G=coo_matrix((w,(E[:,0],E[:,1])),shape=(len(node),)*2).tocsr()
  _,pred=dijkstra(G,indices=s,return_predecessors=True)
  path=[t]
  while path[-1]!=s: path.append(pred[path[-1]])
  return path[::-1]
def evalpath(S,node,path,true_out):
  P=S.P[node[path]]; seg=np.linalg.norm(np.diff(P,axis=0),axis=1)
  o=true_out[path]; ow=0.5*(o[:-1]+o[1:])
  return float((seg*ow).sum()/seg.sum()), float(seg.sum())
out=[]
for sd in [1,2,3,4,5]:
  S=Scenario(sd); node,E,length=graph(S); v=S.idx_valid
  rng=np.random.default_rng(300+sd); idx=rng.choice(v,int(0.02*len(v)),replace=False); Y=S.measure(idx)
  true_out=(sinr_db(S.PR[node])<TH).astype(float)
  g=SkyGP(S).fit(idx,Y); mu,sdv=g.predict(node,return_std=True)
  Mp=fit_pr_kriging(S,idx,Y)[node]
  smp=mu[None]+sdv[None]*np.random.default_rng(sd).standard_normal((MC,)+mu.shape)
  pout=np.stack([(sinr_db(smp[m])<TH) for m in range(MC)]).mean(0)
  costs={'Shortest':np.zeros(len(node)),'PR-Kriging':(sinr_db(Mp)<TH).astype(float),
         'SKYMAP-mean':(sinr_db(mu)<TH).astype(float),'SKYMAP-risk':pout,'Oracle':true_out}
  # outage-prediction quality on graph nodes
  cls={k:dict(acc=float(((c>0.5)==true_out).mean())) for k,c in costs.items() if k not in('Shortest','Oracle')}
  rec={'cls':cls,'paths':{k:[] for k in costs},'node_outage':float(true_out.mean())}
  z=S.P[node,2]; r=np.random.default_rng(700+sd)
  pairs=[]
  while len(pairs)<NP:
    a,b=r.integers(0,len(node),2)
    if np.linalg.norm(S.P[node[a],:2]-S.P[node[b],:2])>=400 and z[a]>=60 and z[b]>=60: pairs.append((a,b))
  for a,b in pairs:
    for k,c in costs.items():
      p=plan(node,E,length,c,a,b); rec['paths'][k].append(evalpath(S,node,p,true_out))
  if sd==1:
    a,b=pairs[0]; rec['example']={k:S.P[node[plan(node,E,length,c,a,b)]].tolist() for k,c in costs.items()}
    k2=np.where(z==90)[0]; rec['slice']=dict(xy=S.P[node[k2],:2].tolist(),true=sinr_db(S.PR[node[k2]]).tolist(),
        sky=sinr_db(mu[k2]).tolist(),pout=pout[k2].tolist())
  out.append(rec)
  print(sd,rec['node_outage'],{k:(round(np.mean([x[0] for x in v_]),3),round(np.mean([x[1] for x in v_]),0)) for k,v_ in rec['paths'].items()},cls,flush=True)
  json.dump(out,open('results/traj.json','w'))
