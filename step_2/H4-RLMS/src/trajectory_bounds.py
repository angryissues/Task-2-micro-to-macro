"""Exact P0/P1 bounds for short binary trajectories."""
from __future__ import annotations
from itertools import product
import numpy as np

def admissible_paths(observed):
    obs=np.asarray(observed,dtype=float); h=obs.size
    out=[]
    for path in product((0,1), repeat=h):
        if all(np.isnan(obs[t]) or int(obs[t])==path[t] for t in range(h)):
            out.append(np.asarray(path,dtype=float))
    return out

def first_exit(path):
    return np.maximum.accumulate(path==0).astype(float)

def loss(path, prediction, target='endpoint'):
    p=np.asarray(prediction,dtype=float); y=np.asarray(path,dtype=float)
    if target=='endpoint': return float((p[-1]-y[-1])**2)
    if target=='multi_horizon': return float(np.mean((p-y)**2))
    if target=='first_exit': return float(np.mean((p-first_exit(y))**2))
    raise ValueError('unknown target')

def bounds(predictions, observed_paths, target='endpoint', weights=None, mode='p1'):
    p=np.asarray(predictions,dtype=float); obs=np.asarray(observed_paths,dtype=float)
    if p.ndim!=3: raise ValueError('predictions must be N x H x K')
    n,h,k=p.shape
    if obs.shape!=(n,h): raise ValueError('observed_paths must be N x H')
    w=np.full(n,1/n) if weights is None else np.asarray(weights,dtype=float)/np.sum(weights)
    lo=np.zeros((k,k)); up=np.zeros((k,k))
    for i in range(n):
        paths=admissible_paths(obs[i])
        for a in range(k):
            for b in range(a+1,k):
                if mode=='p0' and target=='first_exit':
                    vals=[]
                    for exits in product((0,1), repeat=h):
                        feasible=True
                        for t in range(h):
                            if np.any(obs[i,:t+1]==0) and exits[t]!=1: feasible=False
                        if feasible:
                            vals.append(float(np.mean((p[i,:,a]-exits)**2)-np.mean((p[i,:,b]-exits)**2)))
                else:
                    vals=[loss(path,p[i,:,a],target)-loss(path,p[i,:,b],target) for path in paths]
                lo[a,b]+=w[i]*min(vals); up[a,b]+=w[i]*max(vals)
                lo[b,a]=-up[a,b]; up[b,a]=-lo[a,b]
    g=np.max(up,axis=1); return lo,up,g
