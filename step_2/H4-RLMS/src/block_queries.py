"""Public-only minimax queries; an oracle is called strictly after selection."""
from dataclasses import dataclass
from itertools import product
import numpy as np

@dataclass(frozen=True)
class Query:
    row: int
    horizons: tuple[int, ...]
    cost: float
    kind: str

class PublicTrajectoryState:
    def __init__(self,predictions,visible):
        p=np.array(predictions,dtype=float,copy=True);obs=np.array(visible,dtype=float,copy=True)
        if p.ndim!=3 or not 1<=p.shape[1]<=4 or p.shape[0]<1 or p.shape[2]<1:raise ValueError('Expected N x H x K, 1<=H<=4')
        if obs.shape!=p.shape[:2] or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid shapes/probabilities')
        if not np.isin(obs[np.isfinite(obs)],[0,1]).all() or np.isinf(obs).any():raise ValueError('Visible answers must be 0/1 or NaN')
        self.n,self.h,self.k=p.shape;self.predictions=p;p.setflags(write=False);self.visible=obs;self.pow=3**np.arange(self.h);self.codes=(np.where(np.isnan(obs),0,obs+1).astype(int)*self.pow).sum(1)
        paths=np.array(list(product([0.,1.],repeat=self.h)));exits=np.maximum.accumulate(paths==0,axis=1).astype(float)
        losses=((p[:,None,:,:]-exits[None,:,:,None])**2).mean(2);pairs=losses[:,:,:,None]-losses[:,:,None,:];self._cache=np.empty((self.n,3**self.h,self.k,self.k))
        for code in range(3**self.h):
            digits=(code//self.pow)%3;allowed=np.all((digits==0)|(paths==digits-1),axis=1);self._cache[:,code]=pairs[:,allowed].max(1)
        self.upper=self._cache[np.arange(self.n),self.codes].mean(0)
    @property
    def value(self):return float(self.upper.max(1).min())
    def certificate(self,epsilon=.0025):
        g=self.upper.max(1);k=int(g.argmin());return {'model':k,'G':float(g[k]),'certified':bool(g[k]<=epsilon+1e-12)}
    def reveal(self,query,answers):
        if query.row not in range(self.n):raise ValueError('Invalid row')
        ids=np.asarray(query.horizons,int);vals=np.asarray(answers,float)
        if len(ids)==0 or len(set(ids))!=len(ids) or np.any((ids<0)|(ids>=self.h)):raise ValueError('Invalid horizons')
        if vals.shape!=ids.shape or not np.isin(vals,[0,1]).all() or not np.isnan(self.visible[query.row,ids]).all():raise ValueError('Only hidden binary answers may be revealed')
        i=query.row;old=int(self.codes[i]);new=old+int(((vals+1)*self.pow[ids]).sum());self.upper+=(self._cache[i,new]-self._cache[i,old])/self.n;self.codes[i]=new;self.visible[i,ids]=vals

def select_query(state,mode,block_cost,remaining_budget):
    """Worst post-answer minimax value for BOTH points and blocks.

    Ties: worst-case width reduction/cost, cheaper action, lower row/horizon.
    This function has no truth/oracle argument and reads only public state.
    """
    if mode not in ['point_only','block_only','mixed']:raise ValueError('Invalid mode')
    if not np.isfinite(block_cost) or block_cost<=0 or not np.isfinite(remaining_budget) or remaining_budget<0:raise ValueError('Invalid price/budget')
    best=None;bestkey=None;U=state.upper;current=state.value
    for code in np.unique(state.codes):
        rows=np.flatnonzero(state.codes==code);hidden=np.flatnonzero((code//state.pow)%3==0)
        if not len(hidden):continue
        actions=[]
        if mode in ['point_only','mixed'] and remaining_budget>=1-1e-12:actions.extend([(tuple([int(h)]),1.,'point') for h in hidden])
        if mode in ['block_only','mixed'] and remaining_budget>=block_cost-1e-12:actions.append((tuple(map(int,hidden)),float(block_cost),'block'))
        for ids,fee,kind in actions:
            after=np.array([int(code)+sum((bit+1)*state.pow[h] for bit,h in zip(fill,ids)) for fill in product([0,1],repeat=len(ids))],int)
            new_rows=state._cache[rows[:,None],after[None,:]];old_rows=state._cache[rows,int(code)];post=U+(new_rows-old_rows[:,None])/state.n
            worst=post.max(3).min(2).max(1);gain=(current-worst)/fee;width=(U.sum()-post.sum((2,3)).max(1))/fee
            order=np.lexsort((rows,-width,-gain));j=int(order[0]);i=int(rows[j]);key=(float(gain[j]),float(width[j]),-fee,-i,-ids[0])
            if bestkey is None or key>bestkey:bestkey=key;best=Query(i,ids,fee,kind)
    return best

class TrajectoryOracle:
    def __init__(self,truth):
        self._truth=np.array(truth,dtype=float,copy=True)
        if self._truth.ndim!=2 or not np.isin(self._truth,[0,1]).all():raise ValueError('Oracle requires binary N x H truth')
        self.calls=[]
    def query(self,query):
        self.calls.append((query.row,query.horizons));return self._truth[query.row,list(query.horizons)].copy()

def run_queries(predictions,public,oracle,mode,block_cost,budget,epsilon=.0025):
    if mode not in ['point_only','block_only','mixed'] or not np.isfinite(block_cost) or block_cost<=0:raise ValueError('Invalid query mode/price')
    if not np.isfinite(budget) or budget<0 or not np.isfinite(epsilon) or epsilon<0:raise ValueError('Invalid budget/epsilon')
    state=PublicTrajectoryState(predictions,public);spent=0.;ops=0;trace=[];initial=state.certificate(epsilon)
    while not state.certificate(epsilon)['certified']:
        query=select_query(state,mode,block_cost,max(0.,budget-spent))
        if query is None:break
        before=state.value;answers=oracle.query(query);state.reveal(query,answers);spent+=query.cost;ops+=1
        if state.value>before+1e-10:raise AssertionError('Revealing an answer widened exact bounds')
        trace.append({'kind':query.kind,'cost':query.cost,'G_before':before,'G_after':state.value})
    result=state.certificate(epsilon);result.update(total_cost=spent,operations=ops,initial_G=initial['G'],initially_certified=initial['certified'],trace=trace)
    return result
