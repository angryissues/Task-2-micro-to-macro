"""Synthetic integration checks before opening extension outputs."""
import copy,json,os,sys,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parent))
from run import CodaPolicy,configure,run_policy
ROOT=Path(__file__).resolve().parents[4]
AUTHOR=Path(os.environ.get('NLSY97_CODA_AUTHOR','/private/tmp/nlsy97-coda-author'))
h2,CODA=configure(ROOT,AUTHOR)
CFG=json.loads((Path(__file__).parent/'protocol.json').read_text())
torch.set_num_threads(1)

class Tests(unittest.TestCase):
    def setUp(self):
        self.p=np.random.default_rng(731).uniform(.05,.95,(12,3))
        self.y=np.arange(12)%2
    def test_initial_known_labels_condition_posterior(self):
        public=np.full(12,np.nan);public[:3]=self.y[:3]
        bridge=CodaPolicy(self.p,public,False,CFG['coda'],1,CODA)
        conditioned=CodaPolicy(self.p,public,True,CFG['coda'],1,CODA)
        self.assertEqual(conditioned.selector.labels,self.y[:3].tolist())
        self.assertEqual(bridge.selector.labels,[])
        self.assertEqual(bridge.selector.unlabeled_idxs,conditioned.selector.unlabeled_idxs)
        self.assertFalse(torch.equal(bridge.selector.dirichlets,conditioned.selector.dirichlets))
    def test_never_queried_truth_cannot_change_trace(self):
        cfg=copy.deepcopy(CFG);cfg.update(epsilon=0,budgets=[0,1,2])
        for policy in ['CODA_bridge','CODA_conditioned']:
            public=np.full(12,np.nan);public[0]=self.y[0]
            snaps,trace=run_policy(self.p,public,h2.LabelOracle(self.y),policy,cfg,72,h2,CODA)
            altered=self.y.copy();unused=set(range(1,12))-{i for i,y in trace}
            self.assertTrue(unused)
            for i in unused:altered[i]=1-altered[i]
            snaps2,trace2=run_policy(self.p,public,h2.LabelOracle(altered),policy,cfg,72,h2,CODA)
            self.assertEqual(trace,trace2);self.assertEqual(snaps,snaps2)
            self.assertEqual(len(trace),len(set(i for i,y in trace)))
            self.assertTrue(all(i!=0 for i,y in trace))
    def test_already_certified_costs_zero(self):
        p=np.repeat(self.p[:,:1],3,axis=1)
        class NoQuery:
            def query(self,i):raise AssertionError('unnecessary paid query')
        for policy in CFG['policies']:
            snapshots,trace=run_policy(p,np.full(12,np.nan),NoQuery(),policy,CFG,8,h2,CODA)
            self.assertEqual(trace,[])
            self.assertTrue(all(s['cost']==0 and s['certified'] for s in snapshots))
    def test_complete_reveal_matches_true_loss_and_budget(self):
        cfg=copy.deepcopy(CFG);cfg.update(epsilon=0,budgets=[0,1,12])
        for policy in CFG['policies']:
            snapshots,trace=run_policy(self.p,np.full(12,np.nan),h2.LabelOracle(self.y),policy,cfg,42,h2,CODA)
            self.assertLessEqual(len(trace),12)
            for s in snapshots:self.assertLessEqual(s['cost'],s['budget'])
            self.assertTrue(snapshots[-1]['certified'])
            losses=((self.p-self.y[:,None])**2).mean(0)
            self.assertEqual(snapshots[-1]['model_index'],int(losses.argmin()))
if __name__=='__main__':unittest.main()

