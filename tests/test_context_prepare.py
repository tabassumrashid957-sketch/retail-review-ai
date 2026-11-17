import unittest
from scripts.prepare_ucf_context import plan_pairs
from shoplifting.mil import TemporalGRUHead, head_from_checkpoint
import torch

class ContextTests(unittest.TestCase):
    def test_frozen_comparison_uses_validation_and_blocks_source_leakage(self):
        from scripts.select_context_model import prefer_frozen, validate_frozen_groups
        self.assertTrue(prefer_frozen({'pr_auc':.8,'f1':.6,'test_accuracy':0}, {'pr_auc':.7,'f1':1,'test_accuracy':1}))
        self.assertFalse(prefer_frozen({'pr_auc':.8,'f1':.6}, {'pr_auc':.8,'f1':.6}))
        with self.assertRaisesRegex(ValueError,'overlap'):
            validate_frozen_groups([{'group':'one','split':'train'}],[{'group':'one','split':'test'}])
        validate_frozen_groups([{'group':'one','split':'train'}],[{'group':'one','split':'train'},{'group':'two','split':'val'}])
    def test_whole_event_and_duration_matching_with_nonoverlapping_normal(self):
        events=[(300,600),(1100,1200)]
        pairs,skipped=plan_pairs(events,2400,30)
        self.assertEqual(len(pairs),2)
        for p in pairs:
            a,b=p['positive'];c,d=p['normal'];start,end=p['event']
            self.assertLessEqual(a,start);self.assertGreaterEqual(b,end)
            self.assertEqual(b-a,d-c)
            for e,f in events:self.assertTrue(d<=e-30 or c>=f+30)
        self.assertGreater(pairs[0]['positive'][1]-pairs[0]['positive'][0],120)
    def test_matching_shortage_does_not_truncate_the_positive(self):
        pairs,skipped=plan_pairs([(50,950)],1000,30)
        self.assertEqual(pairs,[]);self.assertIn('not truncated',skipped[0]['reason'])
    def test_temporal_head_uses_order_and_round_trips(self):
        torch.manual_seed(2)
        model=TemporalGRUHead(512,16,0).eval();x=torch.randn(6,512)
        with torch.no_grad():
            normal=model(x);reversed_back=model(x.flip(0)).flip(0)
        self.assertEqual(normal.shape,(6,));self.assertFalse(torch.allclose(normal,reversed_back))
        c={'head_type':'temporal_gru','input_dim':512,'hidden_dim':16,'dropout':0,'state_dict':model.state_dict()}
        restored=head_from_checkpoint(c).eval()
        with torch.no_grad():self.assertTrue(torch.equal(restored(x),normal))
