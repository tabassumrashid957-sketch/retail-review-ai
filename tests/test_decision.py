import unittest
import math
from shoplifting.decision import calibrate_policy, classify_decision

class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.policy={'positive_threshold':.8,'negative_threshold':.2,'min_context_seconds':8.0}
    def test_short_weak_clip_never_becomes_a_negative_verdict(self):
        for duration in [3,4,5,7,7.9]:
            self.assertEqual(classify_decision(.01,duration,self.policy)['status'],'insufficient_evidence')
    def test_short_high_signal_can_request_review_without_claiming_theft(self):
        result=classify_decision(.9,4,self.policy)
        self.assertEqual(result['status'],'needs_review')
        self.assertFalse(result['theft_confirmed']);self.assertFalse(result['shelf_to_pocket_verified'])
    def test_long_duration_alone_cannot_resolve_an_uncertain_signal(self):
        self.assertEqual(classify_decision(.5,300,self.policy)['status'],'insufficient_evidence')
        self.assertEqual(classify_decision(.2,8,self.policy)['status'],'no_review_flag')
    def test_visibility_failure_overrides_high_score(self):
        self.assertEqual(classify_decision(.99,20,self.policy,['mostly_blank'])['status'],'insufficient_evidence')
    def test_bounds_respect_empirical_support_and_score_ties(self):
        p=calibrate_policy([0,0,0,1,1,1],[.1,.1,.2,.8,.8,.9],.5)
        self.assertEqual(p['negative_threshold'],.2);self.assertEqual(p['positive_threshold'],.8)
        self.assertEqual(p['validation_support']['positive']['count'],3)
        unsupported=calibrate_policy([0,1],[.1,.9],.5)
        self.assertIsNone(unsupported['positive_threshold']);self.assertIsNone(unsupported['negative_threshold'])
    def test_invalid_and_overlapping_bounds_are_rejected(self):
        for score in [float('nan'),-1,2]:
            with self.assertRaises(ValueError):classify_decision(score,4,self.policy)
        with self.assertRaises(ValueError):classify_decision(.5,4,{'positive_threshold':.2,'negative_threshold':.3})
    def test_predict_uses_no_negative_conclusions_for_legacy_or_manual_threshold(self):
        from shoplifting.predict import decision_for_score
        d,p=decision_for_score(.01,30,{'threshold':.5},[])
        self.assertEqual(d['status'],'insufficient_evidence')
        d,p=decision_for_score(.01,30,{'threshold':.5,'decision_policy':self.policy},[],.6)
        self.assertEqual(d['status'],'insufficient_evidence');self.assertIsNone(p['negative_threshold'])
