"""Exercise exported JSON and annotated video for the requested three outcomes."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import torch
from shoplifting.mil import MILHead
from shoplifting.predict import main

class FakeEncoder:
    def __init__(self,*args,**kwargs):pass
    def encode(self,clips):return np.zeros((len(clips),512),dtype=np.float32)

class PredictionDecisionTests(unittest.TestCase):
    def run_clip(self,duration,bias,blank=False):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'input.mp4';fps=10
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),fps,(320,240))
            rng=np.random.default_rng(10)
            for _ in range(round(duration*fps)):
                frame=np.zeros((240,320,3),dtype=np.uint8) if blank else rng.integers(10,240,(240,320,3),dtype=np.uint8)
                writer.write(frame)
            writer.release()
            original=source.read_bytes()
            head=MILHead(512,8,0)
            for parameter in head.parameters():parameter.data.zero_()
            head.network[-1].bias.data.fill_(bias)
            checkpoint={'head_type':'mil','input_dim':512,'hidden_dim':8,'dropout':0,'state_dict':head.state_dict(),
                'feature_mean':torch.zeros(512),'feature_scale':torch.ones(512),'threshold':.5,'topk_fraction':.25,
                'feature_config':{'backbone':'r3d_18','weights':'KINETICS400_V1','num_frames':16,'window_seconds':2,'stride_seconds':1,'max_windows':0},
                'decision_policy':{'positive_threshold':.8,'negative_threshold':.2,'min_context_seconds':8.0}}
            model=root/'model.pt';torch.save(checkpoint,model)
            output=root/'prediction.json';video=root/'annotated.mp4'
            with patch('shoplifting.predict.VideoEncoder',FakeEncoder),contextlib.redirect_stdout(io.StringIO()):
                result=main([str(source),'--model',str(model),'--output',str(output),'--annotated-video',str(video)])
            saved=json.loads(output.read_text());self.assertEqual(saved['status'],result['status'])
            self.assertEqual(source.read_bytes(),original)
            capture=cv2.VideoCapture(str(video));frames=0
            while True:
                ok,_=capture.read()
                if not ok:break
                frames+=1
            capture.release();self.assertEqual(frames,round(duration*fps))
            self.assertFalse(result['theft_confirmed']);self.assertFalse(result['shelf_to_pocket_sequence_verified'])
            return result
    def test_four_second_weak_signal_is_inconclusive(self):
        self.assertEqual(self.run_clip(4,-4)['status'],'insufficient_evidence')
    def test_four_second_strong_signal_requests_review(self):
        self.assertEqual(self.run_clip(4,4)['status'],'needs_review')
    def test_longer_low_signal_can_be_unflagged_without_confirming_no_theft(self):
        self.assertEqual(self.run_clip(9,-4)['status'],'no_review_flag')
    def test_blank_clip_is_inconclusive_even_with_high_model_score(self):
        self.assertEqual(self.run_clip(4,4,True)['status'],'insufficient_evidence')
