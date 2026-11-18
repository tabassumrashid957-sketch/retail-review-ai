import io
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from shoplifting.web_app import JobStore, browser_preview, create_app

ROOT=Path(__file__).resolve().parents[1]

class JobStoreTests(unittest.TestCase):
    def test_damaged_history_records_do_not_block_valid_saved_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = {'state': 'failed', 'created_at': '2026-11-18T00:00:00+00:00',
                    'reviews': {}, 'suggestions': [], 'filename': 'clip.mp4'}
            invalid = [[], {**base, 'id': None}, {**base, 'state': []},
                       {**base, 'created_at': None}, {**base, 'reviews': []},
                       {**base, 'suggestions': {}}, {**base, 'state': 'complete', 'result': None}]
            originals = {}
            for index, data in enumerate(invalid):
                identifier = f'{index:032x}'
                if isinstance(data, dict) and 'id' not in data:
                    data['id'] = identifier
                folder = root / identifier
                folder.mkdir()
                path = folder / 'job.json'
                path.write_text(json.dumps(data))
                originals[path] = path.read_bytes()
            valid_id = 'f' * 32
            folder = root / valid_id
            folder.mkdir()
            (folder / 'job.json').write_text(json.dumps({**base, 'id': valid_id}))
            store = JobStore(root)
            try:
                self.assertEqual(set(store.jobs), {valid_id})
                self.assertEqual(store.get(valid_id)['filename'], 'clip.mp4')
                for path, content in originals.items():
                    self.assertEqual(path.read_bytes(), content)
            finally:
                store.shutdown()


class BrowserPreviewTests(unittest.TestCase):
    def test_odd_sized_video_becomes_playable_browser_preview(self):
        import imageio_ffmpeg

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'odd-sized.mkv'
            folder = root / 'preview'
            folder.mkdir()
            subprocess.run(
                [imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-y', '-f', 'lavfi',
                 '-i', 'testsrc=size=321x241:rate=5:duration=1', '-c:v', 'ffv1', str(source)],
                check=True, capture_output=True,
            )
            browser_preview(source, folder)

            capture = cv2.VideoCapture(str(folder / 'preview.mp4'))
            try:
                readable, frame = capture.read()
                self.assertTrue(readable)
                self.assertEqual(frame.shape[1] % 2, 0)
                self.assertEqual(frame.shape[0] % 2, 0)
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 5)
            finally:
                capture.release()
            self.assertTrue((folder / 'poster.jpg').is_file())


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        source=self.base/'fixture.mp4';writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(320,240))
        for i in range(40):writer.write(np.full((240,320,3),40+i,dtype=np.uint8))
        writer.release();self.video=source.read_bytes();self.calls=[]
        def preview(source,folder):
            (folder/'preview.mp4').write_bytes(source.read_bytes());(folder/'poster.jpg').write_bytes(b'poster')
        def runner(source,folder,progress):
            self.calls.append(source);progress(70,'Analysing video window 2 of 3')
            return {'status':'insufficient_evidence','result_text':'CANNOT DETERMINE','video_score':.5,'review_threshold':.8,'duration_seconds':4,
              'visibility':{'issues':[]},'decision':{'message':'Incomplete evidence'},
              'review_intervals':[],'windows':[{'start_seconds':0,'end_seconds':2,'review_score':.2},{'start_seconds':1,'end_seconds':3,'review_score':.6},{'start_seconds':2,'end_seconds':4,'review_score':.3}]}
        self.app=create_app(ROOT,self.base/'jobs',runner=runner,previewer=preview);self.client=self.app.test_client();self.headers={'X-Review-Token':self.app.config['REVIEW_TOKEN']}
    def tearDown(self):
        self.app.extensions['review_store'].shutdown();self.temp.cleanup()
    def upload(self,name='clip.mp4',data=None):
        return self.client.post('/api/analyses',data={'video':(io.BytesIO(self.video if data is None else data),name)},headers=self.headers)
    def finish(self,job_id):
        for _ in range(100):
            job=self.client.get('/api/jobs/'+job_id).get_json()
            if job['state'] in ['complete','failed']:return job
            time.sleep(.01)
        self.fail('Analysis did not finish')
    def test_upload_runs_analysis_produces_real_ranges_and_supports_seek(self):
        response=self.upload('../../my retail clip.mp4');self.assertEqual(response.status_code,202)
        job=self.finish(response.get_json()['id']);self.assertEqual(job['state'],'complete');self.assertEqual(len(self.calls),1)
        self.assertEqual(job['result']['status'],'insufficient_evidence');self.assertEqual(job['suggestions'][0]['kind'],'whole_clip')
        self.assertEqual(job['suggestions'][0]['end_seconds'],4)
        self.assertEqual(job['filename'],'my_retail_clip.mp4');self.assertEqual(len(job['windows']),3)
        media=self.client.get(job['preview_url'],headers={'Range':'bytes=0-31'})
        self.assertEqual(media.status_code,206);self.assertEqual(len(media.data),32)
        media.close()
    def test_human_review_is_saved_with_actual_segment_bounds_and_exported(self):
        job=self.finish(self.upload().get_json()['id']);identifier=job['suggestions'][0]['id']
        response=self.client.post(f'/api/jobs/{job["id"]}/review',json={'decision':'unclear','segment_id':identifier,'notes':'The item is hidden.'},headers=self.headers)
        self.assertEqual(response.status_code,200);review=response.get_json()['review']
        self.assertEqual(review['start_seconds'],0);self.assertEqual(review['end_seconds'],4);self.assertEqual(review['source'],'human_review')
        saved=json.loads((self.base/'jobs'/job['id']/'job.json').read_text());self.assertEqual(saved['reviews'][identifier]['decision'],'unclear')
        report=self.client.get(job['report_url']);self.assertEqual(report.status_code,200)
        self.assertEqual(json.loads(report.data)['reviews'][identifier]['notes'],'The item is hidden.')
    def test_bad_uploads_and_forged_ranges_are_rejected(self):
        self.assertEqual(self.upload('clip.html').status_code,400);self.assertEqual(self.upload(data=b'').status_code,400)
        job=self.finish(self.upload().get_json()['id'])
        for data in [{'decision':'guilty'},{'decision':[]},{'decision':'unclear','segment_id':'made-up'},{'decision':'unclear','notes':'x'*2001}]:
            response=self.client.post(f'/api/jobs/{job["id"]}/review',json=data,headers=self.headers);self.assertEqual(response.status_code,400)
    def test_mutation_requires_local_page_token_and_media_cannot_read_arbitrary_files(self):
        response=self.client.post('/api/analyses',data={'video':(io.BytesIO(self.video),'clip.mp4')});self.assertEqual(response.status_code,403)
        self.assertEqual(self.client.get('/api/jobs/not-an-id').status_code,404)
        job=self.finish(self.upload().get_json()['id']);self.assertEqual(self.client.get(f'/api/jobs/{job["id"]}/media/input').status_code,404)
    def test_review_rejects_non_object_payloads_and_invalid_scope(self):
        job = self.finish(self.upload().get_json()['id'])
        endpoint = f'/api/jobs/{job["id"]}/review'
        for data in ([{'decision': 'unclear'}], 'unclear', 1, True, None):
            with self.subTest(payload=data):
                response = self.client.post(endpoint, json=data, headers=self.headers)
                self.assertEqual(response.status_code, 400)
                self.assertTrue(response.is_json)
        for segment_id in ([], {}, 0, False, ''):
            with self.subTest(segment_id=segment_id):
                response = self.client.post(endpoint, json={'decision': 'unclear', 'segment_id': segment_id}, headers=self.headers)
                self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get('/api/jobs/' + job['id']).get_json()['reviews'], {})

    def test_failed_decoder_is_reported_without_a_success_result(self):
        job=self.finish(self.upload(data=b'not a video').get_json()['id'])
        self.assertEqual(job['state'],'failed');self.assertIsNone(job['result'])
    def test_interface_is_served_without_distributing_research_videos(self):
        response=self.client.get('/');self.assertEqual(response.status_code,200);self.assertIn(b'Suggested time ranges',response.data)
        self.assertIn(self.app.config['REVIEW_TOKEN'].encode(),response.data)
        # Third-party research videos are deliberately absent in this checkout.
        self.assertEqual(self.client.get('/api/samples').get_json()['samples'],[])
        response=self.client.get('/assets/app.js');self.assertEqual(response.status_code,200);response.close()

    def test_relaunch_reuses_existing_app_without_reloading_active_jobs(self):
        from shoplifting import web_app
        with patch.object(web_app,'existing_app',return_value=True),patch.object(web_app,'create_app') as create,patch.object(web_app.webbrowser,'open') as open_browser:
            self.assertEqual(web_app.main(['--no-open']),0);create.assert_not_called();open_browser.assert_not_called()
        with patch.object(web_app,'existing_app',return_value=True),patch.object(web_app,'create_app') as create,patch.object(web_app.webbrowser,'open') as open_browser:
            self.assertEqual(web_app.main([]),0);create.assert_not_called();open_browser.assert_called_once_with('http://127.0.0.1:8765')
