"""Local upload, actual video inference, timestamp suggestions and human review."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import threading
import uuid
import webbrowser
import urllib.request

import cv2
from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename
from .review_suggestions import build_suggestions
from .video import video_info

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {'.mp4','.mov','.avi','.mkv','.mpeg','.mpg','.wmv'}
REVIEW_CHOICES = {'suspected', 'not_visible', 'unclear'}


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


class JobStore:
    def __init__(self, root):
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock(); self.jobs = {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='retail-review')
        for p in self.root.glob('*/job.json'):
            try:
                data = json.loads(p.read_text())
                if not isinstance(data, dict):
                    continue
                identifier = data.get('id')
                state = data.get('state')
                if not isinstance(identifier, str) or not re.fullmatch(r'[a-f0-9]{32}', identifier) or p.parent.name != identifier:
                    continue
                if not isinstance(state, str) or state not in {'queued', 'running', 'complete', 'failed'}:
                    continue
                if not isinstance(data.get('created_at'), str):
                    continue
                if not isinstance(data.get('reviews'), dict) or not isinstance(data.get('suggestions'), list):
                    continue
                if state == 'complete' and (not isinstance(data.get('result'), dict) or not data['result'].get('status')):
                    continue
                if data['state'] in {'queued','running'}:
                    data.update(state='failed', message='The app closed before analysis finished. Please analyse the clip again.')
                    atomic_json(p,data)
                self.jobs[data['id']] = data
            except (OSError, ValueError, KeyError):
                continue

    def get(self, job_id):
        with self.lock:
            if not re.fullmatch(r'[a-f0-9]{32}',job_id) or job_id not in self.jobs:
                raise KeyError(job_id)
            return json.loads(json.dumps(self.jobs[job_id]))

    def update(self, job_id, **values):
        with self.lock:
            self.jobs[job_id].update(values)
            atomic_json(self.root/job_id/'job.json',self.jobs[job_id])

    def shutdown(self):
        self.executor.shutdown(wait=True)


def browser_preview(source, folder):
    import imageio_ffmpeg
    capture=cv2.VideoCapture(str(source))
    ok,frame=capture.read();capture.release()
    if ok:cv2.imwrite(str(folder/'poster.jpg'),frame)
    # Browser-compatible H.264 needs an even width as well as an even height.
    preview_scale = "scale=w='max(2,trunc(min(1280,iw)/2)*2)':h=-2"
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-y','-i',str(source),'-map','0:v:0','-map','0:a?',
        '-vf',preview_scale,'-c:v','libx264','-preset','veryfast','-crf','23','-pix_fmt','yuv420p',
        '-c:a','aac','-b:a','96k','-movflags','+faststart',str(folder/'preview.mp4')]
    result=subprocess.run(command,capture_output=True,text=True,timeout=600)
    if result.returncode or not (folder/'preview.mp4').is_file():
        (folder/'preview-error.txt').write_text(result.stderr)
        raise ValueError('This video could not be prepared for playback. Try an MP4 clip.')


def infer_video(project_root, source, folder, progress):
    command=[sys.executable,'-u',str(project_root/'detect.py'),str(source),'--output',str(folder/'prediction.json')]
    with (folder/'analysis.log').open('w') as log:
        process=subprocess.Popen(command,cwd=project_root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        try:
            for line in process.stdout:
                log.write(line);log.flush()
                match=re.search(r'Scored (\d+)/(\d+) video windows',line)
                if match:
                    current,total=map(int,match.groups())
                    progress(12+int(current/max(total,1)*80),f'Analysing video window {current} of {total}')
            code=process.wait()
        finally:
            if process.poll() is None:process.terminate();process.wait(timeout=10)
    if code:
        raise ValueError('The model could not analyse this clip. Check that the file is a playable video; details are saved in its analysis log.')
    return json.loads((folder/'prediction.json').read_text())


def create_app(project_root=ROOT, storage_root=None, runner=None, previewer=None):
    project_root=Path(project_root).resolve()
    app=Flask(__name__,template_folder=str(project_root/'ui'),static_folder=str(project_root/'ui'),static_url_path='/assets')
    app.config.update(MAX_CONTENT_LENGTH=512*1024*1024, REVIEW_TOKEN=secrets.token_urlsafe(32), TRUSTED_HOSTS=['127.0.0.1','localhost','[::1]'])
    store=JobStore(storage_root or project_root/'outputs/ui')
    app.extensions['review_store']=store
    runner=runner or (lambda source,folder,progress:infer_video(project_root,source,folder,progress))
    previewer=previewer or browser_preview
    sample_base=project_root/'data/real_retail/ucf_crime'
    samples={
        'retail-8s':('Retail example · 8 seconds',sample_base/'context/clips/Shoplifting/d686f807c4bc0622-context-positive-569-810.mp4'),
        'retail-19s':('Retail example · 19 seconds',sample_base/'context/clips/Shoplifting/7af64fd7e96f81da-context-positive-1489-2060.mp4'),
        'short-4s':('Short example · 4 seconds',sample_base/'clips/Normal/7af64fd7e96f81da-normal-0-120.mp4'),
    }

    def public(job):
        result=job.get('result')
        return {key:job.get(key) for key in ['id','filename','display_name','state','progress','message','created_at','duration_seconds','suggestions','reviews','source_kind']} | {
            'result':None if not result else {key:result.get(key) for key in ['status','result_text','video_score','duration_seconds','review_threshold','decision']},
            'preview_url':f'/api/jobs/{job["id"]}/media/preview' if job.get('preview_ready') else None,
            'poster_url':f'/api/jobs/{job["id"]}/media/poster' if job.get('preview_ready') else None,
            'report_url':f'/api/jobs/{job["id"]}/report' if job['state']=='complete' else None,
            'windows':[] if not result else result.get('windows',[]),
        }

    def get_job(job_id):
        try:return store.get(job_id)
        except KeyError:abort(404,description='Analysis not found.')

    def work(job_id, source):
        folder=store.root/job_id
        try:
            store.update(job_id,state='running',message='Checking video and preparing playback',progress=3)
            info=video_info(source)
            store.update(job_id,duration_seconds=info['duration'])
            previewer(source,folder)
            store.update(job_id,preview_ready=True,progress=12,message='Starting video analysis')
            result=runner(source,folder,lambda value,message:store.update(job_id,progress=value,message=message))
            store.update(job_id,progress=95,message='Finding time ranges to review')
            suggestions=build_suggestions(result)
            digest=hashlib.sha256()
            with source.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
            store.update(job_id,state='complete',progress=100,message='Analysis complete. Review the suggested time ranges.',
                result=result,suggestions=suggestions,input_sha256=digest.hexdigest(),completed_at=now())
        except Exception as error:
            (folder/'error.txt').write_text(f'{type(error).__name__}: {error}\n')
            store.update(job_id,state='failed',message=str(error),progress=0)

    @app.before_request
    def protect_mutations():
        if request.method in {'POST','PUT','DELETE','PATCH'} and not secrets.compare_digest(request.headers.get('X-Review-Token',''),app.config['REVIEW_TOKEN']):
            abort(403,description='Please refresh the page before submitting.')

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description),error.code

    @app.get('/')
    def index():
        return render_template('index.html',review_token=app.config['REVIEW_TOKEN'])

    @app.get('/api/health')
    def health():
        return jsonify(status='ok',service='retail-review-local',project_key=hashlib.sha256(str(project_root).encode()).hexdigest())

    @app.get('/api/samples')
    def sample_list():
        return jsonify(samples=[{'id':key,'name':name} for key,(name,path) in samples.items() if path.is_file()])

    @app.get('/api/jobs')
    def jobs():
        with store.lock:
            data=sorted(store.jobs.values(),key=lambda x:x['created_at'],reverse=True)[:20]
            return jsonify(jobs=[public(x) for x in data])

    @app.post('/api/analyses')
    def analyse():
        upload=request.files.get('video'); sample_id=request.form.get('sample_id')
        if bool(upload) == bool(sample_id):abort(400,description='Choose one video file or one example clip.')
        if upload:
            filename=secure_filename(upload.filename or '')
            suffix=Path(filename).suffix.lower()
            if not filename or suffix not in EXTENSIONS:abort(400,description='Choose an MP4, MOV, AVI, MKV, MPEG or WMV video.')
        else:
            if sample_id not in samples or not samples[sample_id][1].is_file():abort(400,description='That example is unavailable.')
            filename=samples[sample_id][1].name;suffix=Path(filename).suffix.lower()
        job_id=uuid.uuid4().hex;folder=store.root/job_id;folder.mkdir()
        source=folder/('input'+suffix)
        if upload:upload.save(source)
        else:shutil.copy2(samples[sample_id][1],source)
        if not source.stat().st_size:
            shutil.rmtree(folder);abort(400,description='The selected video is empty.')
        job={'id':job_id,'filename':filename,'display_name':samples[sample_id][0] if sample_id else filename,'state':'queued','progress':0,'message':'Waiting for local analysis',
             'created_at':now(),'duration_seconds':None,'suggestions':[],'reviews':{},'source_kind':'example' if sample_id else 'upload','preview_ready':False}
        with store.lock:store.jobs[job_id]=job;atomic_json(folder/'job.json',job)
        store.executor.submit(work,job_id,source)
        return jsonify(public(store.get(job_id))),202

    @app.get('/api/jobs/<job_id>')
    def status(job_id):return jsonify(public(get_job(job_id)))

    @app.get('/api/jobs/<job_id>/media/<kind>')
    def media(job_id,kind):
        job=get_job(job_id)
        names={'preview':('preview.mp4','video/mp4'),'poster':('poster.jpg','image/jpeg')}
        if kind not in names or not job.get('preview_ready'):abort(404)
        name,mime=names[kind];path=store.root/job_id/name
        if not path.is_file():abort(404)
        return send_file(path,mimetype=mime,conditional=True)

    @app.post('/api/jobs/<job_id>/review')
    def review(job_id):
        job=get_job(job_id)
        if job['state']!='complete':abort(409,description='Wait for the analysis to finish.')
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            abort(400, description='Send the review as a JSON object.')
        choice = data.get('decision')
        segment_id = data.get('segment_id')
        if not isinstance(choice,str) or choice not in REVIEW_CHOICES:abort(400,description='Choose suspected action, not visible, or unclear.')
        if segment_id is not None and (not isinstance(segment_id, str) or not segment_id):
            abort(400, description='Select a valid suggested segment or review the full clip.')
        segment=next((s for s in job['suggestions'] if str(s['id'])==str(segment_id)),None) if segment_id else None
        if segment_id and not segment:abort(400,description='Select a valid suggested segment.')
        notes=data.get('notes','')
        if not isinstance(notes,str) or len(notes)>2000:abort(400,description='Keep review notes within 2,000 characters.')
        entry={'decision':choice,'notes':notes,'segment_id':str(segment_id) if segment_id else None,
            'start_seconds':segment['start_seconds'] if segment else 0,
            'end_seconds':segment['end_seconds'] if segment else job['duration_seconds'],
            'saved_at':now(),'source':'human_review','input_sha256':job.get('input_sha256')}
        with store.lock:
            reviews=store.get(job_id)['reviews'];reviews[str(segment_id) if segment_id else 'whole_clip']=entry
            store.update(job_id,reviews=reviews)
        return jsonify(review=entry,reviews=reviews)

    @app.get('/api/jobs/<job_id>/report')
    def report(job_id):
        job=get_job(job_id)
        if job['state']!='complete':abort(409)
        payload={**job,'interpretation':'Model suggestions are unconfirmed review prompts. Human reviews are recorded separately and are not automatic training labels.'}
        return send_file(io.BytesIO((json.dumps(payload,indent=2,allow_nan=False)+'\n').encode()),mimetype='application/json',as_attachment=True,download_name=f'retail-review-{job_id[:8]}.json')

    return app


def existing_app(url):
    try:
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url+'/api/health',timeout=.5) as response:
            data=json.load(response)
        return data.get('service')=='retail-review-local' and data.get('project_key')==hashlib.sha256(str(ROOT).encode()).hexdigest()
    except (OSError,ValueError):
        return False


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8765);parser.add_argument('--no-open',action='store_true')
    args=parser.parse_args(argv)
    if not 1024<=args.port<=65535:parser.error('Port must be between 1024 and 65535.')
    url=f'http://127.0.0.1:{args.port}'
    if existing_app(url):
        print(f'Retail Review UI is already running: {url}',flush=True)
        if not args.no_open:webbrowser.open(url)
        return 0
    app=create_app()
    from werkzeug.serving import make_server
    server=make_server('127.0.0.1',args.port,app,threaded=True)
    print(f'Retail Review UI: {url}',flush=True)
    if not args.no_open:threading.Timer(.4,lambda:webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.extensions['review_store'].shutdown()
    return 0
