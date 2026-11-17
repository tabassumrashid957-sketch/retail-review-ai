'use strict';
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="review-token"]').content;
const state = {file:null,sample:null,job:null,busy:false,poll:null,segment:null,drafts:{}};
const video = $('player');
const show = (id,on=true) => $(id).classList.toggle('hidden',!on);
const formatTime = value => {const n=Math.max(0,Number(value)||0);return `${String(Math.floor(n/60)).padStart(2,'0')}:${(n%60).toFixed(1).padStart(4,'0')}`;};
async function api(path, options={}) {
  const headers={'Accept':'application/json',...(options.headers||{})};
  if(options.method && options.method!=='GET') headers['X-Review-Token']=token;
  const response=await fetch(path,{...options,headers});
  const body=await response.json();
  if(!response.ok)throw new Error(body.error||'The request could not finish.');
  return body;
}
function error(message){$('error').textContent=message;show('error',!!message);}
function busy(value){state.busy=value;$('analyse').disabled=value||(!state.file&&!state.sample);$('browse').disabled=value;$('file-input').disabled=value;$('new-clip').disabled=value;document.querySelectorAll('#samples button').forEach(b=>b.disabled=value);}
function step(number){['upload','analysis','review'].forEach((name,i)=>{const element=$(`step-${name}`);element.classList.toggle('active',i+1===number);element.classList.toggle('done',i+1<number);});}
function chooseFile(file){
  if(state.busy||!file)return;
  if(file.size>512*1024*1024){error('Choose a video smaller than 512 MB.');return;}
  if(!/\.(mp4|mov|avi|mkv|mpeg|mpg|wmv)$/i.test(file.name)){error('Choose an MP4, MOV, AVI, MKV, MPEG or WMV video.');return;}
  resetView();state.file=file;state.sample=null;
  $('selected-name').textContent=file.name;$('selected-detail').textContent=`${(file.size/1024/1024).toFixed(1)} MB · Ready for analysis`;
  document.querySelectorAll('#samples button').forEach(b=>b.classList.remove('selected'));
  busy(false);error('');
}
function resetView(){
  clearTimeout(state.poll);state.job=null;state.segment=null;state.drafts={};
  video.pause();video.removeAttribute('src');video.removeAttribute('poster');delete video.dataset.url;video.load();
  ['progress-panel','result-banner','review-panel','playback-controls','signal-panel','analysis-details','player'].forEach(id=>show(id,false));
  show('player-empty');$('duration-label').textContent='No clip selected';$('segment-count').textContent='0';
  $('segments').innerHTML='<div class="empty-ranges"><span>◷</span><strong>Time ranges will appear here</strong><p>Suggestions are calculated from the uploaded video.</p></div>';
  $('segment-intro').textContent='After analysis, select a range to play that part of your clip.';
  $('review-form').reset();$('review-status').textContent='';step(1);error('');
}
$('browse').addEventListener('click',()=>$('file-input').click());
$('file-input').addEventListener('change',event=>chooseFile(event.target.files[0]));
['dragenter','dragover'].forEach(type=>$('dropzone').addEventListener(type,event=>{event.preventDefault();if(!state.busy)$('dropzone').classList.add('drag');}));
['dragleave','drop'].forEach(type=>$('dropzone').addEventListener(type,event=>{event.preventDefault();$('dropzone').classList.remove('drag');if(type==='drop')chooseFile(event.dataTransfer.files[0]);}));
$('new-clip').addEventListener('click',()=>{if(state.busy)return;resetView();state.file=null;state.sample=null;$('file-input').value='';$('selected-name').textContent='Drop a video here, or browse files';$('selected-detail').textContent='Choose the original clip you want to review.';document.querySelectorAll('#samples button').forEach(b=>b.classList.remove('selected'));busy(false);window.scrollTo({top:0,behavior:'smooth'});});
async function loadSamples(){
  try {const data=await api('/api/samples');data.samples.forEach(sample=>{
    const button=document.createElement('button');button.type='button';button.textContent=sample.name;button.dataset.sample=sample.id;
    button.addEventListener('click',()=>{if(state.busy)return;resetView();state.file=null;state.sample=sample.id;$('selected-name').textContent=sample.name;$('selected-detail').textContent='Included research clip · Ready for analysis';document.querySelectorAll('#samples button').forEach(b=>b.classList.toggle('selected',b===button));busy(false);});$('samples').append(button);
  });}catch(e){error(e.message);}
}
$('analyse').addEventListener('click',async()=>{
  if(state.busy||(!state.file&&!state.sample))return;
  const form=new FormData();if(state.file)form.append('video',state.file);else form.append('sample_id',state.sample);
  resetView();
  busy(true);show('progress-panel');step(2);error('');$('progress-title').textContent='Analysing your clip';$('progress-message').textContent='Saving your clip locally…';$('progress-fill').style.width='1%';
  try {const job=await api('/api/analyses',{method:'POST',body:form});state.job=job;await pollJob(job.id);refreshHistory();}
  catch(e){error(e.message);busy(false);show('progress-panel',false);step(1);}
});
async function pollJob(id){
  try {
    const job=await api(`/api/jobs/${id}`);if(state.job?.id!==id)return;state.job=job;renderJob(job);
    if(job.state==='queued'||job.state==='running')state.poll=setTimeout(()=>pollJob(id),850);
    else{busy(false);refreshHistory();}
  }catch(e){error(e.message);busy(false);show('progress-panel',false);}
}
function renderJob(job){
  const running=job.state==='queued'||job.state==='running';show('progress-panel',running);
  $('progress-title').textContent=job.state==='queued'?'Waiting for analysis':'Analysing your clip';$('progress-message').textContent=job.message||'';
  $('progress-number').textContent=`${job.progress||0}%`;$('progress-fill').style.width=`${job.progress||0}%`;
  if(job.duration_seconds)$('duration-label').textContent=`${formatTime(job.duration_seconds)} · Original footage`;
  if(job.preview_url && video.dataset.url!==job.preview_url){video.src=job.preview_url;video.dataset.url=job.preview_url;video.poster=job.poster_url||'';video.load();show('player');show('player-empty',false);show('playback-controls');$('playback-time').textContent=`00:00.0 / ${formatTime(job.duration_seconds)}`;}
  if(job.state==='failed'){error(job.message);step(1);return;}
  if(running){step(2);return;}
  if(job.state==='complete'){step(3);renderResult(job);}
}
function renderResult(job){
  const status=job.result.status;
  const messages={
    needs_review:['Possible concealment to review','The model found a signal worth checking. Play the suggested ranges and record what you can see.','!'],
    insufficient_evidence:['Cannot determine from this clip','The model cannot decide confidently. Any suggested ranges are unconfirmed places to inspect.','?'],
    no_review_flag:['No suspicious action flagged','No strong review signal was found. You can still inspect the footage and record your assessment.','✓']
  };
  const [title,message,symbol]=messages[status]||messages.insufficient_evidence;
  $('result-title').textContent=title;$('result-message').textContent=message;$('result-symbol').textContent=symbol;
  $('result-banner').classList.toggle('flagged',status==='needs_review');$('result-banner').classList.toggle('clear',status==='no_review_flag');show('result-banner');
  $('segment-count').textContent=String(job.suggestions.length);$('segment-intro').textContent=job.suggestions.length?'Select a range to play it. Then record whether an action is visible.':'No distinctive time range was found. Review the full clip to make your assessment.';
  $('segments').replaceChildren();
  if(!job.suggestions.length){$('segments').innerHTML='<div class="empty-ranges"><span>◷</span><strong>Review the full clip</strong><p>The model did not identify a reliable or distinctive time range.</p></div>';}
  job.suggestions.forEach((segment,index)=>{
    const button=document.createElement('button');button.type='button';button.className='segment';button.dataset.segment=String(segment.id);button.setAttribute('aria-label',`Play suggested range ${index+1}: ${formatTime(segment.start_seconds)} to ${formatTime(segment.end_seconds)}`);
    const badge=document.createElement('span');badge.className='segment-index';badge.textContent=String(index+1);
    const text=document.createElement('span');text.className='segment-text';const strong=document.createElement('strong');strong.textContent=`${formatTime(segment.start_seconds)} – ${formatTime(segment.end_seconds)}`;
    const small=document.createElement('small');small.textContent=segment.kind==='flagged'?'Model flag · needs human review':segment.kind==='whole_clip'?'Short clip · inspect the complete action':'Unconfirmed model suggestion';
    const arrow=document.createElement('span');arrow.className='segment-arrow';arrow.textContent='▷';text.append(strong,small);button.append(badge,text,arrow);button.addEventListener('click',()=>playRange(segment));$('segments').append(button);
  });
  $('analysis-explanation').textContent=job.result.decision?.message||'No detailed explanation available.';show('analysis-details');
  show('review-panel');$('download').href=job.report_url;drawSignal(job);loadReview();
}
function rememberDraft(){if(!state.job)return;const selected=document.querySelector('input[name="decision"]:checked');state.drafts[state.segment?.id||'whole_clip']={decision:selected?.value||'',notes:$('notes').value};}
function loadReview(){
  const key=state.segment?.id||'whole_clip';const saved=state.job?.reviews?.[key];const data=state.drafts[key]||saved||{};
  document.querySelectorAll('input[name="decision"]').forEach(input=>input.checked=input.value===data.decision);$('notes').value=data.notes||'';$('review-status').textContent=saved?'Saved review loaded. You can update it.':'';
  $('review-scope').textContent=state.segment?`${formatTime(state.segment.start_seconds)}–${formatTime(state.segment.end_seconds)}`:'Full clip';
}
function playRange(segment){
  rememberDraft();state.segment=segment;loadReview();document.querySelectorAll('.segment').forEach(button=>button.classList.toggle('selected',button.dataset.segment===String(segment.id)));
  $('playback-label').textContent=`Suggested range · ${formatTime(segment.start_seconds)}–${formatTime(segment.end_seconds)}`;
  const play=()=>{video.currentTime=segment.start_seconds;video.play().catch(()=>{});};
  if(video.readyState>=1)play();else video.addEventListener('loadedmetadata',play,{once:true});drawSignal(state.job);
}
$('watch-full').addEventListener('click',()=>{rememberDraft();state.segment=null;loadReview();document.querySelectorAll('.segment').forEach(b=>b.classList.remove('selected'));$('playback-label').textContent='Full clip';video.currentTime=0;video.play().catch(()=>{});drawSignal(state.job);});
video.addEventListener('timeupdate',()=>{if(state.segment && video.currentTime>=state.segment.end_seconds){video.pause();}const duration=state.job?.duration_seconds||video.duration||0;$('playback-time').textContent=`${formatTime(video.currentTime)} / ${formatTime(duration)}`;});
function drawSignal(job){
  const chart=$('signal-chart');chart.replaceChildren();const windows=job.windows||[];const duration=job.duration_seconds||1;
  if(!windows.length){show('signal-panel',false);return;}
  const ns='http://www.w3.org/2000/svg';
  if(state.segment){const rect=document.createElementNS(ns,'rect');rect.setAttribute('x',String(state.segment.start_seconds/duration*700));rect.setAttribute('width',String((state.segment.end_seconds-state.segment.start_seconds)/duration*700));rect.setAttribute('y','0');rect.setAttribute('height','65');rect.setAttribute('fill','#e5edc8');chart.append(rect);}
  const points=windows.map(w=>`${((w.start_seconds+w.end_seconds)/2)/duration*700},${60-Math.max(0,Math.min(1,w.review_score))*52}`).join(' ');
  const line=document.createElementNS(ns,'polyline');line.setAttribute('points',points);line.setAttribute('fill','none');line.setAttribute('stroke','#4e8561');line.setAttribute('stroke-width','2');chart.append(line);$('chart-end').textContent=formatTime(duration);show('signal-panel');
}
$('signal-chart').addEventListener('click',event=>{if(!state.job)return;rememberDraft();state.segment=null;loadReview();$('playback-label').textContent='Full clip';document.querySelectorAll('.segment').forEach(b=>b.classList.remove('selected'));video.currentTime=Math.max(0,Math.min(1,(event.clientX-event.currentTarget.getBoundingClientRect().left)/event.currentTarget.getBoundingClientRect().width))*state.job.duration_seconds;drawSignal(state.job);});
$('review-form').addEventListener('submit',async event=>{
  event.preventDefault();const chosen=document.querySelector('input[name="decision"]:checked');
  if(!chosen){$('review-status').textContent='Choose an assessment before saving.';return;}
  $('save-review').disabled=true;$('review-status').textContent='Saving…';
  try {const data=await api(`/api/jobs/${state.job.id}/review`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({decision:chosen.value,notes:$('notes').value,segment_id:state.segment?.id||null})});state.job.reviews=data.reviews;rememberDraft();$('review-status').textContent='Review saved on this laptop.';refreshHistory();}
  catch(e){$('review-status').textContent=e.message;}finally{$('save-review').disabled=false;}
});
async function refreshHistory(){
  try {const data=await api('/api/jobs');$('history').replaceChildren();if(!data.jobs.length){const p=document.createElement('p');p.textContent='No clips analysed yet.';$('history').append(p);}
    data.jobs.slice(0,6).forEach(job=>{const button=document.createElement('button');button.type='button';button.classList.toggle('selected',job.id===state.job?.id);const strong=document.createElement('strong');strong.textContent=job.display_name||job.filename;const small=document.createElement('small');small.textContent=`${job.state==='complete'?'Ready to review':job.state} · ${new Date(job.created_at).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})}`;button.append(strong,small);button.addEventListener('click',()=>{if(state.busy)return;resetView();state.job=job;busy(job.state==='running'||job.state==='queued');pollJob(job.id);});$('history').append(button);});
  }catch(e){error(e.message);}
}
$('refresh-history').addEventListener('click',refreshHistory);
loadSamples();refreshHistory();
