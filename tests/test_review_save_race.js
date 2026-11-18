'use strict';
// Exercise the real browser submit handler with deferred HTTP responses.
// No browser, server, video, private case or network connection is used.
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const {test}=require('node:test');

function fixture(){
  const elements=new Map(),requests=[];
  const radios=['suspected','not_visible','unclear'].map(value=>({value,checked:false}));
  function element(id){
    if(!elements.has(id))elements.set(id,{
      id,value:'',textContent:'',disabled:false,handlers:{},dataset:{},style:{},readyState:1,
      classList:{toggle(){},add(){},remove(){}},
      addEventListener(type,handler){this.handlers[type]=handler;},
      replaceChildren(){},append(){},reset(){},pause(){},load(){},
      play(){return Promise.resolve();},removeAttribute(){},
      getBoundingClientRect(){return {left:0,width:700};}
    });
    return elements.get(id);
  }
  const context=vm.createContext({
    console,setTimeout,clearTimeout,FormData:class {},window:{scrollTo(){}},
    document:{
      getElementById:element,
      querySelector(selector){return selector.startsWith('meta')?{content:'test-token'}:radios.find(r=>r.checked)||null;},
      querySelectorAll(selector){return selector==='input[name="decision"]'?radios:[];},
      createElement(){return element('created');},createElementNS(){return element('svg');}
    },
    fetch:async(url,options)=>{
      if(url.endsWith('/review'))return new Promise((resolve,reject)=>requests.push({
        url,payload:JSON.parse(options.body),resolve:reviews=>resolve({ok:true,json:async()=>({reviews})}),reject
      }));
      return {ok:true,json:async()=>url==='/api/samples'?{samples:[]}:{jobs:[]}};
    }
  });
  vm.runInContext(fs.readFileSync(path.resolve(__dirname,'../ui/app.js'),'utf8'),context);
  function evaluate(source){return vm.runInContext(source,context);}
  function select(jobId,segmentId=null){
    evaluate(`state.job={id:${JSON.stringify(jobId)},state:'complete',reviews:{},windows:[],duration_seconds:10};state.segment=${segmentId?JSON.stringify({id:segmentId,start_seconds:1,end_seconds:3}):'null'};state.drafts={};`);
  }
  function edit(notes,decision='suspected'){
    radios.forEach(r=>r.checked=r.value===decision);element('notes').value=notes;
  }
  return {element,requests,evaluate,select,edit,
    submit:()=>element('review-form').handlers.submit({preventDefault(){}}),
    state:()=>JSON.parse(evaluate('JSON.stringify(state)'))
  };
}

test('same-case success stores the submitted scope and payload',async()=>{
  const f=fixture();f.select('case-a','segment-a');f.edit('Submitted A');
  const pending=f.submit();
  assert.deepEqual(f.requests[0].payload,{decision:'suspected',notes:'Submitted A',segment_id:'segment-a'});
  f.requests[0].resolve({'segment-a':{notes:'Submitted A'}});await pending;
  assert.deepEqual(f.state().job.reviews,{'segment-a':{notes:'Submitted A'}});
  assert.deepEqual(f.state().drafts['segment-a'],{decision:'suspected',notes:'Submitted A'});
  assert.match(f.element('review-status').textContent,/saved on this laptop/);
  assert.equal(f.element('save-review').disabled,false);
});

test('a response for case A never changes case B or its draft/status',async()=>{
  const f=fixture();f.select('case-a');f.edit('Submitted A');const pending=f.submit();
  f.evaluate('resetView()');f.select('case-b');f.edit('Unsaved B','unclear');
  f.evaluate("state.job.reviews={whole_clip:{notes:'Saved B'}};state.drafts={whole_clip:{notes:'Draft B'}}");
  f.element('review-status').textContent='Case B status';
  f.requests[0].resolve({whole_clip:{notes:'Submitted A'}});await pending;
  assert.equal(f.state().job.id,'case-b');
  assert.deepEqual(f.state().job.reviews,{whole_clip:{notes:'Saved B'}});
  assert.deepEqual(f.state().drafts,{whole_clip:{notes:'Draft B'}});
  assert.equal(f.element('notes').value,'Unsaved B');
  assert.equal(f.element('review-status').textContent,'Case B status');
});

test('clearing the current clip before success leaves the empty view intact',async()=>{
  const f=fixture();f.select('case-a');f.edit('Submitted A');const pending=f.submit();
  f.evaluate('resetView()');
  f.requests[0].resolve({whole_clip:{notes:'Submitted A'}});await pending;
  assert.equal(f.state().job,null);assert.deepEqual(f.state().drafts,{});
  assert.equal(f.element('review-status').textContent,'');
  assert.equal(f.element('save-review').disabled,false);
});

test('segment navigation never saves the new segment as the old response draft',async()=>{
  const f=fixture();f.select('case-a','segment-a');f.edit('Submitted A');const pending=f.submit();
  f.evaluate("playRange({id:'segment-b',start_seconds:4,end_seconds:6})");
  f.edit('Unsaved B','unclear');f.element('review-status').textContent='Editing B';
  f.requests[0].resolve({'segment-a':{notes:'Submitted A'}});await pending;
  const state=f.state();assert.equal(state.segment.id,'segment-b');
  assert.deepEqual(state.job.reviews,{'segment-a':{notes:'Submitted A'}});
  assert.deepEqual(state.drafts['segment-a'],{decision:'suspected',notes:'Submitted A'});
  assert.equal(state.drafts['segment-b'],undefined);
  assert.equal(f.element('notes').value,'Unsaved B');
  assert.equal(f.element('review-status').textContent,'Editing B');
});

test('an old-case error cannot overwrite the new case status',async()=>{
  const f=fixture();f.select('case-a');f.edit('Submitted A');const pending=f.submit();
  f.select('case-b');f.element('review-status').textContent='Case B status';
  f.requests[0].reject(new Error('Case A save failed'));await pending;
  assert.equal(f.element('review-status').textContent,'Case B status');
  assert.deepEqual(f.state().job.reviews,{});
});

test('an earlier completion cannot re-enable an overlapping newer save',async()=>{
  const f=fixture();f.select('case-a');f.edit('Submitted A');const first=f.submit();
  f.select('case-b');f.edit('Submitted B');const second=f.submit();
  f.requests[0].resolve({whole_clip:{notes:'Submitted A'}});await first;
  assert.equal(f.element('save-review').disabled,true);
  assert.deepEqual(f.state().job.reviews,{});
  f.requests[1].resolve({whole_clip:{notes:'Submitted B'}});await second;
  assert.equal(f.element('save-review').disabled,false);
  assert.deepEqual(f.state().job.reviews,{whole_clip:{notes:'Submitted B'}});
});
