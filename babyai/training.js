
let current=null, runningNow=false;const q=s=>document.querySelector(s);async function post(path,body={}){try{let r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),d=await r.json();if(!r.ok)throw Error(d.error);q('#error').textContent='';await saves()}catch(e){q('#error').textContent=e.message}}
function createSession(){let s={};for(let k of ['env_id','agent','mission','ollama_url','model','grag_entry'])s[k]=q('#'+k).value;s.cpu_workers=+q('#cpu-workers').value;s.seed=+q('#seed').value;s.max_steps=+q('#max_steps').value;try{s.goal=q('#goal').value.trim()?JSON.parse(q('#goal').value):null;post('/api/new',s)}catch(e){q('#error').textContent=e.message}}
async function saves(){let d=await(await fetch('/api/saves')).json(),old=q('#saves').value;q('#saves').replaceChildren(...d.saves.map(id=>{let o=document.createElement('option');o.value=o.textContent=id;return o}));q('#saves').value=d.saves.includes(old)?old:d.saves[0]||''}
const colors={red:'#f26363',green:'#63c781',blue:'#629df5',purple:'#c48cec',yellow:'#f4dc72',grey:'#8b99a8'};const names=['unseen','empty','wall','floor','door','key','ball','box','goal','lava','agent'];const cs=['red','green','blue','purple','yellow','grey'];
function grid(canvas,w,h,items,pos,dir){
 const c=canvas.getContext('2d'),size=Math.min(canvas.width/w,canvas.height/h);
 c.clearRect(0,0,canvas.width,canvas.height);
 for(let x=0;x<w;x++)for(let y=0;y<h;y++){c.fillStyle='#ffffff';c.fillRect(x*size,y*size,size-1,size-1)}
 for(const o of items){
  const [x,y]=o.position,s=size,px=x*s,py=y*s,color=colors[o.color]||'#8497a1';
  c.fillStyle=color;c.strokeStyle=color;c.lineWidth=Math.max(2,s*.06);
  if(o.type==='unseen'||o.type==='wall'){c.fillStyle='#334b59';c.fillRect(px,py,s-1,s-1);if(o.type==='unseen'){c.fillStyle='#8298a5';c.font=s*.35+'px sans-serif';c.fillText('?',px+s*.4,py+s*.65)}}
  else if(o.type==='ball'){c.beginPath();c.arc(px+s/2,py+s/2,s*.28,0,Math.PI*2);c.fill()}
  else if(o.type==='box'){c.fillRect(px+s*.2,py+s*.2,s*.6,s*.6);c.strokeStyle='#354f60';c.strokeRect(px+s*.2,py+s*.2,s*.6,s*.6);c.beginPath();c.moveTo(px+s*.2,py+s*.2);c.lineTo(px+s*.8,py+s*.8);c.moveTo(px+s*.8,py+s*.2);c.lineTo(px+s*.2,py+s*.8);c.stroke()}
  else if(o.type==='key'){c.beginPath();c.arc(px+s*.35,py+s*.35,s*.13,0,Math.PI*2);c.stroke();c.beginPath();c.moveTo(px+s*.44,py+s*.44);c.lineTo(px+s*.75,py+s*.75);c.lineTo(px+s*.8,py+s*.64);c.moveTo(px+s*.61,py+s*.61);c.lineTo(px+s*.7,py+s*.53);c.stroke()}
  else if(o.type==='door'){if(!o.is_open)c.fillRect(px+s*.15,py+s*.1,s*.7,s*.8);else c.strokeRect(px+s*.15,py+s*.1,s*.7,s*.8);c.fillStyle=o.is_open?color:'#172a39';c.font='bold '+s*.4+'px sans-serif';c.fillText(o.is_open?'O':'D',px+s*.35,py+s*.65)}
  else if(o.type!=='empty'&&o.type!=='floor'){c.fillRect(px+s*.1,py+s*.1,s*.8,s*.8)}
 }
 if(pos){let [x,y]=pos;c.save();c.translate((x+.5)*size,(y+.5)*size);c.rotate(dir*Math.PI/2);c.fillStyle='#087f79';c.strokeStyle='white';c.lineWidth=2;c.beginPath();c.moveTo(size*.34,0);c.lineTo(-size*.25,-size*.25);c.lineTo(-size*.16,0);c.lineTo(-size*.25,size*.25);c.closePath();c.fill();c.stroke();c.restore()}
}
function render(s,running){current=s;runningNow=running;if(!s)return;renderLLM(s);renderOrientation(s.observation.direction);let h=s.history,last=h.slice(-100),rate=last.length?last.filter(x=>x.success).length/last.length:0;q('#status').textContent=`${running?'\u5b9f\u884c\u4e2d':'\u505c\u6b62\u4e2d'} | ${s.episodes} \u8a66\u884c | ${s.total_steps} steps | \u6210\u529f\u7387 ${(rate*100).toFixed(1)}% | ${s.steps} steps | ${s.action||'-'}`;q('#task').textContent=s.observation.mission;
q('#phase-badge').textContent=`\u5b66\u7fd2${s.episodes}\u56de / ${running?'\u5b9f\u884c\u4e2d':'\u505c\u6b62\u4e2d'}`;
q('#live-indicator').textContent=running?'\u5b9f\u884c\u4e2d':'\u5f85\u6a5f';
q('#stat-episodes').textContent=s.episodes.toLocaleString();
q('#stat-success').textContent=last.length?(rate*100).toFixed(1)+'%':'?';
q('#stat-reward').textContent=last.length?(last.reduce((a,b)=>a+b.reward,0)/last.length).toFixed(3):'?';
q('#stat-steps').textContent=s.total_steps.toLocaleString();let o=s.world.objects,w=Math.max(...o.map(x=>x.position[0]))+1,ht=Math.max(...o.map(x=>x.position[1]))+1;grid(q('#world'),w,ht,o,s.world.position,s.world.direction);let im=s.observation.image,items=[];im.forEach((col,x)=>col.forEach((cell,y)=>{if(cell[0]!==1)items.push({type:names[cell[0]],color:cs[cell[1]],position:[x,y],is_open:cell[2]===0})}));grid(q('#view'),im.length,im[0].length,items,[Math.floor(im.length/2),im[0].length-1],3);let c=q('#curve').getContext('2d'),cw=c.canvas.width,ch=c.canvas.height;c.clearRect(0,0,cw,ch);if(!h.length)return;let vals=h.map((_,i)=>h.slice(Math.max(0,i-99),i+1).reduce((a,b)=>a+b.reward,0)/Math.min(i+1,100)),max=Math.max(1,...vals);c.strokeStyle='#087f79';c.beginPath();vals.forEach((v,i)=>{let x=40+i/Math.max(1,vals.length-1)*(cw-60),y=ch-30-v/max*(ch-50);i?c.lineTo(x,y):c.moveTo(x,y)});c.stroke();c.fillStyle='#607785';c.fillText('0',5,ch-30);c.fillText(max.toFixed(2),5,20);c.fillText(`${h[0].episode} - ${h.at(-1).episode} episodes`,40,ch-8)}
async function poll(){try{let d=await(await fetch('/api/state')).json();await syncLLMLogs(d.session);render(d.session,d.running);if(d.error)q('#error').textContent=d.error}catch(e){q('#error').textContent=e.message}setTimeout(poll,runningNow?Math.min(160,+q('#observation-speed').value):600)}
function download(){if(!current)return;let text='episode,reward,steps,success,total_steps\n'+current.history.map(r=>[r.episode,r.reward,r.steps,r.success,r.total_steps].join(',')).join('\n');let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{type:'text/csv'}));a.download='babyai-results.csv';a.click();URL.revokeObjectURL(a.href)}saves();poll();

let logSession=null, logEntries=[], llmHistoryKey='';
const followingOutputs=new WeakMap();
function followingOutput(el){
 if(!followingOutputs.has(el))followingOutputs.set(el,BabyAILogFollow(el));
 return followingOutputs.get(el);
}
function followText(el,text){followingOutput(el).setText(text)}
async function syncLLMLogs(s){
 if(!s)return;
 if(logSession!==s.id){logSession=s.id;logEntries=[];llmHistoryKey='';for(const id of ['#llm-thinking','#llm-content'])followingOutput(q(id)).reset();q('#llm-history').replaceChildren()}
 if((s.llm_log_count??s.llm_history?.length??0)<=logEntries.length)return;
 const data=await(await fetch('/api/llm-logs?after='+logEntries.length)).json();
 if(data.id===s.id)logEntries.push(...data.logs);
}

function originalThought(r){return r.thinking||'\u601d\u8003\u51fa\u529b\u5f85\u3061'}
function renderLLM(s){
 const info=s.execution||{mode:'serial',workers:1};
 q('#execution-info').textContent=`${info.mode==='parallel'?'PPO parallel rollout':'Single environment'}: ${info.workers} worker(s)`;
 q('#llm-panel').hidden=s.settings.agent!=='ollama';
 const t=s.llm||{};
 const labels={translating:'\u65e5\u672c\u8a9e\u306b\u7ffb\u8a33\u4e2d',requesting:'\u63a5\u7d9a\u4e2d',thinking:'\u601d\u8003\u3092\u751f\u6210\u4e2d',generating:'\u884c\u52d5\u3092\u751f\u6210\u4e2d',generated:'\u884c\u52d5\u751f\u6210\u6e08\u307f',executed:'\u884c\u52d5\u5b9f\u884c\u6e08\u307f',error:'\u30a8\u30e9\u30fc'};
 q('#llm-status').textContent=(labels[t.status]||'\u5f85\u6a5f')+(t.elapsed!==undefined?` / ${t.elapsed}s`:'');
 const rows=logEntries.slice();
 const recorded=t.id?rows.some(r=>r.id===t.id):rows.some(r=>r.step===t.step&&r.episode===t.episode&&r.content===t.content);
 if(t.status&&!recorded)rows.push({...t,step:t.step??s.steps+1,episode:t.episode??s.episodes+1});
 const heading=r=>`--- stage ${r.episode??'?'} / step ${r.step} / ${labels[r.status]||r.status} / ${r.action||'...'} ---\n`;
 followText(q('#llm-thinking'),rows.map(r=>heading(r)+originalThought(r)).join('\n\n'));
 followText(q('#llm-content'),rows.map(r=>heading(r)+(r.content||'\u751f\u6210\u5f85\u3061')+(r.error?'\n'+r.error:'')).join('\n\n'));
 q('#llm-action').textContent=t.action||[...rows].reverse().find(r=>r.action)?.action||'\u2014';
 q('#llm-error').textContent=t.error||'';
 const key=s.id+':'+logEntries.length;
 if(key===llmHistoryKey)return;
 llmHistoryKey=key;
 const container=q('#llm-history');
 for(let i=container.childElementCount;i<logEntries.length;i++){
  const r=logEntries[i], item=document.createElement('details'),title=document.createElement('summary'),body=document.createElement('pre');
  title.textContent=`stage ${r.episode??'?'} / step ${r.step}: ${r.action||r.status} / ${r.elapsed}s / reward ${r.reward??'-'}`;
  body.textContent=originalThought(r)+'\n\n'+(r.content||'')+(r.error?'\n'+r.error:'');
  item.append(title,body);container.append(item);
 }
}
fetch('/api/cpu').then(r=>r.json()).then(d=>{q('#cpu-workers').max=d.logical_cpus;q('#cpu-workers').title=`${d.logical_cpus} logical CPUs; auto: ${d.default_workers} workers`}).catch(()=>{});

function renderOrientation(d){
 const compass=['\u6771','\u5357','\u897f','\u5317'];
 q('#view-orientation').textContent=`\u6b63\u9762: ${compass[d]} / \u53f3: ${compass[(d+1)%4]} / \u5de6: ${compass[(d+3)%4]} / \u5f8c\u308d: ${compass[(d+2)%4]}`;
}
