const $ = id => document.getElementById(id);
const state = {page:"overview", logged:false, paused:false, logText:"", history:[], refreshing:false, chartData:{}, logQueryTimer:null, visuals:null, visualIndex:0, visualTimer:null,liveRecords:[],liveCursor:null,livePaused:false,liveLoading:false,liveSelected:null,liveDetailRequest:0};
const titles = {overview:"服务概览",history:"监控历史",logs:"实时日志",live:"实时图片历史",visuals:"离线样本",benchmarks:"历史测速",instance:"实例信息"};
const num = (x,d=1) => x===null||x===undefined||!Number.isFinite(Number(x)) ? "—" : Number(x).toLocaleString("zh-CN",{maximumFractionDigits:d,minimumFractionDigits:d});
const date = ts => ts ? new Date(ts*1000).toLocaleString("zh-CN",{hour12:false}) : "—";
const clock = ts => ts ? new Date(ts*1000).toLocaleTimeString("zh-CN",{hour12:false}) : "—";
const text = (id,value) => {$(id).textContent=value;};
const escapeHtml = str => String(str).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function duration(seconds) {if(seconds===undefined)return "—"; const s=Math.floor(seconds);return s>=86400?`${Math.floor(s/86400)}天 ${Math.floor(s%86400/3600)}小时`:s>=3600?`${Math.floor(s/3600)}小时 ${Math.floor(s%3600/60)}分`:`${Math.floor(s/60)}分 ${s%60}秒`;}
async function api(path,options={}) {
  const r=await fetch(path,{credentials:"same-origin",...options});
  if(r.status===401){showLogin();throw new Error(path==="/api/login"?"用户名或密码错误":"登录已过期，请重新登录");}
  if(!r.ok){const error=await r.json().catch(()=>({detail:"请求失败"}));throw new Error(error.detail||`HTTP ${r.status}`);}
  return r.json();
}
function showLogin(){state.logged=false;stopVisualPlay();$("imageDialog").close();$("imageDialogImg").removeAttribute("src");state.liveRecords=[];state.liveSelected=null;state.liveDetailRequest++;$("liveRecords").innerHTML="";$("liveDetailGrid").innerHTML="";$("liveDetail").classList.add("hidden");$("appView").classList.add("hidden");$("loginView").classList.remove("hidden");}
function showApp(){state.logged=true;$("loginView").classList.add("hidden");$("appView").classList.remove("hidden");}
function error(message){text("globalError",message);$("globalError").classList.toggle("hidden",!message);}
$("loginForm").addEventListener("submit",async e=>{
 e.preventDefault();$("loginButton").disabled=true;text("loginError","");
 try{await api("/api/login",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({username:$("username").value,password:$("password").value})});$("password").value="";showApp();await refresh();}
 catch(err){text("loginError",err.message);}finally{$("loginButton").disabled=false;}
});
$("logout").onclick=async()=>{try{await api("/api/logout",{method:"POST"});}finally{showLogin();}};
$("logoutTop").onclick=$("logout").onclick;
function page(name) {
 if(name!=="visuals")stopVisualPlay();
 state.page=name;
 document.querySelectorAll(".page").forEach(p=>p.classList.toggle("hidden",p.id!==`page-${name}`));
 document.querySelectorAll(".nav-item").forEach(p=>p.classList.toggle("active",p.dataset.page===name));
 text("breadcrumb",titles[name]);refresh();
}
document.querySelectorAll("[data-page]").forEach(el=>el.onclick=()=>page(el.dataset.page));
document.querySelectorAll("[data-open]").forEach(el=>el.onclick=()=>page(el.dataset.open));
$("refresh").onclick=()=>refresh();
$("historyRange").onchange=()=>loadHistory(true);
$("eventLevel").onchange=()=>loadEvents();
$("logSource").onchange=()=>loadLogs();
$("logLines").onchange=()=>loadLogs();
$("logQuery").oninput=()=>{clearTimeout(state.logQueryTimer);state.logQueryTimer=setTimeout(loadLogs,300);};
$("pauseLogs").onclick=()=>{state.paused=!state.paused;text("pauseLogs",state.paused?"继续刷新":"暂停刷新");if(!state.paused)loadLogs();};
$("downloadLog").onclick=()=>{const blob=new Blob([state.logText],{type:"text/plain;charset=utf-8"});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download=`cast-${$("logSource").value}-tail.txt`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
function eventsHTML(events) {
 if(!events.length)return '<div class="empty">这个时间段，暂时没有新事件。</div>';
 return events.map(e=>`<div class="event ${escapeHtml(e.level)}"><span class="event-indicator"></span><div><div class="event-message">${escapeHtml(e.message)}</div><div class="event-time">${escapeHtml(date(e.ts))} · ${escapeHtml(e.source)} · ${escapeHtml(e.level.toUpperCase())}</div></div></div>`).join("");
}
function renderOverview(d) {
 const s=d.snapshot||{}, health=s.health||{}, gpu=(s.gpus||[])[0]||{}, p=s.process||{},host=s.host||{}, c=s.connections||{},counts=s.log_observed_counts||{};
 const stale=d.monitor.stale;
 $("serviceBadge").className=`pill ${stale||!health.ready?"bad":"good"}`;
 text("serviceBadge",stale?"采样延迟":health.ready?(health.gpu_busy?"PROCESSING":"SERVICE HEALTHY"):"NOT READY");
 text("serviceValue",stale?"数据过期":health.ready?"运行中":"未就绪");
 text("serviceFoot",health.model?`${health.model.model_family} · DINOv3 · step ${health.model.global_step}${health.acceleration?.enabled?' · CUDA Graph':''}`:`DINOv3 · :8080 · ${health.gpu_busy?"GPU 工作线程忙":"等待 / 处理请求"}`);
 text("gpuValue",`${num(gpu.util,0)} %`);
 text("gpuFoot",gpu.name?`${gpu.name} · 整卡指标`:"GPU 指标不可用");
 text("connValue",num(c.established,0));
 text("connFoot",`TCP上限 ${c.configured_limit??"—"} · 活跃HTTP ${c.active_requests??"—"}/${c.request_limit??"—"} · 帧 ${c.pipeline_pending??"—"}/${c.pipeline_capacity??"—"}`);
 text("vramText",`${num(gpu.memory_used_mb/1024,1)} / ${num(gpu.memory_total_mb/1024,1)} GB`);
 $("vramBar").value=gpu.memory_total_mb?gpu.memory_used_mb/gpu.memory_total_mb*100:0;
 text("ramText",`${num(host.memory_used_gb)} / ${num(host.memory_total_gb)} GB`);
 $("ramBar").value=host.memory_total_gb?host.memory_used_gb/host.memory_total_gb*100:0;
 text("gpuThermal",`${num(gpu.temperature,0)} °C / ${num(gpu.power_w,0)} W`);
 text("processCpu",`${num(p.cpu_percent_one_core)} %`);
 text("processMem",`${num(p.rss_mb,0)} MB`);
 text("processUptime",duration(p.uptime_seconds));
 text("probeTime",`${num(health.probe_ms)} ms`);
 text("concurrencyCount",num(counts.concurrency_limit,0));text("warningCount",num(counts.warnings,0));text("errorCount",num(counts.errors,0));
 $("alerts").innerHTML=eventsHTML(d.alerts);
 $("navAlert").classList.toggle("hidden",!counts.concurrency_limit);
 text("refreshLabel",s.ts?`${clock(s.ts)} 更新`:"正在首次采样");
 text("footerMeta",`5秒采样 · 已保存 ${num(d.monitor.sample_count,0)} 个样本 · 保留7天`);
 text("instancePid",p.alive?String(p.pid):"不可见（跨容器时正常）");
 text("instanceApiUrl",d.instance.url);text("instanceApiPort",String(d.instance.api_port));text("instanceAdminPort",String(d.instance.admin_port));
 text("instanceModel",health.model?`${health.model.model_family} / DINOv3 / step ${health.model.global_step}`:'DINOv3 / bf16');
 text("instanceCheckpoint",health.model?.checkpoint_name||'未上报');
 text("deployPath",d.instance.deployment);
 if(stale)error(`监控采样超过20秒未更新。${d.monitor.sampler_error||"正在初始化或采集异常，请查看管理员日志。"}`);
 else if(d.monitor.sampler_error)error(`采集异常：${d.monitor.sampler_error}`);
 else error("");
}
function draw(id,points) {
 state.chartData[id]=points;
 const canvas=$(id),r=canvas.getBoundingClientRect();if(!r.width)return;
 const scale=window.devicePixelRatio||1;canvas.width=r.width*scale;canvas.height=r.height*scale;
 const ctx=canvas.getContext("2d");ctx.scale(scale,scale);
 const w=r.width,h=r.height,pad={l:32,r:10,t:13,b:30},cw=w-pad.l-pad.r,ch=h-pad.t-pad.b;
 ctx.font="9px Inter, sans-serif";ctx.textAlign="right";ctx.strokeStyle="#eeebe4";ctx.fillStyle="#b6ac9c";ctx.lineWidth=1;
 [0,25,50,75,100].forEach(v=>{const y=pad.t+ch*(1-v/100);ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(w-pad.r,y);ctx.stroke();ctx.fillText(v,pad.l-9,y+3);});
 if(!points.length){ctx.textAlign="center";ctx.fillText("暂无采样历史",w/2,h/2);return;}
 const min=points[0].ts,max=points[points.length-1].ts;
 const x=p=>pad.l+(points.length===1?.5:(p.ts-min)/(max-min||1))*cw;
 [["gpu_util","#c77555"],["cpu","#a0afb2"]].forEach(([key,color])=>{
   ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();let started=false;
   points.forEach((p,i)=>{if(p[key]===null||p[key]===undefined){started=false;return;}const px=x(p),py=pad.t+ch*(1-Math.min(100,Math.max(0,p[key]))/100);if(i&&p.ts-points[i-1].ts>Math.max(30,(max-min)/points.length*3))started=false;if(started)ctx.lineTo(px,py);else ctx.moveTo(px,py);started=true;});ctx.stroke();
   if(points.length===1&&points[0][key]!=null){ctx.fillStyle=color;ctx.beginPath();ctx.arc(x(points[0]),pad.t+ch*(1-points[0][key]/100),3,0,Math.PI*2);ctx.fill();}
 });
 ctx.fillStyle="#b6ac9c";ctx.textAlign="left";ctx.fillText(clock(min),pad.l,h-5);ctx.textAlign="right";ctx.fillText(clock(max),w-pad.r,h-5);
}
async function loadHistory(full=false) {
 const d=await api(`/api/history?minutes=${full?$("historyRange").value:60}`);
 draw(full?"historyChart":"overviewChart",d.points);
 if(full)text("historyMeta",`${num(d.raw_count,0)} 个原始样本 · 展示 ${d.points.length} 个点 · 抽样间隔 ${d.stride} · 非峰值聚合`);
 else text("chartRange",d.points.length?`${clock(d.points[0].ts)} — ${clock(d.points.at(-1).ts)}`:"等待采样");
}
async function loadEvents(){const d=await api(`/api/events?level=${$("eventLevel").value}`);$("historyEvents").innerHTML=eventsHTML(d.events);}
async function loadLogs() {
 const d=await api(`/api/logs?source=${$("logSource").value}&lines=${$("logLines").value}&query=${encodeURIComponent($("logQuery").value)}`);
 state.logText=d.text;const box=$("logContent"),atBottom=box.scrollHeight-box.scrollTop-box.clientHeight<50;
 box.innerHTML=d.text?d.text.split("\n").map(line=>`<span class="${/ERROR|Traceback|Exception/.test(line)?"err":/WARNING/.test(line)?"warn":""}">${escapeHtml(line)}</span>`).join("\n"):"暂无匹配日志";
 if(atBottom)box.scrollTop=box.scrollHeight;
 text("logMeta",`${d.source}.log · ${num(d.bytes/1024,1)} KiB · 文件最后更新 ${date(d.modified)} · ${state.paused?"刷新已暂停":"5秒自动刷新"}`);
}
async function loadBenchmarks() {
 const d=await api("/api/benchmarks");
 $("benchmarkCards").innerHTML=d.items.length?d.items.map(b=>{
  const r=b.roundtrip||{},shape=b.shape?`${b.shape[1]} × ${b.shape[0]}`:"未知尺寸",local=(b.url||"").includes("127.0.0.1");
  return `<article class="card benchmark-card"><div class="eyebrow">${local?"LOOPBACK":"CROSS-HOST"} / DINOv3</div><h2>${escapeHtml(shape)}</h2><div class="benchmark-fps">${num(r.fps_from_mean,2)}<small>FPS</small></div><span class="benchmark-tag">历史测试结果 · 非实时</span><dl class="details"><div><dt>平均往返</dt><dd>${num(r.mean_ms,2)} ms</dd></div><div><dt>P95 / P99</dt><dd>${num(r.p95_ms,2)} / ${num(r.p99_ms,2)} ms</dd></div><div><dt>首帧</dt><dd>${num(b.first_frame_ms,2)} ms</dd></div><div><dt>测量 / 预热帧</dt><dd>${b.frames} / ${b.warmup}</dd></div><div><dt>逐帧 JPEG 编码</dt><dd>${b.encoding?"已计入":"未计入"}</dd></div><div><dt>骨骼数</dt><dd>${b.joints??"—"}</dd></div></dl><div class="benchmark-file">${escapeHtml(b.date||"")}<br>${escapeHtml(b.file)}</div></article>`;
 }).join(""):'<div class="empty">暂无历史测速记录</div>';
 document.querySelectorAll('.benchmark-card').forEach((card,i)=>{
  const b=d.items[i];if(!b.window)return;
  const tag=card.querySelector('.benchmark-tag');tag.textContent=`${b.window}帧在途 · 历史吞吐（非实时）`;
  const note=document.createElement('p');note.className='subtle-note';
  note.textContent=`${b.label||''} ${b.response_interval?`平均返回间隔 ${num(b.response_interval.mean_ms,2)} ms。`:''}`;
  card.appendChild(note);
 });
}
const visualNames = {
 original:["原图","JPEG 解码后的 RGB"],
 bbox:["裁剪框","红色为 crop 框；首帧黄色为初始提示"],
 mask_overlay:["SAM 叠加","绿色区域为分割 mask"],
 sam_mask:["SAM mask","白色为 mask 前景 · 编码器 512"],
 crop:["裁剪结果","原始 crop，保持比例显示"],
 model_input_256:["模型输入","256×256，归一化前 RGB"]
};
function stopVisualPlay(){clearInterval(state.visualTimer);state.visualTimer=null;if($("visualPlay"))text("visualPlay","播放抽样");}
function openVisual(url,title){
 text("imageDialogSource",url.includes("/api/live-history/")?"线上动捕抽样历史 · 已归档画面":"离线诊断图片 · 非实时画面");
 $("imageDialogImg").src=url;$("imageDialogImg").alt=title;text("imageDialogTitle",title);
 $("imageDownload").href=url;$("imageDownload").download=url.split("/").at(-1);
 $("imageDialog").showModal();
}
$("imageDialogClose").onclick=()=>$("imageDialog").close();
$("imageDialog").addEventListener("click",e=>{if(e.target===$("imageDialog"))$("imageDialog").close();});
function selectVisual(index){
 if(!state.visuals?.frames.length)return;
 state.visualIndex=(index+state.visuals.frames.length)%state.visuals.frames.length;
 renderVisual();
}
$("visualPrev").onclick=()=>{stopVisualPlay();selectVisual(state.visualIndex-1);};
$("visualNext").onclick=()=>{stopVisualPlay();selectVisual(state.visualIndex+1);};
$("visualPlay").onclick=()=>{
 if(state.visualTimer){stopVisualPlay();return;}
 text("visualPlay","暂停抽样");
 state.visualTimer=setInterval(()=>selectVisual(state.visualIndex+1),1200);
};
$("frameStrip").addEventListener("click",e=>{
 const button=e.target.closest("[data-frame-index]");if(button){stopVisualPlay();selectVisual(Number(button.dataset.frameIndex));}
});
function attachVisualClicks(container){
 container.querySelectorAll("[data-image-url]").forEach(el=>{
  el.onclick=()=>openVisual(el.dataset.imageUrl,el.dataset.imageTitle);
 });
 container.querySelectorAll("img").forEach(img=>img.addEventListener("error",()=>{
   img.alt="图片加载失败，请重新登录或刷新";
 }));
}
function renderVisual(){
 const f=state.visuals.frames[state.visualIndex],seq=String(f.seq).padStart(3,"0"),box=f.bbox;
 text("visualFrameTitle",`Frame ${seq}`);
 text("visualFrameMeta",box?`bbox [${box.join(", ")}] · crop ${box[2]-box[0]}×${box[3]-box[1]} · 原图512×512`:"无框元数据");
 $("frameStrip").querySelectorAll("button").forEach((b,i)=>{b.classList.toggle("selected",i===state.visualIndex);b.setAttribute("aria-pressed",String(i===state.visualIndex));});
 $("visualGrid").innerHTML=Object.entries(visualNames).map(([key,[label,description]])=>{
  const url=f.files[key];
  return `<article class="card visual-tile"><div class="card-head"><div><h2>${label}</h2><p>${description}</p></div><span class="visual-index">${seq}</span></div>${url?`<button class="visual-image-button ${key==="sam_mask"?"mask-background":""}" data-image-url="${escapeHtml(url)}" data-image-title="${seq} · ${label}" aria-label="放大${label}"><img src="${escapeHtml(url)}" alt="第${seq}帧 ${label}" decoding="async"></button><div class="visual-image-footer"><span>点击放大</span><a href="${escapeHtml(url)}" download>下载 ↗</a></div>`:'<div class="empty">未保存该类型图片</div>'}</article>`;
 }).join("");
 attachVisualClicks($("visualGrid"));
 drawBBox();
}
function drawBBox(){
 const canvas=$("bboxChart"),rect=canvas.getBoundingClientRect();if(!rect.width||!state.visuals)return;
 const dpr=window.devicePixelRatio||1;canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;
 const ctx=canvas.getContext("2d");ctx.scale(dpr,dpr);
 const w=rect.width,h=rect.height,l=40,t=15,cw=w-55,ch=h-45;
 ctx.font="10px sans-serif";ctx.strokeStyle="#ece7df";ctx.fillStyle="#a79c8d";ctx.textAlign="right";
 [0,25,50,75,100].forEach(v=>{const y=t+ch*(1-v/100);ctx.beginPath();ctx.moveTo(l,y);ctx.lineTo(w-15,y);ctx.stroke();ctx.fillText(v+"%",l-6,y+3);});
 const points=(state.visuals.boxes||[]).filter(p=>p.bbox_xyxy?.length===4);if(!points.length)return;
 const max=Math.max(1,...points.map(p=>p.seq)),area=state.visuals.source_size[0]*state.visuals.source_size[1];
 const xy=p=>{const b=p.bbox_xyxy;return[l+p.seq/max*cw,t+ch*(1-(b[2]-b[0])*(b[3]-b[1])/area)];};
 ctx.strokeStyle="#c77555";ctx.lineWidth=2;ctx.beginPath();points.forEach((p,i)=>{const [x,y]=xy(p);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
 const current=points.find(p=>p.seq===state.visuals.frames[state.visualIndex].seq);
 if(current){const [x,y]=xy(current);ctx.fillStyle="#c77555";ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.fill();}
 ctx.fillStyle="#a79c8d";ctx.textAlign="left";ctx.fillText("Frame 0",l,h-5);ctx.textAlign="right";ctx.fillText("Frame "+max,w-15,h-5);
}
async function loadVisuals(){
 if(state.visuals){drawBBox();return;}
 const d=await api("/api/visuals");state.visuals=d;
 text("visualSourceNote",d.note);
 $("visualEmpty").classList.toggle("hidden",!!d.frames.length);
 $("visualContent").classList.toggle("hidden",!d.frames.length);
 if(!d.frames.length)return;
 $("frameStrip").innerHTML=d.frames.map((f,i)=>`<button data-frame-index="${i}" aria-label="查看第${f.seq}帧"><img src="${escapeHtml(f.files.original)}" alt="" loading="lazy"><span>${String(f.seq).padStart(3,"0")}</span></button>`).join("");
 const labels={"contact_sheet.jpg":"11帧诊断总览","first_frames.jpg":"前几帧检查","first_frame_variants.jpg":"首帧策略对照 · 未上线"};
 $("visualSummaries").innerHTML=d.summaries.map(s=>`<button class="visual-summary" data-image-url="${escapeHtml(s.url)}" data-image-title="${escapeHtml(labels[s.name]||s.name)}"><img src="${escapeHtml(s.url)}" alt="${escapeHtml(labels[s.name]||s.name)}" loading="lazy"><strong>${escapeHtml(labels[s.name]||s.name)}</strong><span>点击打开完整长图 ↗</span></button>`).join("");
 attachVisualClicks($("visualSummaries"));renderVisual();
}
function liveCard(r){
 return `<button class="live-record ${state.liveSelected===r.id?"selected":""}" data-record-id="${r.id}"><img src="/api/live-history/${r.id}/image/bbox" loading="lazy" alt="线上第${r.seq}帧裁剪框"><div class="live-record-body"><strong>${escapeHtml(clock(r.ts))} · Frame ${r.seq}</strong><span>${escapeHtml(r.rig_id)} · ${r.joints}骨 · ${r.shape[1]}×${r.shape[0]}</span><span>${escapeHtml(r.session_label)} / ${r.stream.slice(0,6)} · ${num(r.server_ms,1)} ms</span>${r.tracking?`<span>${r.motion_valid===false?'无人/目标丢失 · 姿态保持':escapeHtml(r.tracking.mode)+' · '+escapeHtml(r.tracking.crop_source)}</span>`:''}</div></button>`;
}
async function loadLive(older=false){
 if(state.liveLoading)return;state.liveLoading=true;
 try{
  const selected=$("liveStream").value;
  const url=`/api/live-history?limit=36&stream=${encodeURIComponent(selected)}`+(older&&state.liveCursor?`&before=${state.liveCursor}`:"");
  const d=await api(url),s=d.status||{},fresh=s.updated_at&&Date.now()/1000-s.updated_at<20;
  $("liveCaptureBadge").className=`pill ${s.enabled&&fresh?"good":"neutral"}`;
  text("liveCaptureBadge",s.enabled?(fresh?"CAPTURE ENABLED":"采集器状态过期"):"等待采集器");
  text("liveCaptureStatus",`保留 ${d.total_retained} 条 · ${num((s.bytes||0)/1024**2,1)} MiB · 最近状态 ${date(s.updated_at)}`);
  text("liveCaptureDetail",`后台队列 ${s.queue_depth??0}/${s.queue_capacity??4} · 本次进程丢弃历史样本 ${s.dropped_this_run??0} · 写入错误 ${s.write_errors_this_run??0} · 采样间隔 ${s.interval_seconds??1}s。停止发送图片时不会产生新记录。`);
  $("liveStream").innerHTML='<option value="">全部会话</option>'+d.streams.map(g=>`<option value="${g.stream}">${escapeHtml(g.rig_id)} · ${g.session_label}/${g.stream.slice(0,6)} · ${g.samples}条</option>`).join("");
  $("liveStream").value=d.streams.some(g=>g.stream===selected)?selected:"";
  if(older){const seen=new Set(state.liveRecords.map(r=>r.id));state.liveRecords.push(...d.records.filter(r=>!seen.has(r.id)));}
  else state.liveRecords=d.records;
  state.liveCursor=d.next_before;
  $("liveRecords").innerHTML=state.liveRecords.map(liveCard).join("");
  $("liveEmpty").classList.toggle("hidden",!!state.liveRecords.length);
  $("liveOlder").classList.toggle("hidden",!d.has_more);
 }catch(err){if(state.logged)error(`图片历史读取失败：${err.message}`);}finally{state.liveLoading=false;}
}
async function liveDetail(id){
 const request=++state.liveDetailRequest;
 try{
  const d=await api(`/api/live-history/${id}`);if(request!==state.liveDetailRequest)return;
  const r=d.summary;state.liveSelected=id;
  $("liveDetail").classList.remove("hidden");
  text("liveDetailTitle",`${date(r.ts)} · Frame ${r.seq}`);
  text("liveDetailMeta",`${r.rig_id} · ${r.joints}骨 · ${r.shape[1]}×${r.shape[0]} · bbox ${r.bbox?`[${r.bbox.join(", ")}]`:'无人/无有效框'} · 会话 ${r.session_label} / 段 ${r.stream.slice(0,6)} · ${r.tracking?`${r.tracking.status} / ${r.tracking.crop_source}`:''}`);
  const t=d.timing_ms;
  text("liveTiming",`解码 ${num(t.decode,2)} ms · SAM+crop ${num(t.sam_crop,2)} ms · 模型+D2H ${num(t.model_and_d2h,2)} ms · 归档采样前服务耗时 ${num(t.server_total,2)} ms。图像写盘异步执行，不计入这里；归档RGB为JPEG重编码。`);
  if(r.tracking){const tr=r.tracking;text('liveTiming',$('liveTiming').textContent+` 人体检测 ${num(tr.detector_ms,2)}ms · 分数 ${num(tr.detector_score,3)} · motion_valid=${r.motion_valid} · ${tr.motion_source||''}。`);}
  $("liveDetailGrid").innerHTML=Object.entries(visualNames).map(([key,[label,description]])=>{
   const url=d.files[key];return `<article class="card visual-tile"><div class="card-head"><div><h2>${label}</h2><p>${key==="bbox"?"线上实际crop框":description}</p></div><span class="visual-index">${r.seq}</span></div>${url?`<button class="visual-image-button" data-image-url="${url}" data-image-title="${escapeHtml(date(r.ts))} Frame ${r.seq} · ${label}"><img src="${url}" alt="${label}" decoding="async"></button><div class="visual-image-footer"><span>线上归档 · 点击放大</span><a href="${url}" download>下载 ↗</a></div>`:'<div class="empty">无有效目标或未保存此类型</div>'}</article>`;
  }).join("");
  attachVisualClicks($("liveDetailGrid"));
  $("liveRecords").querySelectorAll("[data-record-id]").forEach(el=>el.classList.toggle("selected",el.dataset.recordId===id));
  $("liveDetail").scrollIntoView({behavior:"smooth",block:"start"});
 }catch(err){if(state.logged)error(`该记录无法打开：${err.message}（可能已到期清理）`);}
}
$("liveRecords").onclick=e=>{const el=e.target.closest("[data-record-id]");if(el)liveDetail(el.dataset.recordId);};
$("liveStream").onchange=()=>{state.liveCursor=null;loadLive();};
$("liveRefresh").onclick=()=>{state.liveCursor=null;loadLive();};
$("liveOlder").onclick=()=>{state.livePaused=true;text("livePause","继续自动刷新");loadLive(true);};
$("livePause").onclick=()=>{state.livePaused=!state.livePaused;text("livePause",state.livePaused?"继续自动刷新":"暂停自动刷新");if(!state.livePaused)loadLive();};
$("liveDetailClose").onclick=()=>{$("liveDetail").classList.add("hidden");state.liveSelected=null;state.liveDetailRequest++;};
async function refresh() {
 if(!state.logged||state.refreshing)return;
 state.refreshing=true;
 try {
  renderOverview(await api("/api/overview"));
  if(state.page==="overview")await loadHistory();
  if(state.page==="history")await Promise.all([loadHistory(true),loadEvents()]);
  if(state.page==="logs"&&!state.paused)await loadLogs();
  if(state.page==="benchmarks")await loadBenchmarks();
  if(state.page==="visuals")await loadVisuals();
  if(state.page==="live"&&!state.livePaused)await loadLive();
 }catch(err){if(state.logged)error(`无法更新：${err.message}。页面保留上次数据。`);}
 finally{state.refreshing=false;}
}
window.addEventListener("resize",()=>{Object.entries(state.chartData).forEach(([id,p])=>draw(id,p));if(state.page==="visuals")drawBBox();});
setInterval(()=>{if(!document.hidden)refresh();},5000);
document.addEventListener("visibilitychange",()=>{if(!document.hidden)refresh();else stopVisualPlay();});
(async()=>{try{await api("/api/overview");showApp();await refresh();}catch{showLogin();}})();
