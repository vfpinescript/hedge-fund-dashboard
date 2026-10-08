/* Hedge Fund v2 — Quant Desk dashboard */
const INSTRUMENTS = ["XAUUSD","EURUSD","GBPUSD","USDJPY","AUDUSD","USOIL","ES1!","NQ1!"];
const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const fmtPct = v => (v==null||isNaN(v)) ? "–" : (v*100).toFixed(2)+"%";
const fmtNum = v => (v==null||isNaN(v)) ? "–" : (v===Infinity?"∞":v.toFixed(2));
const fmtInt = v => (v==null) ? "–" : Math.round(v);

const PLOT_CFG = {displayModeBar:false, responsive:true};
const axis = {gridcolor:"#1e2530", zerolinecolor:"#2a3341", tickfont:{size:11}, showline:false};
const baseLayout = extra => Object.assign({
  paper_bgcolor:"transparent", plot_bgcolor:"transparent",
  font:{color:"#7c8698", family:"-apple-system,Inter,sans-serif"},
  margin:{l:44,r:16,t:10,b:34}, xaxis:{...axis}, yaxis:{...axis}, showlegend:false, hovermode:"x unified"
}, extra||{});

/* ---------- tabs ---------- */
function showTab(name){
  document.querySelectorAll(".tab").forEach(x=>x.classList.toggle("active", x.dataset.tab===name));
  ["strategies","matrix","forward","analytics","live","signals","lab"].forEach(v=>
    document.getElementById("view-"+v).classList.toggle("hidden", v!==name));
  if(name==="matrix" && !window._matrixLoaded){ loadMatrix(); window._matrixLoaded=1; }
  if(name==="forward" && !window._fwdLoaded){ loadForward(); window._fwdLoaded=1; }
  if(name==="analytics" && !window._anLoaded){ loadAnalytics(); window._anLoaded=1; }
  if(name==="live" && !window._liveLoaded){ loadLive(); window._liveLoaded=1; }
  if(name==="signals" && !window._sigLoaded){ loadSignals(); window._sigLoaded=1; }
  if(name==="lab" && !window._labLoaded){ loadLab(); window._labLoaded=1; }
}
document.querySelectorAll(".tab").forEach(t=>t.onclick=()=>showTab(t.dataset.tab));

/* ---------- strategies ---------- */
const METRICS = [
  ["total_return","Total Return","pct","blue"],["max_drawdown","Max Drawdown","pct","red"],
  ["sharpe","Sharpe Ratio","num",""],["win_rate","Win Rate","pct",""],
  ["profit_factor","Profit Factor","num",""],["sortino","Sortino Ratio","num",""],
  ["avg_win","Avg Win","pct","blue"],["avg_loss","Avg Loss","pct","red"],
  ["risk_reward","Risk–Reward Ratio","num",""],["max_win_streak","Max Win Streak","int",""],
  ["time_in_dd","Time in DD","pct",""],["max_loss_streak","Max Losing Streak","int",""],
];
let ASSET_GROUPS={}, CUR_STRAT="reversal_7", CUR_ASSET=null;

async function loadStrategies(){
  const lib = await (await fetch("/api/strategy_library")).json();
  const picker=document.getElementById("stratPicker");
  picker.innerHTML=lib.strategies.map(s=>`<option value="${s.id}">${s.name}</option>`).join("");
  if([...picker.options].some(o=>o.value===CUR_STRAT)) picker.value=CUR_STRAT; else CUR_STRAT=picker.value;
  picker.onchange=()=>{ CUR_STRAT=picker.value; if(CUR_ASSET) selectAsset(CUR_ASSET); };
  ASSET_GROUPS=lib.instrument_groups;
  const list=document.getElementById("stratList");
  let html="";
  for(const [g,syms] of Object.entries(ASSET_GROUPS)){
    html+=`<div class="list-head">${g}</div>`;
    for(const s of syms) html+=`<div class="strat-item" data-asset="${s}"><div class="si-top"><span class="si-name">${s}</span></div></div>`;
  }
  list.innerHTML=html;
  list.querySelectorAll(".strat-item").forEach(el=>el.onclick=()=>selectAsset(el.dataset.asset));
  const first=list.querySelector(".strat-item"); if(first) selectAsset(first.dataset.asset);
}

function selectAsset(asset){
  CUR_ASSET=asset;
  document.querySelectorAll("#stratList .strat-item").forEach(x=>x.classList.toggle("active", x.dataset.asset===asset));
  renderRun(CUR_STRAT, asset);
}

// entry point from the Strategy Matrix: set strategy + asset, then show the page
function openStrategy(strat, asset){
  CUR_STRAT=strat;
  const picker=document.getElementById("stratPicker"); if(picker && [...picker.options].some(o=>o.value===strat)) picker.value=strat;
  showTab("strategies");
  selectAsset(asset);
}

function dsrChip(dsr, verdict){
  if(dsr==null || isNaN(dsr)) return "";
  const col=verdict==="robust"?"var(--pos)":verdict==="inconclusive"?"var(--warn)":"var(--neg)";
  return `<div style="text-align:right">
    <div style="font-size:10px;color:var(--mut);text-transform:uppercase;letter-spacing:.9px">Overfitting check · DSR</div>
    <div style="font-family:var(--mono);font-size:22px;font-weight:700;color:${col}">${(dsr*100).toFixed(1)}%</div>
    <div style="font-size:11px;color:${col};font-weight:600">${verdict}</div></div>`;
}

async function renderRun(strat, asset){
  const box=document.getElementById("stratDetail");
  box.innerHTML=`<div class="loading">Backtesting ${strat} on ${asset}…</div>`;
  let d;
  try{ d=await (await fetch(`/api/strategy_run?strategy=${encodeURIComponent(strat)}&instrument=${encodeURIComponent(asset)}`)).json(); }
  catch(e){ box.innerHTML=`<div class="loading">Error: ${e}</div>`; return; }
  if(d.error){ box.innerHTML=`<div class="loading">${d.error}</div>`; return; }
  const m=d.metrics;
  const metricHTML=METRICS.map(([k,label,fmt,color])=>{
    let v=m[k], disp=fmt==="pct"?fmtPct(v):fmt==="int"?fmtInt(v):fmtNum(v);
    let c=color; if(!c && fmt==="pct") c=(v>=0?"":"red");
    return `<div class="metric"><div class="ml">${label}</div><div class="mv ${c}">${disp}</div></div>`;
  }).join("");
  const tr=m.total_return, trCls=tr>=0?"blue":"red";
  box.innerHTML=`
    <div class="detail-head">
      <div><h1>${d.strategy}</h1><div class="desc">${asset} · net of costs · ${d.equity.dates.length} days</div></div>
      ${dsrChip(d.dsr,d.dsr_verdict)}
    </div>
    <div class="mgrid" style="margin-bottom:22px">${metricHTML}</div>
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:baseline"><h3>Total Profit &amp; Loss</h3>
        <span class="muted" style="font-size:12px">Strategy vs Buy &amp; Hold</span></div>
      <div class="pnl-big" style="color:var(--${trCls})">${fmtPct(tr)}</div>
      <div class="chart" id="eqChart" style="height:320px"></div></div>
    <div class="card-row">
      <div class="card"><h3>Drawdown</h3><div class="sub">Underwater equity</div>
        <div class="chart" id="ddChart" style="height:240px"></div></div>
      <div class="card"><h3>Monthly Returns</h3><div class="sub">Calendar heatmap</div>
        <div id="heat" style="padding:14px 6px"></div></div>
    </div>`;
  Plotly.newPlot("eqChart",[
    {x:d.equity.dates,y:d.equity.strategy,type:"scatter",mode:"lines",name:"Strategy",line:{color:"#6d97cf",width:2.4},fill:"tozeroy",fillcolor:"rgba(109,151,207,.08)"},
    {x:d.equity.dates,y:d.equity.buyhold,type:"scatter",mode:"lines",name:"Buy & Hold",line:{color:"#7c8698",width:1.5,dash:"dash"}}
  ],baseLayout({yaxis:{...axis,ticksuffix:"%"},showlegend:true,legend:{orientation:"h",y:1.1,font:{size:11}}}),PLOT_CFG);
  Plotly.newPlot("ddChart",[{x:d.drawdown.dates,y:d.drawdown.values,type:"scatter",mode:"lines",line:{color:"#d76b63",width:1},fill:"tozeroy",fillcolor:"rgba(215,107,99,.25)"}],baseLayout({yaxis:{...axis,ticksuffix:"%"}}),PLOT_CFG);
  renderHeat(d.monthly);
}

function renderHeat(monthly){
  let h=`<div class="heat"><div></div>`+MONTHS.map(m=>`<div class="hh">${m}</div>`).join("");
  for(const y of monthly.years){
    h+=`<div class="hy">${y}</div>`;
    monthly.grid[y].forEach(v=>{
      if(v==null){h+=`<div class="hc empty"></div>`;return;}
      const a=Math.min(.85,Math.abs(v)/7*.8+.14);
      const bg=v>=0?`rgba(79,174,130,${a})`:`rgba(215,107,99,${a})`;
      h+=`<div class="hc" style="background:${bg}">${v.toFixed(1)}</div>`;
    });
  }
  h+=`</div>`;
  document.getElementById("heat").innerHTML=h;
}

/* ---------- live signals ---------- */
function fillSelect(id){ const sel=document.getElementById(id);
  sel.innerHTML=INSTRUMENTS.map(i=>`<option value="${i}">${i}</option>`).join(""); return sel; }

async function loadSignals(){
  const sel=fillSelect("signalInstrument");
  sel.onchange=()=>renderSignals(sel.value);
  renderSignals(sel.value);
}
async function renderSignals(inst){
  const body=document.getElementById("signalsBody");
  body.innerHTML=`<div class="loading">Running committee on ${inst}…</div>`;
  const r=await (await fetch("/api/signals?instrument="+encodeURIComponent(inst))).json();
  if(r.error){body.innerHTML=`<div class="loading">Error: ${r.error}</div>`;return;}
  const d=r.decision, dir=d.direction, conv=Math.round(d.conviction*100);
  const dcls=dir==="bull"?"dir-bull":dir==="bear"?"dir-bear":"dir-neutral";
  const votes=d.votes.map(v=>agentCard(v.agent,"directional",v.dir,v.conviction,v.why)).join("");
  const risks=d.risk.map(x=>agentCard(x.agent,"risk",x.veto?"VETO":`×${x.scale}`,null,x.why)).join("");
  body.innerHTML=`
    <div class="decision-banner">
      <div class="dir-chip ${dcls}">${dir.toUpperCase()}</div>
      <div><div class="muted" style="font-size:12px">${inst} · ${r.last_price} · as of ${r.as_of}</div>
        <div style="font-weight:700;margin-top:2px">Committee decision · conviction ${conv}%</div></div>
      <div class="conv-bar"><div class="muted" style="font-size:11px">conviction × risk scale = ${(d.conviction*d.risk_scale).toFixed(2)}</div>
        <div class="conv-track"><div class="conv-fill" style="width:${conv}%"></div></div></div>
    </div>
    <div class="sec-title">Directional agents (vote)</div>
    <div class="agent-grid">${votes}</div>
    <div class="sec-title">Risk agents (scale / veto)</div>
    <div class="agent-grid">${risks}</div>`;
}
function agentCard(name,role,verdict,conv,why){
  let vc = verdict==="bull"?"var(--green)":verdict==="bear"?"var(--red)":
    (verdict==="VETO"?"var(--red)":verdict==="abstain"?"var(--muted2)":"var(--text)");
  const convTxt = conv!=null?` · ${(conv*100).toFixed(0)}%`:"";
  return `<div class="agent"><div class="a-top"><span class="a-name">${name}</span>
    <span class="a-role">${role}</span></div>
    <div class="a-verdict" style="color:${vc}">${String(verdict).toUpperCase()}${convTxt}</div>
    <div class="a-why">${why||""}</div></div>`;
}

/* ---------- strategy matrix ---------- */
let MATRIX=null;
function matrixColor(v, metric){
  if(v==null) return "transparent";
  // map to a red->grey->green scale; scale midpoint/ranges per metric
  const ranges={total_return:[-60,60], sharpe:[-1,1], sortino:[-1.5,1.5], dsr:[0,1]};
  const [lo,hi]=ranges[metric]||[-1,1];
  let x=metric==="dsr"?(v-lo)/(hi-lo):(v-lo)/(hi-lo);
  x=Math.max(0,Math.min(1,x));
  const a=Math.abs(x-0.5)*2*0.55+0.06;
  if(x>=0.5) return `rgba(79,174,130,${a})`;      // green
  return `rgba(215,107,99,${a})`;                 // red
}
function matrixVal(cell, metric){
  if(!cell) return null;
  return metric==="dsr" ? cell.dsr : cell[metric];
}
function matrixDisp(cell, metric){
  const v=matrixVal(cell,metric);
  if(v==null) return "–";
  if(metric==="total_return") return v.toFixed(0)+"%";
  if(metric==="dsr") return (v*100).toFixed(0)+"%";
  return v.toFixed(2);
}
async function loadMatrix(){
  const body=document.getElementById("matrixBody");
  try{ MATRIX=await (await fetch("/api/strategy_matrix")).json(); }
  catch(e){ body.innerHTML=`<div class="loading">Error: ${e}</div>`; return; }
  if(MATRIX.error){ body.innerHTML=`<div class="loading">${MATRIX.error}</div>`; return; }
  document.getElementById("matrixMetric").onchange=renderMatrix;
  renderMatrix();
}
function renderMatrix(){
  const metric=document.getElementById("matrixMetric").value;
  const S=MATRIX.strategies;
  let h=`<div class="matrix-wrap"><table class="mtx"><thead><tr><th class="rowhead corner">Asset \\ Strategy</th>`;
  h+=S.map(s=>`<th title="${s.name}">${s.name.replace(/ \(.*\)/,"")}</th>`).join("");
  h+=`</tr></thead><tbody>`;
  for(const row of MATRIX.rows){
    h+=`<tr><td class="rowhead">${row.asset}</td>`;
    for(const s of S){
      const cell=row.cells[s.id];
      if(!cell){ h+=`<td class="cell na">–</td>`; continue; }
      const v=matrixVal(cell,metric);
      h+=`<td class="cell" style="background:${matrixColor(v,metric)}" data-strat="${s.id}" data-asset="${row.asset}"
        title="${row.asset} · ${s.name}\nreturn ${cell.total_return}%  Sharpe ${cell.sharpe}  Sortino ${cell.sortino}  DSR ${(cell.dsr*100).toFixed(0)}% (${cell.verdict})">${matrixDisp(cell,metric)}</td>`;
    }
    h+=`</tr>`;
  }
  h+=`</tbody></table></div>
    <div class="mtx-legend">Lower<span class="mtx-swatch"></span>Higher · click any cell to open it on the Strategies page. Colour = ${metric.replace("_"," ")}.</div>`;
  const body=document.getElementById("matrixBody");
  body.innerHTML=h;
  body.querySelectorAll("td.cell[data-strat]").forEach(td=>td.onclick=()=>openStrategy(td.dataset.strat, td.dataset.asset));
}

/* ---------- forward test ---------- */
let FWD=null, FWD_SORT="sharpe";
async function loadForward(){
  const body=document.getElementById("forwardBody");
  try{ FWD=await (await fetch("/api/forward_test")).json(); }
  catch(e){ body.innerHTML=`<div class="loading">Error: ${e}</div>`; return; }
  if(FWD.error){ body.innerHTML=`<div class="loading">${FWD.error}</div>`; return; }
  renderForward();
}
function renderForward(){
  const rows=[...FWD.rows].sort((a,b)=> (b[FWD_SORT]??-1e9)-(a[FWD_SORT]??-1e9));
  const days=FWD.rows[0]?FWD.rows[0].days:0;
  const col=v=>v==null?"":v>=0?"var(--pos)":"var(--neg)";
  const hd=(k,l)=>`<th class="sortable" data-k="${k}" style="cursor:pointer${FWD_SORT===k?';color:var(--gold)':''}">${l}${FWD_SORT===k?' ▾':''}</th>`;
  let h=`<div class="card" style="padding:16px 18px;margin-bottom:16px">
    <div style="display:flex;gap:28px;align-items:baseline;flex-wrap:wrap">
      <div><div class="ml">Tracking since</div><div class="mv" style="font-size:16px">${FWD.start}</div></div>
      <div><div class="ml">Out-of-sample days</div><div class="mv" style="font-size:16px">${days}</div></div>
      <div><div class="ml">Combinations tracked</div><div class="mv" style="font-size:16px">${FWD.n}</div></div>
      <div class="muted" style="font-size:12px;max-width:420px">Ranked by out-of-sample ${FWD_SORT}. Small samples so far: treat as accruing evidence, not proof.</div>
    </div></div>
    <div class="matrix-wrap"><table class="mtx" style="font-size:12.5px"><thead><tr>
      <th class="rowhead corner">Strategy · Asset</th>${hd("return","Return %")}${hd("sharpe","Sharpe")}${hd("sortino","Sortino")}${hd("max_dd","Max DD %")}${hd("rr","Return/Risk")}<th>Days</th></tr></thead><tbody>`;
  for(const r of rows){
    h+=`<tr class="fwd-row" data-strat="${r.strategy_id}" data-asset="${r.asset}" style="cursor:pointer">
      <td class="rowhead" style="min-width:230px">${r.strategy} · <b>${r.asset}</b></td>
      <td class="cell num" style="color:${col(r.return)}">${r.return}%</td>
      <td class="cell num" style="color:${col(r.sharpe)}">${r.sharpe}</td>
      <td class="cell num" style="color:${col(r.sortino)}">${r.sortino}</td>
      <td class="cell num" style="color:var(--neg)">${r.max_dd}%</td>
      <td class="cell num">${r.rr==null?"–":r.rr}</td>
      <td class="cell num muted">${r.days}</td></tr>`;
  }
  h+=`</tbody></table></div>`;
  const body=document.getElementById("forwardBody"); body.innerHTML=h;
  body.querySelectorAll("th.sortable").forEach(th=>th.onclick=()=>{ FWD_SORT=th.dataset.k; renderForward(); });
  body.querySelectorAll("tr.fwd-row").forEach(tr=>tr.onclick=()=>openStrategy(tr.dataset.strat, tr.dataset.asset));
}

/* ---------- analytics (monte carlo + surfaces) ---------- */
async function loadAnalytics(){
  const lib=await (await fetch("/api/strategy_library")).json();
  const sp=document.getElementById("anStrat");
  sp.innerHTML=lib.strategies.map(s=>`<option value="${s.id}">${s.name}</option>`).join("");
  if([...sp.options].some(o=>o.value==="reversal_7")) sp.value="reversal_7";
  const ap=document.getElementById("anAsset");
  let opts="";
  for(const [g,syms] of Object.entries(lib.instrument_groups)) opts+=`<optgroup label="${g}">`+syms.map(s=>`<option value="${s}">${s}</option>`).join("")+`</optgroup>`;
  ap.innerHTML=opts; ap.value="AUDUSD";
  sp.onchange=renderAnalytics; ap.onchange=renderAnalytics;
  renderAnalytics();
}
async function renderAnalytics(){
  const strat=document.getElementById("anStrat").value, asset=document.getElementById("anAsset").value;
  const body=document.getElementById("analyticsBody");
  body.innerHTML=`<div class="loading">Running 2,000 Monte Carlo paths and computing surfaces for ${strat} on ${asset}…</div>`;
  let a,S;
  try{ [a,S]=await Promise.all([
    fetch(`/api/analytics?strategy=${encodeURIComponent(strat)}&instrument=${encodeURIComponent(asset)}`).then(r=>r.json()),
    fetch(`/api/surfaces?instrument=${encodeURIComponent(asset)}`).then(r=>r.json())
  ]); }catch(e){ body.innerHTML=`<div class="loading">Error: ${e}</div>`; return; }
  if(a.error){ body.innerHTML=`<div class="loading">${a.error}</div>`; return; }
  const mc=a.mc, col=v=>v>=0?"var(--pos)":"var(--neg)";
  body.innerHTML=`
    <div class="decision-banner">
      <div><div class="ml">Median 1y return</div><div class="mv" style="font-size:24px;color:${col(mc.median_return_pct)}">${mc.median_return_pct>=0?"+":""}${mc.median_return_pct}%</div></div>
      <div style="border-left:1px solid var(--line);padding-left:20px"><div class="ml">5th / 95th pct</div><div class="mv" style="font-size:18px">${mc.p5_return_pct}% / +${mc.p95_return_pct}%</div></div>
      <div style="border-left:1px solid var(--line);padding-left:20px"><div class="ml">Chance of loss (1y)</div><div class="mv" style="font-size:18px;color:var(--warn)">${mc.prob_loss_pct}%</div></div>
      <div style="border-left:1px solid var(--line);padding-left:20px"><div class="ml">Median / worst max DD</div><div class="mv" style="font-size:18px;color:var(--neg)">${mc.median_max_dd_pct}% / ${mc.worst_max_dd_pct}%</div></div>
    </div>
    <div class="card"><h3>Monte Carlo — range of 1-year outcomes</h3><div class="sub">2,000 block-bootstrap paths from the strategy's own daily returns. The shaded cone is the 5th to 95th percentile, the line is the median. A single backtest is just one of these paths.</div>
      <div id="mcFan" style="height:380px"></div></div>
    <div class="card-row">
      <div class="card"><h3>Return Distribution</h3><div class="sub">Daily returns · skew ${a.dist.skew} · excess kurtosis ${(a.dist.kurt-3).toFixed(1)}</div><div id="anDist" style="height:260px"></div></div>
      <div class="card"><h3>Rolling Sharpe</h3><div class="sub">126-day window</div><div id="anRS" style="height:260px"></div></div>
    </div>
    <div class="sec-title">3D Quant Surfaces · ${asset}</div>
    <div class="surf-grid">
      ${surfDiv("sTail","Tail-Exponent Surface","Hill index over window × order statistic — fat-tail risk")}
      ${surfDiv("sVol","Volatility Surface","Implied vol over strike × expiry (SPY options)")}
      ${surfDiv("sGamma","Gamma Surface","Option gamma over spot × maturity")}
      ${surfDiv("sCharm","Charm Surface","Delta decay over spot × maturity")}
    </div>`;
  const days=[...Array(mc.days).keys()];
  const band=(lo,hi,c)=>[
    {x:days,y:mc.bands[hi],type:"scatter",mode:"lines",line:{width:0},showlegend:false,hoverinfo:"skip"},
    {x:days,y:mc.bands[lo],type:"scatter",mode:"lines",line:{width:0},fill:"tonexty",fillcolor:c,showlegend:false,hoverinfo:"skip"}];
  const traces=[...band("5","95","rgba(109,151,207,.12)"),...band("25","75","rgba(109,151,207,.22)"),
    {x:days,y:mc.bands["50"],type:"scatter",mode:"lines",line:{color:"#6d97cf",width:2.4},name:"median"}];
  (mc.sample_paths||[]).slice(0,18).forEach(p=>traces.push({x:days,y:p,type:"scatter",mode:"lines",line:{color:"#c6a15b",width:0.6},opacity:.22,showlegend:false,hoverinfo:"skip"}));
  Plotly.newPlot("mcFan",traces,baseLayout({yaxis:{...axis,ticksuffix:"%"},xaxis:{...axis,title:"trading days forward"}}),PLOT_CFG);
  const e=a.dist.edges, centers=e.slice(0,-1).map((x,i)=>(x+e[i+1])/2);
  Plotly.newPlot("anDist",[{x:centers,y:a.dist.counts,type:"bar",marker:{color:"#6d97cf"}}],baseLayout({xaxis:{...axis,title:"daily return %"},bargap:0.02}),PLOT_CFG);
  Plotly.newPlot("anRS",[{x:a.rolling_sharpe.dates,y:a.rolling_sharpe.values,type:"scatter",mode:"lines",line:{color:"#cf9f45",width:1.3}}],baseLayout({}),PLOT_CFG);
  if(!S.error){
    plotSurface("sTail",S.tail); plotSurface("sVol",S.vol,"Viridis");
    plotSurface("sGamma",S.gamma); plotSurface("sCharm",S.charm,[[0,"#0b2b4a"],[.5,"#2a7de1"],[1,"#f9e07f"]]);
  }
}
const SURF_SCALE=[[0,"#1a1140"],[.3,"#7a2a8f"],[.55,"#c73e6b"],[.78,"#f5844a"],[1,"#f9e07f"]];
function surfDiv(id,title,sub){return `<div class="card"><h3>${title}</h3><div class="sub">${sub}</div><div id="${id}" style="height:340px"></div></div>`;}
function plotSurface(id,Sf,scale){
  const el=document.getElementById(id); if(!el) return;
  if(!Sf){el.innerHTML=`<div class="loading">No data feed available — abstained.</div>`;return;}
  Plotly.newPlot(id,[{type:"surface",x:Sf.x,y:Sf.y,z:Sf.z,colorscale:scale||SURF_SCALE,showscale:false,contours:{z:{show:true,usecolormap:true}}}],
    {paper_bgcolor:"transparent",font:{color:"#7c8698",size:10},margin:{l:0,r:0,t:0,b:0},
     scene:{xaxis:{title:Sf.xlabel||"",gridcolor:"#2a3341",backgroundcolor:"transparent",showbackground:false},
       yaxis:{title:Sf.ylabel||"",gridcolor:"#2a3341",backgroundcolor:"transparent",showbackground:false},
       zaxis:{title:Sf.zlabel||"",gridcolor:"#2a3341",backgroundcolor:"transparent",showbackground:false},
       camera:{eye:{x:1.6,y:-1.5,z:.9}}}},PLOT_CFG);
}

/* ---------- live paper ---------- */
document.getElementById("liveRefresh").onclick=()=>loadLive();
const usd = v => (v==null||isNaN(v)) ? "–" : (v<0?"-$":"$")+Math.abs(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
async function loadLive(){
  const body=document.getElementById("liveBody");
  body.innerHTML=`<div class="loading">Reading the live Alpaca paper account…</div>`;
  let d;
  try{ d=await (await fetch("/api/live")).json(); }
  catch(e){ body.innerHTML=`<div class="loading">Error: ${e}</div>`; return; }
  if(d.error){ body.innerHTML=`<div class="loading">Snapshot unavailable: ${d.error}<br><span class="muted" style="font-size:12px">Alpaca paper API — check the keys and market status.</span></div>`; return; }
  const pos=d.positions||[];
  const acct=d.account||{};
  const unreal=acct.unrealizedPnl!=null?acct.unrealizedPnl:pos.reduce((s,p)=>s+(p.pl||0),0);
  const real=acct.realizedPnl||0;
  const totalPnl=real+unreal;
  const equity=acct.equity!=null?acct.equity:(d.equityEst!=null?d.equityEst:d.start_equity+totalPnl);
  const ret=(equity/d.start_equity-1)*100;
  const col=v=>v>=0?"var(--green)":"var(--red)";
  const fwd=d.forward||{}; const pr=d.portfolio_risk||{};

  const rows=pos.map(p=>{
    const c=col(p.pl||0), sc=p.side==="long"?"var(--green)":"var(--red)";
    return `<tr><td>${p.symbol}</td><td style="color:${sc};font-weight:700">${p.side.toUpperCase()}</td>
      <td class="num">${p.qty.toLocaleString()}</td><td class="num">${(p.avgPrice||0).toFixed(5)}</td>
      <td class="num">${(p.lastPrice||0).toFixed(5)}</td>
      <td class="num" style="color:${c}">${usd(p.pl)}</td>
      <td class="num" style="color:${c}">${((p.plPercent||0)>=0?"+":"")}${(p.plPercent||0).toFixed(2)}%</td></tr>`;
  }).join("");
  const trades=(d.trades||[]).slice(-12).reverse().map(t=>{
    const sc=t.side==="buy"?"var(--green)":"var(--red)";
    return `<tr><td>${t.symbol}</td><td style="color:${sc};font-weight:700">${t.side.toUpperCase()}</td>
      <td class="num">${(t.qty||0).toLocaleString()}</td><td class="num">${t.price!=null?Number(t.price).toFixed(5):"–"}</td></tr>`;
  }).join("");
  const prRows=(pr.contributions||[]).slice(0,6).map(c=>`<tr><td>${c.symbol}</td>
      <td class="num">${c.weight}</td><td class="num">${(c.vol*100).toFixed(1)}%</td>
      <td class="num" style="color:${col(c.risk_contribution_pct)}">${c.risk_contribution_pct}%</td></tr>`).join("");

  const bk=d.book||{};
  const bookHTML = bk.error ? `<div class="card"><h3>Book Construction</h3><div class="sub" style="padding-bottom:14px">${bk.error}</div></div>` : (()=>{
    const reg=bk.regime||{};
    const rc = reg.state==="turbulent"?"var(--warn)":reg.state==="calm"?"var(--pos)":"var(--mut)";
    const w=bk.weights||{};
    const bar=(label,val,c)=>`<div style="margin-bottom:9px">
      <div style="display:flex;justify-content:space-between;font-size:12px"><span>${label}</span><span class="num" style="color:${c}">${((val||0)*100).toFixed(1)}%</span></div>
      <div style="height:7px;background:var(--line-2);border-radius:3px;margin-top:4px;overflow:hidden"><div style="height:100%;width:${((val||0)*100).toFixed(0)}%;background:${c}"></div></div></div>`;
    return `<div class="card-row">
      <div class="card"><h3>Risk Regime</h3><div class="sub">Jump-model on equities · scales the whole book</div>
        <div style="padding:14px 18px 18px">
          <div class="dir-chip" style="background:${rc}22;color:${rc};display:inline-block;font-size:15px">${(reg.state||"—").toUpperCase()}</div>
          <div class="mgrid" style="grid-template-columns:1fr 1fr;margin-top:14px">
            <div class="metric"><div class="ml">Exposure scale</div><div class="mv">×${reg.exposure_scale??"—"}</div></div>
            <div class="metric"><div class="ml">Switches / yr</div><div class="mv">${reg.switches_per_year??"—"}</div></div>
            <div class="metric"><div class="ml">Book vol (ann)</div><div class="mv">${bk.book_vol!=null?(bk.book_vol*100).toFixed(1)+"%":"—"}</div></div>
            <div class="metric"><div class="ml">Net invested</div><div class="mv">${bk.invested!=null?(bk.invested*100).toFixed(0)+"%":"—"}</div></div>
          </div></div></div>
      <div class="card"><h3>Sleeve Weights</h3><div class="sub">Equal risk contribution across uncorrelated sleeves</div>
        <div style="padding:14px 18px 18px">
          ${bar("FX reversal",w.fx,"var(--blue)")}
          ${bar("Leverage rotation · "+(bk.lev_regime||""),w.leverage,"var(--gold)")}
          ${bar("Crypto trend · "+(bk.crypto_regime||""),w.crypto,"var(--pos)")}
          <div class="muted" style="font-size:11px;margin-top:8px">Vol target ${bk.target_vol!=null?(bk.target_vol*100).toFixed(0)+"%":"—"} · exposure cap ×${bk.vol_scale??"—"}</div>
        </div></div>
    </div>`;
  })();

  body.innerHTML=`
    <div class="decision-banner">
      <div><div class="muted" style="font-size:11px;text-transform:uppercase;letter-spacing:.6px">Total P&L</div>
        <div style="font-size:30px;font-weight:800;color:${col(totalPnl)}">${usd(totalPnl)}</div></div>
      <div style="border-left:1px solid var(--border);padding-left:20px">
        <div class="muted" style="font-size:11px;text-transform:uppercase;letter-spacing:.6px">Realized</div>
        <div style="font-size:20px;font-weight:750;color:${col(real)}">${usd(real)}</div></div>
      <div style="border-left:1px solid var(--border);padding-left:20px">
        <div class="muted" style="font-size:11px;text-transform:uppercase;letter-spacing:.6px">Unrealized</div>
        <div style="font-size:20px;font-weight:750;color:${col(unreal)}">${usd(unreal)}</div></div>
      <div style="border-left:1px solid var(--border);padding-left:20px">
        <div class="muted" style="font-size:11px;text-transform:uppercase;letter-spacing:.6px">Equity</div>
        <div style="font-size:20px;font-weight:750">${usd(equity)}</div></div>
      <div style="border-left:1px solid var(--border);padding-left:20px">
        <div class="muted" style="font-size:11px;text-transform:uppercase;letter-spacing:.6px">Return</div>
        <div style="font-size:20px;font-weight:750;color:${col(ret)}">${ret>=0?"+":""}${ret.toFixed(3)}%</div></div>
    </div>
    ${bookHTML}
    <div class="card"><h3>Equity Curve</h3><div class="sub">Paper equity, recorded daily — real forward out-of-sample performance</div>
      <div id="liveEq" style="height:280px"></div></div>
    <div class="card-row">
      <div class="card"><h3>Forward Validation</h3><div class="sub">The live proof, accruing daily</div>
        <div class="mgrid" style="grid-template-columns:1fr 1fr">
          <div class="metric"><div class="ml">Days Live</div><div class="mv">${fwd.days_live||0}</div></div>
          <div class="metric"><div class="ml">Inception Return</div><div class="mv ${(fwd.inception_return_pct||0)>=0?'green':'red'}">${(fwd.inception_return_pct||0).toFixed(3)}%</div></div>
          <div class="metric"><div class="ml">Forward Sharpe</div><div class="mv">${fwd.forward_sharpe!=null?fwd.forward_sharpe:"—<span style='font-size:11px'> (need 20d)</span>"}</div></div>
          <div class="metric"><div class="ml">Forward Max DD</div><div class="mv red">${fwd.forward_max_dd!=null?fwd.forward_max_dd+"%":"—"}</div></div>
        </div></div>
      <div class="card"><h3>Portfolio Risk</h3><div class="sub">Correlation-adjusted, whole book</div>
        ${pr.error?`<div class="muted">${pr.error}</div>`:`<div class="mgrid" style="grid-template-columns:1fr 1fr">
          <div class="metric"><div class="ml">Portfolio Vol</div><div class="mv">${pr.portfolio_vol_annual}%</div></div>
          <div class="metric"><div class="ml">VaR 99% (daily)</div><div class="mv red">${pr.var_99_daily}%</div></div>
          <div class="metric"><div class="ml">CVaR 99%</div><div class="mv red">${pr.cvar_99_daily}%</div></div>
          <div class="metric"><div class="ml">Diversification</div><div class="mv blue">${pr.diversification_ratio}×</div></div>
        </div>`}</div>
    </div>
    <div class="card"><h3>Open Positions</h3><div class="sub">Live from Alpaca paper · updated on refresh</div>
      <div style="overflow-x:auto"><table class="labtable"><thead><tr><th>Symbol</th><th>Side</th><th>Qty</th><th>Avg</th><th>Last</th><th>P&L</th><th>P&L&nbsp;%</th></tr></thead>
      <tbody>${rows||`<tr><td colspan=7 class="muted">No open positions.</td></tr>`}</tbody></table></div></div>
    <div class="card-row">
      <div class="card"><h3>Risk Contributions</h3><div class="sub">Who drives the book's risk</div>
        <div style="overflow-x:auto"><table class="labtable"><thead><tr><th>Symbol</th><th>Weight</th><th>Vol</th><th>Risk&nbsp;%</th></tr></thead>
        <tbody>${prRows||`<tr><td colspan=4 class="muted">—</td></tr>`}</tbody></table></div></div>
      <div class="card"><h3>Trade Log</h3><div class="sub">Recent executions</div>
        <div style="overflow-x:auto"><table class="labtable"><thead><tr><th>Symbol</th><th>Side</th><th>Qty</th><th>Price</th></tr></thead>
        <tbody>${trades||`<tr><td colspan=4 class="muted">No trades yet.</td></tr>`}</tbody></table></div></div>
    </div>`;

  const h=d.history||[];
  if(h.length){
    const start=d.start_equity;
    Plotly.newPlot("liveEq",[{x:h.map(x=>x.date),y:h.map(x=>(x.equity/start-1)*100),
      type:"scatter",mode:"lines+markers",line:{color:"#6d97cf",width:2.4},
      fill:"tozeroy",fillcolor:"rgba(109,151,207,.08)"}],
      baseLayout({yaxis:{...axis,ticksuffix:"%"}}),PLOT_CFG);
  } else { document.getElementById("liveEq").innerHTML=`<div class="loading">Equity history builds daily — check back after the next scheduled run.</div>`; }
}

/* ---------- strategy lab ---------- */
function vchip(v){
  const c = v==="robust"?"var(--green)":v==="inconclusive"?"var(--amber)":"var(--red)";
  return `<span style="color:${c};font-weight:700">${v}</span>`;
}
async function loadLab(){
  const body=document.getElementById("labBody");
  body.innerHTML=`<div class="loading">Running DSR-gated search &amp; scanning Kalshi (first load computes backtests)…</div>`;
  let lab, kal;
  try { lab=await (await fetch("/api/strategy_lab")).json(); }
  catch(e){ body.innerHTML=`<div class="loading">Search error: ${e}</div>`; return; }
  const combo=lab.combo, single=lab.single;
  const cw=combo.winner, sw=single.winner;

  const banner = cw
    ? `<div class="decision-banner"><div class="dir-chip dir-bull">ROBUST BLEND</div>
        <div><div style="font-weight:700">${cw.blend}</div>
        <div class="muted" style="font-size:12px">Sharpe ${cw.sharpe} · DSR ${(cw.dsr*100).toFixed(0)}% · avg corr ${cw.avg_corr}</div></div></div>`
    : `<div class="decision-banner"><div class="dir-chip dir-neutral">NO ROBUST BLEND</div>
        <div><div style="font-weight:700">Best so far: ${combo.leaderboard[0].blend}</div>
        <div class="muted" style="font-size:12px">Sharpe ${combo.leaderboard[0].sharpe}, DSR ${(combo.leaderboard[0].dsr*100).toFixed(0)}% (${combo.leaderboard[0].verdict}) — searched ${combo.n_trials} blends. Need more uncorrelated edge.</div></div></div>`;

  const comboRows = combo.leaderboard.slice(0,10).map(r=>`<tr>
    <td>${r.blend}</td><td class="num">${r.sharpe.toFixed(2)}</td>
    <td class="num">${r.avg_corr.toFixed(2)}</td><td class="num">${(r.dsr*100).toFixed(0)}%</td>
    <td>${vchip(r.verdict)}</td><td class="num">${r.max_dd.toFixed(1)}%</td></tr>`).join("");
  const singleRows = single.leaderboard.slice(0,8).map(r=>`<tr>
    <td>${r.strategy}</td><td class="num">${r.sharpe.toFixed(2)}</td>
    <td class="num">${(r.dsr*100).toFixed(0)}%</td><td>${vchip(r.verdict)}</td>
    <td class="num">${r.sharpe_is}→${r.sharpe_oos}</td></tr>`).join("");

  body.innerHTML=`${banner}
    <div class="card"><h3>Combination Search — signal blends</h3>
      <div class="sub">Sharpe deflated for ${combo.n_trials} trials. Low avg-corr = genuinely uncorrelated blend.</div>
      <div style="overflow-x:auto"><table class="labtable"><thead><tr><th>Blend</th><th>Sharpe</th><th>Avg&nbsp;Corr</th><th>DSR</th><th>Verdict</th><th>Max&nbsp;DD</th></tr></thead><tbody>${comboRows}</tbody></table></div></div>
    <div class="card"><h3>Single-Signal Search</h3>
      <div class="sub">${single.n_trials} candidates. IS→OOS shows out-of-sample decay.</div>
      <div style="overflow-x:auto"><table class="labtable"><thead><tr><th>Strategy</th><th>Sharpe</th><th>DSR</th><th>Verdict</th><th>IS→OOS</th></tr></thead><tbody>${singleRows}</tbody></table></div></div>
    ${combo.alt_status ? `<div class="card"><h3>Alternative Streams (news / Kalshi)</h3>
      <div class="sub">Wired into the pool, gated by history — they join the blend once they reach ${combo.alt_status.min_history} days (no look-ahead). Free data gives only weeks today.</div>
      <table class="labtable"><thead><tr><th>Stream</th><th>History</th><th>Status</th></tr></thead><tbody>${
        combo.alt_status.streams.map(r=>`<tr><td>${r.stream}</td><td class="num">${r.days} days</td>
        <td>${r.eligible?'<span style="color:var(--green);font-weight:700">eligible</span>':'<span style="color:var(--amber)">'+r.status+'</span>'}</td></tr>`).join("")
      }</tbody></table></div>` : ""}
    <div class="card" id="kalshiCard"><h3>Kalshi Arbitrage Scanner</h3>
      <div class="sub">Near-riskless prediction-market locks (yes_ask + no_ask &lt; $1), net of fees. Uncorrelated with price signals.</div>
      <div id="kalshiBody"><div class="loading">Scanning live Kalshi markets…</div></div></div>`;

  try { kal=await (await fetch("/api/kalshi")).json(); } catch(e){ kal={error:String(e)}; }
  const kb=document.getElementById("kalshiBody");
  if(kal.error){ kb.innerHTML=`<div class="muted">Scanner unavailable: ${kal.error}</div>`; return; }
  if(!kal.available){ kb.innerHTML=`<div class="muted">Market data unavailable right now.</div>`; return; }
  if(kal.n_buy_both>0){
    kb.innerHTML=`<table class="labtable"><thead><tr><th>Market</th><th>Yes+No</th><th>Net&nbsp;¢</th><th>Vol</th></tr></thead><tbody>`+
      kal.buy_both.map(x=>`<tr><td>${x.ticker}</td><td class="num">${x.yes_ask}+${x.no_ask}=${x.cost_cents}¢</td>
      <td class="num" style="color:var(--green)">+${x.net_profit_cents}¢</td><td class="num">${x.volume}</td></tr>`).join("")+
      `</tbody></table><div class="muted" style="margin-top:8px;font-size:12px">${kal.note}</div>`;
  } else {
    kb.innerHTML=`<div style="padding:8px 0"><span style="color:var(--green);font-weight:700">✓ No arbitrage right now</span>
      <span class="muted"> — scanned ${kal.priced} liquid markets across ${kal.events_seen} events; all efficiently priced (the normal, honest case). The scanner will flag a lock the moment one appears.</span></div>`;
  }
}

loadStrategies();
