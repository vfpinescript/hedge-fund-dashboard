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
document.querySelectorAll(".tab").forEach(t=>t.onclick=()=>{
  document.querySelectorAll(".tab").forEach(x=>x.classList.remove("active"));
  t.classList.add("active");
  ["strategies","live","builder","signals","lab","surfaces"].forEach(v=>
    document.getElementById("view-"+v).classList.toggle("hidden", v!==t.dataset.tab));
  if(t.dataset.tab==="live" && !window._liveLoaded){ loadLive(); window._liveLoaded=1; }
  if(t.dataset.tab==="builder" && !window._builderLoaded){ loadBuilder(); window._builderLoaded=1; }
  if(t.dataset.tab==="signals" && !window._sigLoaded){ loadSignals(); window._sigLoaded=1; }
  if(t.dataset.tab==="lab" && !window._labLoaded){ loadLab(); window._labLoaded=1; }
  if(t.dataset.tab==="surfaces" && !window._surfLoaded){ loadSurfaces(); window._surfLoaded=1; }
});

/* ---------- strategies ---------- */
const METRICS = [
  ["total_return","Total Return","pct","blue"],["max_drawdown","Max Drawdown","pct","red"],
  ["sharpe","Sharpe Ratio","num",""],["win_rate","Win Rate","pct",""],
  ["profit_factor","Profit Factor","num",""],["sortino","Sortino Ratio","num",""],
  ["avg_win","Avg Win","pct","blue"],["avg_loss","Avg Loss","pct","red"],
  ["risk_reward","Risk–Reward Ratio","num",""],["max_win_streak","Max Win Streak","int",""],
  ["time_in_dd","Time in DD","pct",""],["max_loss_streak","Max Losing Streak","int",""],
];
let STRATS=[];

async function loadStrategies(){
  STRATS = await (await fetch("/api/strategies")).json();
  const list = document.getElementById("stratList");
  const groups = {};
  STRATS.forEach((s,i)=>{ (groups[s.category]=groups[s.category]||[]).push({...s,_i:i}); });
  let html="";
  for(const cat of Object.keys(groups)){
    html+=`<div class="list-head">${cat}</div>`;
    for(const s of groups[cat]){
      const sh=s.metrics.sharpe||0, cls=sh>=0?"pos":"neg";
      html+=`<div class="strat-item" data-i="${s._i}">
        <div class="si-top"><span class="si-name">${s.name}</span>
        <span class="badge ${cls}">Sharpe ${sh.toFixed(2)}</span></div>
        <div class="si-cat">${s.category}</div></div>`;
    }
  }
  list.innerHTML=html;
  list.querySelectorAll(".strat-item").forEach(el=>el.onclick=()=>selectStrategy(+el.dataset.i, el));
  const first=list.querySelector(".strat-item"); if(first) selectStrategy(0, first);
}

function selectStrategy(i, el){
  document.querySelectorAll(".strat-item").forEach(x=>x.classList.remove("active"));
  el.classList.add("active");
  renderDetail(STRATS[i]);
}

function overfitChip(o){
  if(!o || o.dsr==null || isNaN(o.dsr)) return "";
  const col = o.verdict==="robust"?"var(--green)":o.verdict==="inconclusive"?"var(--amber)":"var(--red)";
  return `<div style="text-align:right">
    <div style="font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.6px">Overfitting check · DSR</div>
    <div style="font-size:22px;font-weight:750;color:${col}">${(o.dsr*100).toFixed(1)}%</div>
    <div style="font-size:11px;color:${col};font-weight:600">${o.verdict}</div>
    <div style="font-size:10.5px;color:var(--muted2)" title="Deflated Sharpe: prob. true Sharpe>0 after ${o.n_trials} trials">
      SR ${o.sharpe_ann.toFixed(2)} vs bar SR* ${o.sr_star_ann.toFixed(2)} · ${o.n_trials} trials</div>
  </div>`;
}

function gaugeSVG(score){
  const r=54, cx=65, cy=65, circ=Math.PI*r, frac=score/100;
  const off=circ*(1-frac);
  return `<svg class="gauge-svg" viewBox="0 0 130 130">
    <path d="M 11 65 A 54 54 0 0 1 119 65" fill="none" stroke="#24262c" stroke-width="9" stroke-linecap="round"/>
    <path d="M 11 65 A 54 54 0 0 1 119 65" fill="none" stroke="#6d97cf" stroke-width="9"
      stroke-linecap="round" stroke-dasharray="${circ}" stroke-dashoffset="${off}"/>
    <text x="65" y="60" text-anchor="middle" fill="#e6eaf1" font-size="30" font-weight="750">${score}</text>
    <text x="65" y="80" text-anchor="middle" fill="#7c8698" font-size="12">${score>=60?"Viable":score>=40?"Marginal":"Weak"}</text>
  </svg>`;
}

function renderDetail(s){
  const m=s.metrics;
  const metricHTML = METRICS.map(([k,label,fmt,color])=>{
    let v=m[k], disp = fmt==="pct"?fmtPct(v):fmt==="int"?fmtInt(v):fmtNum(v);
    let c=color; if(!c && (fmt==="pct")) c = (v>=0?"":"red");
    return `<div class="metric"><div class="ml">${label}</div><div class="mv ${c}">${disp}</div></div>`;
  }).join("");
  const tr=m.total_return, trCls=tr>=0?"blue":"red";
  document.getElementById("stratDetail").innerHTML=`
    <div class="detail-head">
      <div><h1>${s.name}</h1><div class="desc">${s.description}</div></div>
      ${overfitChip(s.overfit)}
    </div>
    <div class="mgrid" style="margin-bottom:30px">${metricHTML}</div>
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:baseline">
        <h3>Total Profit &amp; Loss</h3>
        <span class="muted" style="font-size:12px">Strategy vs Buy &amp; Hold</span>
      </div>
      <div class="pnl-big" style="color:var(--${trCls})">${fmtPct(tr)}
        <small>net of costs · long/short · ${s.equity.dates.length} days</small></div>
      <div class="chart" id="eqChart" style="height:340px"></div>
    </div>
    <div class="card"><h3>Monthly Returns</h3><div class="sub">Calendar performance heatmap</div>
      <div id="heat"></div></div>
    <div class="card-row">
      <div class="card"><h3>Drawdown</h3><div class="sub">Underwater equity</div>
        <div class="chart" id="ddChart" style="height:240px"></div></div>
      <div class="card"><h3>Rolling Sharpe</h3><div class="sub">126-day window</div>
        <div class="chart" id="rsChart" style="height:240px"></div></div>
    </div>
    ${s.trades?`<div class="card"><h3>Trade Distribution</h3>
      <div class="chart" id="tradeChart" style="height:260px"></div></div>`:""}`;

  // equity
  const traces=[{x:s.equity.dates,y:s.equity.values,type:"scatter",mode:"lines",
    line:{color:"#6d97cf",width:2.4},fill:"tozeroy",fillcolor:"rgba(109,151,207,.08)",name:"Strategy"}];
  if(s.buyhold){ traces.push({x:s.buyhold.dates,y:s.buyhold.values,type:"scatter",mode:"lines",
    name:"Buy & Hold",line:{color:"#7c8698",width:1.5,dash:"dash"}}); }
  if(s.components){ const cols=["#cf9f45","#4fae82","#c6a15b","#d76b63","#6d97cf","#c6a15b"];
    Object.entries(s.components).forEach(([k,c],i)=>traces.push({x:c.dates,y:c.values,type:"scatter",
      mode:"lines",line:{color:cols[i%cols.length],width:1},opacity:.5,name:k})); }
  Plotly.newPlot("eqChart",traces,baseLayout({yaxis:{...axis,ticksuffix:"%"},showlegend:true,
    legend:{font:{size:11},orientation:"h",y:1.1}}),PLOT_CFG);

  // heatmap
  renderHeat(s.monthly);

  // drawdown
  Plotly.newPlot("ddChart",[{x:s.drawdown.dates,y:s.drawdown.values,type:"scatter",mode:"lines",
    line:{color:"#d76b63",width:1},fill:"tozeroy",fillcolor:"rgba(215,107,99,.25)"}],
    baseLayout({yaxis:{...axis,ticksuffix:"%"}}),PLOT_CFG);

  // rolling sharpe
  Plotly.newPlot("rsChart",[{x:s.rolling_sharpe.dates,y:s.rolling_sharpe.values,type:"scatter",
    mode:"lines",line:{color:"#cf9f45",width:1.3}}],
    baseLayout({shapes:[{type:"line",x0:s.rolling_sharpe.dates[0],x1:s.rolling_sharpe.dates.slice(-1)[0],
      y0:1,y1:1,line:{color:"#6d97cf",width:1,dash:"dash"}}]}),PLOT_CFG);

  // trade donut
  if(s.trades){
    Plotly.newPlot("tradeChart",[{values:[s.trades.long,s.trades.short],labels:["Long","Short"],
      type:"pie",hole:.62,marker:{colors:["#6d97cf","#d76b63"]},textinfo:"label+value",
      textfont:{color:"#fff",size:13}}],
      baseLayout({margin:{l:10,r:10,t:10,b:10},annotations:[{text:`${s.trades.total}<br>trades`,
        showarrow:false,font:{size:16,color:"#e6eaf1"}}]}),PLOT_CFG);
  }
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

/* ---------- strategy builder ---------- */
async function loadBuilder(){
  const body=document.getElementById("builderBody");
  let lib;
  try{ lib=await (await fetch("/api/strategy_library")).json(); }
  catch(e){ body.innerHTML=`<div class="loading">Error: ${e}</div>`; return; }
  const stratOpts=lib.strategies.map(s=>`<option value="${s.id}" title="${s.desc}">${s.name}</option>`).join("");
  const instOpts=Object.entries(lib.instrument_groups).map(([g,syms])=>
    `<optgroup label="${g}">`+syms.map(s=>`<option value="${s}">${s}</option>`).join("")+`</optgroup>`).join("");
  body.innerHTML=`
    <div class="card" style="display:flex;gap:16px;align-items:flex-end;flex-wrap:wrap">
      <div><div class="ml" style="margin-bottom:6px">Strategy</div>
        <select id="bStrat" class="select" style="min-width:230px">${stratOpts}</select></div>
      <div><div class="ml" style="margin-bottom:6px">Instrument</div>
        <select id="bInst" class="select" style="min-width:180px">${instOpts}</select></div>
      <div><div class="ml" style="margin-bottom:6px">…or any ticker</div>
        <input id="bCustom" class="select" placeholder="e.g. AAPL, TLT, GLD" style="min-width:150px"></div>
      <button id="bRun" class="dir-chip dir-bull" style="cursor:pointer;border:none;font-size:15px;padding:11px 24px">▶ Run Backtest</button>
      <div id="bStratDesc" class="muted" style="font-size:12.5px;flex-basis:100%"></div>
    </div>
    <div id="bResult"></div>`;
  const descOf=()=>lib.strategies.find(s=>s.id===document.getElementById("bStrat").value)?.desc||"";
  const showDesc=()=>document.getElementById("bStratDesc").textContent=descOf();
  document.getElementById("bStrat").onchange=showDesc; showDesc();
  document.getElementById("bRun").onclick=runBacktest;
  document.getElementById("bCustom").addEventListener("keydown",e=>{ if(e.key==="Enter") runBacktest(); });
}
async function runBacktest(){
  const strat=document.getElementById("bStrat").value;
  const custom=document.getElementById("bCustom").value.trim();
  const inst=custom||document.getElementById("bInst").value;
  const out=document.getElementById("bResult");
  out.innerHTML=`<div class="loading">Backtesting ${strat} on ${inst}…</div>`;
  let d;
  try{ d=await (await fetch(`/api/strategy_run?strategy=${encodeURIComponent(strat)}&instrument=${encodeURIComponent(inst)}`)).json(); }
  catch(e){ out.innerHTML=`<div class="loading">Error: ${e}</div>`; return; }
  if(d.error){ out.innerHTML=`<div class="card"><div class="muted">Couldn't run: ${d.error}</div></div>`; return; }
  const m=d.metrics;
  const dcol=d.dsr_verdict==="robust"?"var(--green)":d.dsr_verdict==="inconclusive"?"var(--amber)":"var(--red)";
  const c=v=>v>=0?"blue":"red";
  const M=[["total_return","Total Return","pct"],["sharpe","Sharpe","num"],["sortino","Sortino","num"],
    ["max_drawdown","Max Drawdown","pct"],["win_rate","Win Rate","pct"],["profit_factor","Profit Factor","num"],
    ["risk_reward","Risk-Reward","num"],["time_in_dd","Time in DD","pct"]];
  const grid=M.map(([k,l,f])=>{const v=m[k];const disp=f==="pct"?fmtPct(v):fmtNum(v);
    return `<div class="metric"><div class="ml">${l}</div><div class="mv ${f==='pct'?c(v):''}">${disp}</div></div>`;}).join("");
  out.innerHTML=`
    <div class="decision-banner">
      <div class="dir-chip" style="background:${dcol}22;color:${dcol};font-size:15px">DSR ${(d.dsr*100).toFixed(0)}%</div>
      <div><div style="font-weight:750;font-size:16px">${d.strategy} · ${d.instrument}</div>
        <div class="muted" style="font-size:12.5px">Overfitting check: <b style="color:${dcol}">${d.dsr_verdict}</b> — is this edge real or noise?</div></div>
      <div style="margin-left:auto;text-align:right">
        <div class="muted" style="font-size:11px;text-transform:uppercase;letter-spacing:.6px">Total Return</div>
        <div style="font-size:26px;font-weight:800;color:var(--${c(m.total_return)})">${fmtPct(m.total_return)}</div></div>
    </div>
    <div class="card"><h3>Equity Curve</h3><div class="sub">Strategy vs Buy &amp; Hold · ${d.equity.dates.length} days</div>
      <div id="bEq" style="height:320px"></div></div>
    <div class="card"><h3>Performance</h3><div class="mgrid" style="margin-top:4px">${grid}</div></div>`;
  Plotly.newPlot("bEq",[
    {x:d.equity.dates,y:d.equity.strategy,type:"scatter",mode:"lines",name:"Strategy",line:{color:"#6d97cf",width:2.4},fill:"tozeroy",fillcolor:"rgba(109,151,207,.08)"},
    {x:d.equity.dates,y:d.equity.buyhold,type:"scatter",mode:"lines",name:"Buy & Hold",line:{color:"#7c8698",width:1.5,dash:"dash"}}
  ],baseLayout({yaxis:{...axis,ticksuffix:"%"},showlegend:true,legend:{orientation:"h",y:1.1,font:{size:11}}}),PLOT_CFG);
}

/* ---------- live paper ---------- */
document.getElementById("liveRefresh").onclick=()=>loadLive();
const usd = v => (v==null||isNaN(v)) ? "–" : (v<0?"-$":"$")+Math.abs(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
async function loadLive(){
  const body=document.getElementById("liveBody");
  body.innerHTML=`<div class="loading">Reading the live TradingView paper account…</div>`;
  let d;
  try{ d=await (await fetch("/api/live")).json(); }
  catch(e){ body.innerHTML=`<div class="loading">Error: ${e}</div>`; return; }
  if(d.error){ body.innerHTML=`<div class="loading">Snapshot unavailable: ${d.error}<br><span class="muted" style="font-size:12px">Is TradingView running with the paper broker connected?</span></div>`; return; }
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
    <div class="card"><h3>Open Positions</h3><div class="sub">Live from TradingView paper · updated on refresh</div>
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

/* ---------- surfaces ---------- */
const SURF_SCALE=[[0,"#1a1140"],[.3,"#7a2a8f"],[.55,"#c73e6b"],[.78,"#f5844a"],[1,"#f9e07f"]];
async function loadSurfaces(){
  const sel=fillSelect("surfInstrument"); sel.onchange=()=>renderSurfaces(sel.value);
  renderSurfaces(sel.value);
}
function surfDiv(id,title,sub){return `<div class="card"><h3>${title}</h3><div class="sub">${sub}</div>
  <div id="${id}" style="height:360px"></div></div>`;}
function plotSurface(id,S,scale){
  if(!S){document.getElementById(id).innerHTML=`<div class="loading">No data feed available — abstained.</div>`;return;}
  Plotly.newPlot(id,[{type:"surface",x:S.x,y:S.y,z:S.z,colorscale:scale||SURF_SCALE,showscale:false,
    contours:{z:{show:true,usecolormap:true,project:{z:false}}}}],
    {paper_bgcolor:"transparent",font:{color:"#7c8698",size:10},margin:{l:0,r:0,t:0,b:0},
     scene:{xaxis:{title:S.xlabel,gridcolor:"#2a3341",backgroundcolor:"transparent",showbackground:false},
       yaxis:{title:S.ylabel,gridcolor:"#2a3341",backgroundcolor:"transparent",showbackground:false},
       zaxis:{title:S.zlabel,gridcolor:"#2a3341",backgroundcolor:"transparent",showbackground:false},
       camera:{eye:{x:1.6,y:-1.5,z:.9}}}},PLOT_CFG);
}
async function renderSurfaces(inst){
  const body=document.getElementById("surfacesBody");
  body.innerHTML=`<div class="loading">Computing surfaces for ${inst} (fetching option chain…)</div>`;
  const S=await (await fetch("/api/surfaces?instrument="+encodeURIComponent(inst))).json();
  if(S.error){body.innerHTML=`<div class="loading">Error: ${S.error}</div>`;return;}
  body.innerHTML=`<div class="surf-grid">
    ${surfDiv("sTail","Tail-Exponent Surface","Hill index α over window × order-statistic k — fat-tail risk ("+inst+")")}
    ${surfDiv("sVol","Volatility Surface","Implied vol over strike × expiry — SPX/SPY options")}
    ${surfDiv("sGamma","Gamma Surface","∂²V/∂S² over spot × maturity — Black-Scholes")}
    ${surfDiv("sCharm","Charm Surface","∂Δ/∂t over spot × maturity — dealer decay")}
  </div>
  <div class="card"><h3>Return Distribution Across Horizons</h3>
    <div class="sub">Densities (σ units) — the fat left tail Gaussian VaR misses ("+inst+")</div>
    <div id="sDist" style="height:300px"></div></div>
  <div class="card"><h3>Yield-Curve PCA</h3><div class="sub">Level / Slope / Curvature — live Treasury tenors</div>
    <div id="sPca" style="height:120px" ></div><div id="pcaText"></div></div>`;

  plotSurface("sTail",S.tail);
  plotSurface("sVol",S.vol,"Viridis");
  plotSurface("sGamma",S.gamma);
  plotSurface("sCharm",S.charm,[[0,"#0b2b4a"],[.5,"#2a7de1"],[1,"#f9e07f"]]);

  // return distribution
  const cols=["#6d97cf","#4fae82","#cf9f45","#d76b63","#c6a15b"];
  const dt=S.return_dist.map((h,i)=>({x:h.x,y:h.y,type:"scatter",mode:"lines",name:h.horizon+"d",
    line:{color:cols[i%cols.length],width:1.6}}));
  // gaussian ref
  const gx=S.return_dist[0].x, gy=gx.map(v=>Math.exp(-v*v/2)/Math.sqrt(2*Math.PI));
  dt.push({x:gx,y:gy,type:"scatter",mode:"lines",name:"Gaussian",line:{color:"#57606f",width:1.4,dash:"dash"}});
  Plotly.newPlot("sDist",dt,baseLayout({showlegend:true,legend:{orientation:"h",y:1.12,font:{size:11}},
    xaxis:{...axis,title:"σ"},margin:{l:44,r:16,t:20,b:34}}),PLOT_CFG);

  // pca
  const p=S.pca, ev=p.explained_variance_ratio;
  Plotly.newPlot("sPca",[{x:p.factor_names,y:ev.map(x=>x*100),type:"bar",
    marker:{color:["#6d97cf","#4fae82","#cf9f45"]},text:ev.map(x=>(x*100).toFixed(1)+"%"),textposition:"outside",
    textfont:{color:"#e6eaf1"}}],baseLayout({yaxis:{...axis,ticksuffix:"%"},margin:{l:44,r:10,t:20,b:24}}),PLOT_CFG);
  document.getElementById("pcaText").innerHTML=`<div class="muted" style="font-size:12.5px;margin-top:8px">
    Current curve: `+Object.entries(p.current_curve_bps).map(([k,v])=>`${k} ${v}bps`).join(" · ")+`</div>`;
}

loadStrategies();
