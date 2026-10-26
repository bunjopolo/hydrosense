"use client";

import { useEffect, useRef, useState } from "react";

type Asset = { id: string; name: string; kind: string; tags: string[]; running?: { tag: string; min: number } };
type Config = { source_timezone?: string; timestamp_column: string; poll_seconds: number; stale_seconds: number; max_skew_seconds: number; persistence: number; lookback: number; horizon: number; epochs: number; batch_size: number; hidden_size: number; device: string; assets: Asset[] };
type Forecast = { timestamp: string; expected: number; lower: number; upper: number };
type Signal = { tag: string; actual: number | null; expected: number | null; residual: number | null; tolerance: number | null; ratio: number | null; status: string; consecutive: number; forecast?: Forecast[] };
type Result = { id: string; name: string; kind: string; status: string; detail: string; score: number | null; signals: Signal[]; timestamp: string; components: { isolation_forest?: number; lstm_autoencoder?: number } };
type Report = { id: string; name: string; rows: number; excluded_rows: number; fit_rows: number; calibration_rows: number; test_rows: number; test_windows: number; test_flag_rate: number; tags: { tag: string; mae: number; baseline_mae: number; tolerance: number }[] };
type Connection = { url?: string; data_server?: string; auth?: string; username?: string; ca_bundle?: string; web_ids?: Record<string, string>; has_secret?: boolean };
type Notice = { id: string; timestamp: string; unit: string; tag: string; status: string; actual: number | null; expected: number | null; ratio: number | null };
type State = { config: Config; connection: Connection; job: { status: string; error?: string; message?: string }; mode: string; source: string; error: string | null; migration_note?: string; results: Result[]; history: { timestamp: string; assets: Result[] }[]; events: Notice[]; has_export: boolean; sample_seconds: number | null; updated_at: string | null; training_rows: number; replay_rows: number; replay_index: number; model: { created_at: string; reports: Report[]; compatible: boolean } | null };
type Inspection = { columns: string[]; numeric_columns: string[]; rows: number };
const API = "http://127.0.0.1:8765";
const COLAB = "https://colab.research.google.com/github/bunjopolo/hydrosense/blob/main/notebooks/hydrosense_training.ipynb";
const names: Record<string, string> = { normal: "Within baseline", deviation: "Deviation", alert: "Persistent deviation", data_quality: "Data quality", not_running: "Not running", warming_up: "Collecting history" };
const fmt = (n: number | null | undefined) => n == null ? "—" : n.toLocaleString(undefined, { maximumFractionDigits: 3 });

async function request<T = { ok?: boolean; message?: string }>(path: string, data?: unknown) {
  const response = await fetch(API + path, data === undefined ? undefined : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
  const body = await response.json() as T & { error?: string };
  if (!response.ok) throw new Error(body.error || "Request failed");
  return body;
}

function Trace({ history, unitId, tag, future }: { history: State["history"]; unitId: string; tag: string; future: Forecast[] }) {
  const points = history.map(h => ({ time: Date.parse(h.timestamp), signal: h.assets.find(a => a.id === unitId)?.signals.find(s => s.tag === tag) }));
  const values = [...points.flatMap(p => [p.signal?.actual, p.signal?.expected]), ...future.flatMap(p => [p.lower, p.upper, p.expected])].filter((n): n is number => n != null && Number.isFinite(n));
  if (!values.length) return <div className="chart-empty">Waiting for tag observations. This graph will update with every prediction.</div>;
  const min = Math.min(...values), max = Math.max(...values), span = Math.max(max - min, .001);
  const start = points[0]?.time ?? Date.parse(future[0].timestamp);
  const now = points[points.length - 1]?.time ?? start;
  const end = future.length ? Date.parse(future[future.length - 1].timestamp) : now;
  const x = (t: number) => 48 + (t - start) / Math.max(end - start, 1000) * 500;
  const y = (v: number) => 149 - (v - min) / span * 120;
  const line = (field: "actual" | "expected") => points.map((p, i) => p.signal?.[field] != null ? `${i === 0 || points[i - 1].signal?.[field] == null ? "M" : "L"}${x(p.time)},${y(p.signal[field])}` : "").join(" ");
  const forecastPath = future.map((p, i) => `${i ? "L" : "M"}${x(Date.parse(p.timestamp))},${y(p.expected)}`).join(" ");
  const band = [...future.map(p => `${x(Date.parse(p.timestamp))},${y(p.upper)}`), ...[...future].reverse().map(p => `${x(Date.parse(p.timestamp))},${y(p.lower)}`)].join(" ");
  return <svg className="trace tag-trace" viewBox="0 0 580 185" role="img" aria-label={`Actual and forecast graph for ${tag}`}>
    {[0, 1, 2].map(i => <g key={i}><line x1="48" x2="548" y1={29 + i * 60} y2={29 + i * 60} stroke="#28404b" strokeDasharray="3 5"/><text x="40" y={33 + i * 60} textAnchor="end">{fmt(max - i * span / 2)}</text></g>)}
    {future.length > 0 && <><rect x={x(now)} y="20" width={Math.max(0, 548 - x(now))} height="138" fill="#63d8c4" opacity=".035"/><polygon points={band} fill="#63d8c4" opacity=".13"/><line x1={x(now)} x2={x(now)} y1="20" y2="158" stroke="#718c99" strokeDasharray="2 4"/><text x={Math.min(x(now) + 5, 495)} y="15">FORECAST →</text></>}
    <path d={line("actual")} fill="none" stroke="#8abaff" strokeWidth="2"/>
    <path d={line("expected")} fill="none" stroke="#d8b577" strokeWidth="1.6" strokeDasharray="4 3"/>
    <path d={forecastPath} fill="none" stroke="#63d8c4" strokeWidth="2" strokeDasharray="5 3"/>
    <text x="48" y="177">{new Date(start).toLocaleTimeString()}</text><text x="548" y="177" textAnchor="end">{new Date(end).toLocaleTimeString()}</text>
  </svg>;
}

export default function Home() {
  const [state, setState] = useState<State | null>(null);
  const [tab, setTab] = useState("Monitor");
  const [offline, setOffline] = useState(false);
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [historyCount, setHistoryCount] = useState(120);
  const [seek, setSeek] = useState(0);
  const [selected, setSelected] = useState("");
  const [configText, setConfigText] = useState("");
  const [connection, setConnection] = useState<Connection>({ auth: "basic" });
  const [secret, setSecret] = useState("");
  const [webIds, setWebIds] = useState("{}");
  const [fileData, setFileData] = useState<Record<string, string> | null>(null);
  const [fileName, setFileName] = useState("");
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [healthy, setHealthy] = useState(false);
  const [unitName, setUnitName] = useState("Hydro unit 1");
  const [timestampColumn, setTimestampColumn] = useState("timestamp");
  const [sheet, setSheet] = useState("");
  const [sourceTimezone, setSourceTimezone] = useState("UTC");
  const initialized = useRef(false);
  const ticking = useRef(false);
  const jobStatus = useRef("");
  async function refresh() {
    const next = await request<State>("/status");
    setState(next); setOffline(false);
    if (jobStatus.current === "training" && next.job.status === "complete") setNotice("Training complete. Every configured tag now has a forecast model.");
    jobStatus.current = next.job.status;
    if (!initialized.current) {
      setConfigText(JSON.stringify(next.config, null, 2)); setTimestampColumn(next.config.timestamp_column);
      setSourceTimezone(next.config.source_timezone || "UTC"); setConnection({ auth: "basic", ...next.connection });
      setWebIds(JSON.stringify(next.connection.web_ids || {}, null, 2)); initialized.current = true;
    }
    return next;
  }
  useEffect(() => {
    const poll = () => { refresh().catch(() => setOffline(true)); };
    poll(); const timer = setInterval(poll, 2500); return () => clearInterval(timer);
  }, []);
  async function act<T = { ok?: boolean; message?: string }>(path: string, data: unknown = {}, success = "") {
    setBusy(true); setNotice("");
    try { const result = await request<T & { message?: string }>(path, data); await refresh(); setNotice(success || result.message || ""); return result; }
    catch (error) { setNotice(error instanceof Error ? error.message : String(error)); return null; }
    finally { setBusy(false); }
  }
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(async () => {
      if (ticking.current) return;
      ticking.current = true;
      try { await request("/replay", { steps: speed }); const next = await refresh(); if (next.replay_index >= next.replay_rows) setPlaying(false); }
      catch (error) { setPlaying(false); setNotice(String(error)); }
      finally { ticking.current = false; }
    }, 1000);
    return () => clearInterval(timer);
  }, [playing, speed]);
  const config = state?.config;
  const unit = config?.assets.find(a => a.id === selected) || config?.assets[0];
  const result = state?.results.find(r => r.id === unit?.id);
  const disabled = busy || offline || state?.job.status === "training" || state?.mode === "live" || playing;
  const deviations = result?.signals.filter(s => s.status === "alert" || s.status === "deviation") || [];
  const history = (state?.history || []).slice(-historyCount);
  const payload = { ...fileData, timestamp_column: timestampColumn, source_timezone: sourceTimezone, sheet };
  async function loadFile(file?: File) {
    if (!file) return;
    if (file.size > 45 * 1024 * 1024) { setNotice("Use the notebook or CLI for files larger than 45 MB."); return; }
    setInspection(null); setHealthy(false); setFileName(file.name);
    let content: Record<string, string>;
    if (/\.xlsx$/i.test(file.name)) {
      const buffer = new Uint8Array(await file.arrayBuffer()); let binary = "";
      for (let i = 0; i < buffer.length; i += 8192) binary += String.fromCharCode(...buffer.subarray(i, i + 8192));
      content = { xlsx: btoa(binary) };
    } else content = { csv: await file.text() };
    setFileData(content);
    const info = await act<Inspection>("/csv", { ...content, purpose: "inspect", timestamp_column: timestampColumn, source_timezone: sourceTimezone, sheet });
    if (info) setInspection(info);
  }
  async function importTraining() {
    const saved = await act("/csv", { ...payload, purpose: "training", auto_configure: true, unit_name: unitName }, "All numeric tags imported for forecasting. Review training settings, then train.");
    if (saved) { const next = await refresh(); setConfigText(JSON.stringify(next.config, null, 2)); setSelected(""); }
  }

  return <main className="shell"><aside className="sidebar">
    <div className="brand"><span className="brand-icon">≈</span><div>HydroSense<small>PREDICTIVE MAINTENANCE</small></div></div>
    <span className="nav-label">PILOT WORKSPACE</span><nav>{["Monitor", "Data & training", "Units & tags", "PI connection"].map((item, i) => <button key={item} className={tab === item ? "nav-item active" : "nav-item"} onClick={() => setTab(item)}><span>{["◉", "▤", "⌘", "⇄"][i]}</span>{item}</button>)}</nav>
    <div className="sidebar-note"><span className="dot"/> ALL-TAG FORECASTING<p>TFT predictions<br/>LSTM + Isolation Forest</p><small>Historical replay · Live PI<br/>Engineering review</small></div>
  </aside><section className="workspace">
    <header><span>Hydro pilot <span className="slash">/</span> {tab}</span><span className={`pill ${offline ? "amber" : "mint"}`}>{offline ? "Service offline" : state ? "Local service connected" : "Connecting…"}</span></header>
    <div className="page-title"><div><div className="eyebrow">PREDICT EVERY VARIABLE. INVESTIGATE EVERY DEVIATION.</div><h1>{tab === "Monitor" ? "Your whole unit, in view." : tab}</h1><p>{tab === "Monitor" ? "Every tag is forecast together. Each deviation is identified individually." : tab === "Data & training" ? "Import unit history. All numeric columns become forecast targets automatically." : tab === "Units & tags" ? "Group related tags into units and configure the forecast history and horizon." : "Enter your PI details locally when you are ready for the parallel pilot."}</p></div>{tab === "Monitor" && <button className="primary" disabled={disabled} onClick={async () => { const r = await act("/demo", {}, "Training the synthetic all-tag demo…"); if (r) { const s = await refresh(); setConfigText(JSON.stringify(s.config, null, 2)); setSelected(""); } }}>Load synthetic demo</button>}</div>
    {offline && <div role="alert" className="banner warning">The local Python service is offline. Start it with the dashboard launcher; predictions are not updating.</div>}
    {notice && <div role="status" className="banner">{notice}<button onClick={() => setNotice("")} aria-label="Dismiss message">×</button></div>}
    {state?.migration_note && <div className="banner warning">{state.migration_note}</div>}
    {state?.error && <div role="alert" className="banner warning">PI monitoring interrupted: {state.error}</div>}
    {state?.job.status === "training" && <div className="banner">{state.job.message || "Preparing training…"}</div>}
    {state?.job.status === "failed" && <div role="alert" className="banner warning">Training failed: {state.job.error}</div>}
    {state?.model && !state.model.compatible && <div className="banner warning">Configuration changed. Retrain or activate a matching notebook model.</div>}

    {tab === "Monitor" && <>
      <div className="unit-toolbar"><label>Unit<select value={unit?.id || ""} onChange={e => setSelected(e.target.value)}>{config?.assets.map(a => <option key={a.id} value={a.id}>{a.name}</option>)}</select></label><label>Visible history<select value={historyCount} onChange={e => setHistoryCount(Number(e.target.value))}>{[30, 60, 120, 360].map(n => <option key={n} value={n}>{n} observations</option>)}</select></label><div className="source-label"><span className="pill">{state?.mode === "live" ? "LIVE PI" : state?.source === "synthetic" ? "SYNTHETIC DEMO" : state?.mode === "replay" ? "FILE REPLAY" : "NOT RUNNING"}</span><small>{state?.updated_at ? `Last observation ${new Date(state.updated_at).toLocaleString()}` : "Awaiting observations"}</small></div></div>
      <div className="stats"><div><span>Variables in this unit</span><strong>{unit?.tags.length ?? "—"}</strong><small>Every tag has its own graph</small></div><div><span>Tags with future predictions</span><strong>{result?.signals.filter(s => s.forecast?.length).length || 0}<em> / {unit?.tags.length || 0}</em></strong><small>{config?.horizon ?? 12} future samples per tag</small></div><div><span>Variables to investigate</span><strong className="amber-text">{offline || state?.error ? "—" : deviations.length}</strong><small>Independent per-tag deviations</small></div><div><span>Forecast horizon</span><strong className="text-stat">{fmt(((state?.sample_seconds || config?.poll_seconds || 30) * (config?.horizon || 12)) / 60)} min</strong><small>{config?.lookback || 24} past samples → future values</small></div></div>
      {result?.detail && <p className="monitor-detail">{result.detail}</p>}
      {deviations.length > 0 && <div className="deviation-strip" role="status"><strong>Investigate {deviations.length} {deviations.length === 1 ? "variable" : "variables"}</strong>{deviations.map(s => <a key={s.tag} href={`#tag-${unit?.id}-${unit?.tags.indexOf(s.tag)}`}>{s.tag} <b>{fmt(s.ratio)}×</b></a>)}</div>}
      <div className="panel replay-panel"><div className="replay-controls"><button disabled={!state?.replay_rows || offline || busy || state?.job.status === "training" || state.mode === "live" || !state.model?.compatible} onClick={() => setPlaying(!playing)}>{playing ? "Pause" : "Play replay"}</button><button disabled={disabled || !state?.replay_rows || !state.model?.compatible} onClick={() => act("/replay")}>Step</button><button disabled={disabled || !state?.replay_rows || !state.model?.compatible} onClick={() => act("/replay", { reset: true })}>Restart</button><label>Replay speed<select aria-label="Replay speed" value={speed} onChange={e => setSpeed(Number(e.target.value))}>{[1, 2, 5, 10].map(n => <option key={n} value={n}>{n} rows / tick</option>)}</select></label><span className="muted">{state?.replay_index || 0} / {state?.replay_rows || 0} rows</span>{state?.has_export && <a className="button-link" href={API + "/export/predictions"}>Export pilot predictions</a>}</div><div className="seek-controls"><input aria-label="Replay position" type="range" min="0" max={Math.max(0, (state?.replay_rows || 1) - 1)} value={seek} onChange={e => setSeek(Number(e.target.value))}/><button disabled={disabled || !state?.replay_rows || !state.model?.compatible} onClick={() => act("/replay", { seek })}>Go to row {seek + 1}</button></div></div>
      <div className="section-heading"><h2>All variable graphs <span className="muted">/ {unit?.tags.length || 0}</span></h2><span>Same time window · independent value scales</span></div>
      <div className="legend"><span className="actual">━ Actual</span><span className="past-prediction">┄ Previously predicted</span><span className="expected">┄ Future TFT median</span><span>Shaded: 10–90% forecast quantiles</span></div>
      <div className="tag-grid">{unit?.tags.map((tag, index) => { const s = result?.signals.find(v => v.tag === tag); const status = offline || state?.error ? "Connection unavailable" : names[s?.status || "warming_up"] || s?.status; return <article id={`tag-${unit.id}-${index}`} className={`panel tag-card ${s?.status === "alert" || s?.status === "deviation" ? "tag-deviation" : ""}`} key={tag}><div className="tag-heading"><h3 title={tag}>{tag}</h3><span className={`pill ${s?.status === "alert" || s?.status === "deviation" || s?.status === "data_quality" ? "amber" : ""}`}>{status}</span></div><div className="tag-values"><div><small>ACTUAL</small><strong>{fmt(s?.actual)}</strong></div><div><small>PREDICTED FOR NOW</small><strong className="mint-text">{fmt(s?.expected)}</strong></div><div><small>DEVIATION</small><strong className={s?.ratio != null && s.ratio >= 1 ? "amber-text" : "muted"}>{s?.ratio == null ? "—" : `${fmt(s.ratio)}×`}</strong></div></div><Trace history={history} unitId={unit.id} tag={tag} future={s?.forecast || []}/><div className="tag-foot"><span>{s?.forecast?.length || 0} future predictions</span><span>Residual {fmt(s?.residual)} · band ±{fmt(s?.tolerance)}</span></div></article>; })}</div>
      {!state?.model && <div className="panel empty"><h2>Train on your unit’s historical data.</h2><p>All numeric columns become predicted variables. Load the synthetic demo or import your CSV / Excel file to begin.</p><button onClick={() => setTab("Data & training")}>Import historical data →</button></div>}
      <div className="two-columns"><div className="panel"><div className="panel-heading"><h2>Variable notifications</h2><span className="muted">Deviation, persistence and recovery</span></div><div className="notifications">{(state?.events || []).filter(e => e.unit === unit?.name).slice(-30).reverse().map(e => <div className="notification" key={e.id}><span className={`pill ${e.status === "normal" ? "mint" : "amber"}`}>{e.status === "normal" ? "Recovered" : names[e.status]}</span><strong>{e.tag}</strong><small>{new Date(e.timestamp).toLocaleString()} · {fmt(e.ratio)}× band</small></div>)}{!state?.events.some(e => e.unit === unit?.name) && <p className="muted">No variable notifications yet. A deviation on one tag does not stop the other predictions.</p>}</div></div><div className="panel"><h2>Model diagnostics</h2><p className="muted">The TFT forecasts all variables. Isolation Forest and the LSTM autoencoder provide additional unit-level context.</p><div className="diagnostic"><span>Isolation Forest</span><strong>{fmt(result?.components.isolation_forest)}×</strong></div><div className="diagnostic"><span>LSTM reconstruction</span><strong>{fmt(result?.components.lstm_autoencoder)}×</strong></div><p className="fine-print">Scores are relative to healthy calibration data, not failure probabilities. Per-variable notifications use previously issued forecast residuals.</p></div></div>
    </>}

    {tab === "Data & training" && <>
      <div className="two-columns"><div className="panel"><div className="step-label">01 / IMPORT A UNIT</div><h2>Predict every numeric column</h2><p className="muted">One row per timestamp, any number of tags with any names. All numeric columns are included automatically; there is no input/target selection.</p><label>Unit name<input value={unitName} onChange={e => setUnitName(e.target.value)}/></label><label>Timestamp column<input value={timestampColumn} onChange={e => setTimestampColumn(e.target.value)}/></label><label>Timezone for timestamps without an offset<input value={sourceTimezone} onChange={e => setSourceTimezone(e.target.value)}/></label><label>Excel sheet name (blank = first sheet)<input value={sheet} onChange={e => setSheet(e.target.value)}/></label><label className="upload">Choose CSV or Excel workbook<input type="file" accept=".csv,.xlsx" disabled={disabled} onChange={e => { loadFile(e.target.files?.[0]).catch(error => setNotice(String(error))); e.target.value = ""; }}/></label><p className="mono muted">{fileName || "No file selected"}</p><button disabled={disabled || !fileData} onClick={async () => { const r = await act<Inspection>("/csv", { ...payload, purpose: "inspect" }); if (r) setInspection(r); }}>Inspect columns</button>{inspection && <><h3>{inspection.rows.toLocaleString()} rows · {inspection.numeric_columns.length} predicted variables</h3><div className="tag-list">{inspection.numeric_columns.map(t => <span className="mono" key={t}>{t}</span>)}</div><p className="muted">Non-numeric columns excluded: {inspection.columns.filter(t => t !== timestampColumn && !inspection.numeric_columns.includes(t)).join(", ") || "none"}</p><div className="actions"><button className="primary" disabled={disabled} onClick={importTraining}>Use all tags for training</button><button disabled={disabled || !state?.model} onClick={() => act("/csv", { ...payload, purpose: "replay" }, "Replay loaded. Open Monitor to view every tag.")}>Use file for replay</button></div></>}</div>
      <div><div className="panel"><div className="step-label">02 / TRAIN & EVALUATE</div><h2>AirSense model stack, hydro variables</h2><p className="muted">Temporal Fusion Transformer forecasts every tag. LSTM and Isolation Forest learn normal behaviour. Chronological 60 / 20 / 20 fit, calibration and test blocks.</p><p><strong>{state?.training_rows.toLocaleString() || 0}</strong> rows loaded · <strong>{config?.assets.reduce((n, a) => n + a.tags.length, 0) || 0}</strong> forecast tags</p><p className="muted">{config?.lookback} history samples → {config?.horizon} future samples · {config?.epochs} epochs</p><label className="checkbox"><input type="checkbox" checked={healthy} onChange={e => setHealthy(e.target.checked)}/> This history represents healthy operation, with known faults and maintenance removed.</label><button className="primary" disabled={disabled || !healthy || !state?.training_rows} onClick={() => act("/train", { healthy_confirmed: true })}>Train all-tag models locally</button><p className="fine-print">Forecasting uses past data only. Compare the pilot with SmartSignal on the same unseen period. Training settings are in Units & tags.</p></div><div className="panel"><div className="step-label">OPTIONAL / GPU TRAINING</div><h2>Train in Jupyter or Google Colab</h2><p className="muted">Run the same training code on a GPU, review held-out metrics, then download and activate the model in this app.</p><a className="button-link primary-link" href={COLAB} target="_blank" rel="noreferrer">Open training notebook in Colab ↗</a><p className="fine-print">Colab uploads data to Google. Use it only for data your organization permits there; the notebook also runs locally.</p><p className="muted">Stop the app, then activate your own exported model:</p><code className="command">python -m hydrosense.cli activate --model hydro-model.joblib --trust-model</code><p className="muted">Restart the app after activation. Load only a model you trained or trust.</p></div></div></div>
      {state?.model && <div className="panel"><div className="panel-heading"><h2>Held-out forecast performance</h2><span className="muted">{new Date(state.model.created_at).toLocaleString()}</span></div><p className="muted">MAE covers the full forecast horizon, in each tag’s original units. Last-value MAE is the simple persistence baseline. Lower is better; neither metric measures maintenance fault-detection accuracy.</p>{state.model.reports.map(r => <div className="report" key={r.id}><h3>{r.name}</h3><p className="muted">{r.fit_rows} fit / {r.calibration_rows} calibration / {r.test_rows} test rows · {r.test_windows} forecast windows · {(r.test_flag_rate * 100).toFixed(1)}% one-step windows flagged</p><div className="table-wrap"><table><thead><tr><th>Variable</th><th>TFT MAE</th><th>Last-value MAE</th><th>Calibrated band ±</th></tr></thead><tbody>{r.tags.map(t => <tr key={t.tag}><td className="mono">{t.tag}</td><td>{fmt(t.mae)}</td><td>{fmt(t.baseline_mae)}</td><td>{fmt(t.tolerance)}</td></tr>)}</tbody></table></div></div>)}</div>}
    </>}

    {tab === "Units & tags" && <div className="two-columns"><div className="panel"><h2>Units and forecast settings</h2><p className="muted">Every tag listed in a unit is both observed and forecast from history. Changes require retraining.</p><label>Configuration JSON<textarea className="config-editor" spellCheck={false} value={configText} onChange={e => setConfigText(e.target.value)}/></label><button className="primary" disabled={disabled} onClick={async () => { try { const c = JSON.parse(configText); const r = await act("/config", c, "Configuration saved. Reload your training file without replacing the configuration."); if (r) { setTimestampColumn(c.timestamp_column); setSourceTimezone(c.source_timezone || "UTC"); } } catch { setNotice("Configuration must be valid JSON."); } }}>Save configuration</button>{fileData && <button disabled={disabled} onClick={() => act("/csv", { ...payload, purpose: "training" }, "Training file reloaded with the existing unit configuration.")}>Reload file with this configuration</button>}</div><div className="panel help"><h2>One unit. Every variable.</h2><h3>assets → tags</h3><p>Keep all related variables together to learn their temporal relationships. For multiple units, add one entry per unit with its tag names.</p><h3>lookback & horizon</h3><p>Number of past samples used and future samples predicted. At one-minute sampling, 60 and 12 mean one hour of history and a 12-minute forecast.</p><h3>epochs & device</h3><p>Training duration and compute device: cpu, cuda or auto. Colab GPU training uses the same code and model format.</p><h3>persistence</h3><p>Each variable has its own counter. A first out-of-band observation raises a deviation notification; consecutive deviations escalate that variable.</p><h3>Comparable pilot results</h3><p>Export timestamped actuals, previously issued predictions, residuals and notifications. Align them with SmartSignal output and engineer-reviewed events.</p></div></div>}
      {tab === "PI connection" && <div className="two-columns"><div className="panel"><h2>PI Web API settings</h2><p className="muted">Enter these locally when available. The connector reads snapshots and never writes back to PI.</p><label>PI Web API base URL<input placeholder="https://your-host/piwebapi" value={connection.url || ""} onChange={e => setConnection({ ...connection, url: e.target.value })}/></label><label>PI Data Archive server<input placeholder="Your PI server name" value={connection.data_server || ""} onChange={e => setConnection({ ...connection, data_server: e.target.value })}/></label><label>Authentication<select value={connection.auth || "basic"} onChange={e => setConnection({ ...connection, auth: e.target.value })}><option value="basic">Basic over HTTPS</option><option value="windows">Windows integrated (Windows service host)</option><option value="bearer">Bearer token (compatible gateway)</option><option value="anonymous">Anonymous (if enabled on server)</option></select></label>{connection.auth === "basic" && <label>Username<input autoComplete="off" value={connection.username || ""} onChange={e => setConnection({ ...connection, username: e.target.value })}/></label>}{(connection.auth === "basic" || connection.auth === "bearer") && <label>{connection.auth === "bearer" ? "Access token" : "Password"}<input type="password" autoComplete="new-password" value={secret} placeholder={state?.connection.has_secret ? "Saved in service memory for this session" : "Enter when connecting"} onChange={e => setSecret(e.target.value)}/></label>}<label>Custom CA certificate bundle <span className="muted">(optional local path)</span><input value={connection.ca_bundle || ""} onChange={e => setConnection({ ...connection, ca_bundle: e.target.value })}/></label><label>Tag → Web ID overrides <span className="muted">(optional JSON)</span><textarea spellCheck={false} value={webIds} onChange={e => setWebIds(e.target.value)} rows={4}/></label><button className="primary" disabled={disabled} onClick={async () => { try { const r = await act("/connection", { ...connection, web_ids: JSON.parse(webIds), secret }, "Connection settings saved locally. Secret retained only in service memory."); if (r) setSecret(""); } catch { setNotice("Web ID overrides must be a JSON object."); } }}>Save connection</button></div><div><div className="panel"><div className="step-label">CONNECTION & MONITORING</div><h2>Bring the baseline online.</h2><p className="muted">Train on your plant data first, test tag access, then start monitoring. Every tag is forecast. CSV headers are resolved as PI point names unless you provide Web IDs.</p><div className="stack-actions"><button disabled={disabled || !state?.connection.url} onClick={() => act("/connection/test")}>Test connection & tags</button><button className="primary" disabled={disabled || !state?.model?.compatible || state.source === "synthetic" || !state.connection.url} onClick={() => act("/live/start", {}, "PI polling started.")}>Start live prediction</button><button disabled={state?.mode !== "live" || busy} onClick={() => act("/live/stop", {}, "Live monitoring stopped.")}>Stop live prediction</button></div><p className="fine-print">Passwords and tokens are never saved to disk. Re-enter them after restarting the Python service. TLS certificate validation stays enabled.</p></div><div className="panel help"><h2>Data quality comes first.</h2><p>Bad, questionable, substituted, missing, stale or misaligned PI values are marked as data-quality issues. Repeated snapshots do not advance alert persistence.</p><p>The model forecasts all tags for the configured horizon. Actual readings are checked against predictions issued earlier. Allow one history window to warm up after starting live monitoring.</p></div></div></div>}    <footer><span>HYDROSENSE <span className="slash">/</span> Predictive maintenance pilot</span><span>All-tag forecasts · Per-variable investigation · Local data</span></footer>
  </section></main>;
}
