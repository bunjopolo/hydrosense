"use client";

import { useEffect, useRef, useState } from "react";

type Asset = { id: string; name: string; kind: string; inputs: string[]; targets: string[]; running?: { tag: string; min: number } };
type Config = { source_timezone?: string; timestamp_column: string; poll_seconds: number; stale_seconds: number; max_skew_seconds: number; persistence: number; assets: Asset[] };
type Signal = { tag: string; actual: number; expected: number; residual: number; tolerance: number; ratio: number };
type Result = { id: string; name: string; kind: string; status: string; detail: string; score: number | null; signals: Signal[]; consecutive: number; timestamp: string | null };
type Report = { id: string; name: string; rows: number; excluded_rows: number; fit_rows: number; calibration_rows: number; test_rows: number; test_flag_rate: number; test_out_of_domain_rate: number; tags: { tag: string; mae: number; tolerance: number }[] };
type Connection = { url?: string; data_server?: string; auth?: string; username?: string; ca_bundle?: string; web_ids?: Record<string, string>; has_secret?: boolean };
type State = { config: Config; connection: Connection; job: { status: string; error?: string }; mode: string; source: string; error: string | null; results: Result[]; history: { timestamp: string; assets: Result[] }[]; updated_at: string | null; training_rows: number; replay_rows: number; replay_index: number; model: { created_at: string; compatible: boolean; reports: Report[] } | null };
type Inspection = { columns: string[]; rows: number; missing: string[]; preview: Record<string, string>[] };
const API = "http://127.0.0.1:8765";
const names: Record<string, string> = { normal: "Within baseline", deviation: "Deviation", alert: "Persistent deviation", data_quality: "Data quality", out_of_domain: "Unseen operation", not_running: "Not running" };
const fmt = (n: number | null | undefined) => n == null ? "—" : n.toLocaleString(undefined, { maximumFractionDigits: 3 });

async function request<T = { ok?: boolean; message?: string }>(path: string, data?: unknown) {
  const response = await fetch(API + path, data === undefined ? undefined : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
  const body = await response.json() as T & { error?: string };
  if (!response.ok) throw new Error(body.error || "Request failed");
  return body;
}

function Trace({ history, assetId, tag }: { history: State["history"]; assetId: string; tag: string }) {
  const points = history.map(h => h.assets.find(a => a.id === assetId)?.signals.find(s => s.tag === tag));
  const values = points.flatMap(p => p ? [p.actual, p.expected - p.tolerance, p.expected + p.tolerance] : []);
  if (values.length < 4) return <div className="chart-empty">Start replay or live monitoring to plot actual and expected values.</div>;
  const min = Math.min(...values), max = Math.max(...values), span = Math.max(max - min, .001);
  const y = (v: number) => 185 - (v - min) / span * 155;
  const x = (i: number) => 60 + i / Math.max(points.length - 1, 1) * 820;
  const path = (field: "actual" | "expected") => points.map((p, i) => p ? `${i === 0 || !points[i - 1] ? "M" : "L"}${x(i)},${y(p[field])}` : "").join(" ");
  return <svg className="trace" viewBox="0 0 920 230" role="img" aria-label={`Actual and expected values for ${tag}`}>
    {[0, 1, 2, 3].map(i => <g key={i}><line x1="60" x2="880" y1={30 + i * 155 / 3} y2={30 + i * 155 / 3} stroke="#243844" strokeDasharray="3 5"/><text x="48" y={34 + i * 155 / 3} textAnchor="end">{fmt(max - i * span / 3)}</text></g>)}
    {points.map((p, i) => p && <line key={i} x1={x(i)} x2={x(i)} y1={y(p.expected - p.tolerance)} y2={y(p.expected + p.tolerance)} stroke="#48c8bf" strokeOpacity=".1" strokeWidth={Math.max(1, 820 / points.length)}/>)}
    <path d={path("expected")} fill="none" stroke="#54d7c5" strokeWidth="2" strokeDasharray="6 4"/>
    <path d={path("actual")} fill="none" stroke="#89b7ff" strokeWidth="2.5"/>
    <text x="60" y="216">{new Date(history[0].timestamp).toLocaleTimeString()}</text><text x="880" y="216" textAnchor="end">{new Date(history[history.length - 1].timestamp).toLocaleTimeString()}</text>
  </svg>;
}

export default function Home() {
  const [state, setState] = useState<State | null>(null);
  const [tab, setTab] = useState("Monitor");
  const [offline, setOffline] = useState(false);
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [configText, setConfigText] = useState("");
  const [connection, setConnection] = useState<Connection>({ auth: "basic" });
  const [secret, setSecret] = useState("");
  const [webIds, setWebIds] = useState("{}");
  const [fileData, setFileData] = useState<Record<string, string> | null>(null);
  const [fileName, setFileName] = useState("");
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [healthy, setHealthy] = useState(false);
  const [selected, setSelected] = useState("");
  const [tag, setTag] = useState("");
  const [assetName, setAssetName] = useState("My asset");
  const [inputTags, setInputTags] = useState<string[]>([]);
  const [targetTags, setTargetTags] = useState<string[]>([]);
  const [timestampColumn, setTimestampColumn] = useState("timestamp");
  const [sheet, setSheet] = useState("");
  const [sourceTimezone, setSourceTimezone] = useState("UTC");
  const initialized = useRef(false);
  const ticking = useRef(false);

  async function refresh() {
    const next = await request<State>("/status");
    setState(next); setOffline(false);
    if (!initialized.current) {
      setConfigText(JSON.stringify(next.config, null, 2)); setTimestampColumn(next.config.timestamp_column);
      setSourceTimezone(next.config.source_timezone || "UTC"); setConnection({ auth: "basic", ...next.connection }); setWebIds(JSON.stringify(next.connection.web_ids || {}, null, 2));
      initialized.current = true;
    }
    return next;
  }
  useEffect(() => {
    let active = true;
    const poll = () => { if (active) refresh().catch(() => { if (active) setOffline(true); }); };
    poll(); const timer = setInterval(poll, 2000);
    return () => { active = false; clearInterval(timer); };
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
      try { await request("/replay", { steps: 2 }); const next = await refresh(); if (next.replay_index >= next.replay_rows) setPlaying(false); }
      catch (error) { setPlaying(false); setNotice(String(error)); }
      finally { ticking.current = false; }
    }, 800);
    return () => clearInterval(timer);
  }, [playing]);

  const config = state?.config;
  const asset = config?.assets.find(a => a.id === selected) || config?.assets[0];
  const result = state?.results.find(a => a.id === asset?.id);
  const activeTag = asset?.targets.includes(tag) ? tag : asset?.targets[0] || "";
  const disabled = busy || offline || state?.job.status === "training" || state?.mode === "live" || playing;
  const updateTag = (value: string, checked: boolean, role: "input" | "target") => {
    const setter = role === "input" ? setInputTags : setTargetTags;
    setter(current => checked ? [...current, value] : current.filter(t => t !== value));
    if (checked) (role === "input" ? setTargetTags : setInputTags)(current => current.filter(t => t !== value));
  };
  async function loadFile(file?: File) {
    if (!file) return;
    if (file.size > 45 * 1024 * 1024) { setNotice("Use files below 45 MB in the browser, or use the Python CLI for larger files."); return; }
    setInspection(null); setHealthy(false); setFileName(file.name);
    let payload: Record<string, string>;
    if (/\.xlsx$/i.test(file.name)) {
      const buffer = new Uint8Array(await file.arrayBuffer());
      let binary = ""; for (let i = 0; i < buffer.length; i += 8192) binary += String.fromCharCode(...buffer.subarray(i, i + 8192));
      payload = { xlsx: btoa(binary) };
    } else payload = { csv: await file.text() };
    setFileData(payload);
    const info = await act<Inspection>("/csv", { ...payload, purpose: "inspect", timestamp_column: timestampColumn, source_timezone: sourceTimezone, sheet });
    if (info) { setInspection(info); setInputTags([]); setTargetTags([]); }
  }
  async function addAsset(replace: boolean) {
    if (!config || !inputTags.length || !targetTags.length) { setNotice("Select at least one operating input and one predicted tag."); return; }
    const added: Asset = { id: "asset-" + Date.now(), name: assetName || "My asset", kind: "Asset", inputs: inputTags, targets: targetTags };
    const next = { ...config, timestamp_column: timestampColumn, source_timezone: sourceTimezone, assets: replace ? [added] : [...config.assets, added] };
    const saved = await act("/config", next, "Asset mapping saved. Import this file for training next.");
    if (saved) { setConfigText(JSON.stringify(next, null, 2)); setSelected(added.id); }
  }

  return <main className="shell">
    <aside className="sidebar"><div className="brand"><span className="brand-icon">≈</span><div>HydroSense<small>ASSET INTELLIGENCE</small></div></div>
      <span className="nav-label">WORKSPACE</span><nav>{["Monitor", "Data & training", "Assets & tags", "PI connection"].map((item, i) => <button key={item} className={tab === item ? "nav-item active" : "nav-item"} onClick={() => setTab(item)}><span>{["◉", "▤", "⌘", "⇄"][i]}</span>{item}</button>)}</nav>
      <div className="sidebar-note"><span className="dot"/> LOCAL WORKSPACE<p>Your files and PI credentials stay on this machine.</p><small>Normal-behaviour modelling<br/>Generators · Bearings · Transformers</small></div>
    </aside>
    <section className="workspace"><header><span>Operations <span className="slash">/</span> {tab}</span><span className={`pill ${offline ? "amber" : "mint"}`}>{offline ? "Service offline" : state ? "Local service connected" : "Connecting…"}</span></header>
      <div className="page-title"><div><div className="eyebrow">HYDRO CONDITION MONITORING</div><h1>{tab === "Monitor" ? "Know what normal looks like." : tab}</h1><p>{tab === "Monitor" ? "Learn your assets. Follow expected behaviour. Investigate the departures." : tab === "Data & training" ? "Your columns, your tags, your models. Train from healthy operating history." : tab === "Assets & tags" ? "Define each asset’s inputs, predicted tags and operating conditions." : "Connect to your PI Web API when you’re ready. No server details are prefilled."}</p></div>
      {tab === "Monitor" && <button className="primary" disabled={disabled} onClick={async () => { const r = await act("/demo", {}, "Training on synthetic hydro data…"); if (r) { const s = await refresh(); setConfigText(JSON.stringify(s.config, null, 2)); setSelected(""); } }}>Load synthetic demo</button>}</div>
      {offline && <div className="banner warning">The local Python service is unavailable. Start both services with <code>./scripts/run_dashboard.sh</code>. Monitoring is not updating.</div>}
      {notice && <div role="status" className="banner">{notice}<button onClick={() => setNotice("")} aria-label="Dismiss message">×</button></div>}
      {state?.error && <div role="alert" className="banner warning">PI monitoring interrupted: {state.error}</div>}
      {state?.job.status === "training" && <div className="banner">Training models and evaluating a chronological holdout…</div>}
      {state?.job.status === "failed" && <div role="alert" className="banner warning">Training failed: {state.job.error}</div>}
      {state?.model && !state.model.compatible && <div className="banner warning">Asset configuration changed. Retrain before replay or live monitoring.</div>}

      {tab === "Monitor" && <>
        <div className="stats"><div><span>Configured assets</span><strong>{config?.assets.length ?? "—"}</strong><small>Independent models per asset</small></div><div><span>Predicted tags</span><strong>{config?.assets.reduce((n, a) => n + a.targets.length, 0) ?? "—"}</strong><small>Learned from your operating data</small></div><div><span>Monitoring source</span><strong className="text-stat">{state?.mode === "live" ? "PI Web API" : state?.mode === "replay" ? "CSV / Excel replay" : "Not running"}</strong><small>{state?.source === "synthetic" ? "Synthetic model · demonstration only" : state?.model ? "Model trained from imported data" : "No trained model yet"}</small></div><div><span>Persistent deviations</span><strong className="amber-text">{offline || state?.error || !state?.results.length ? "—" : state.results.filter(a => a.status === "alert").length}</strong><small>{config?.persistence ?? 3} consecutive fresh observations</small></div></div>
        <div className="section-heading"><h2>Asset overview</h2><span>{state?.updated_at ? `Last scored ${new Date(state.updated_at).toLocaleString()}` : "Awaiting observations"}</span></div>
        <div className="asset-grid">{config?.assets.map(a => { const r = state?.results.find(s => s.id === a.id); const status = offline ? "Service offline" : state?.error ? "Connection error" : names[r?.status || ""] || "Awaiting data"; return <button className={`asset-card ${asset?.id === a.id ? "selected" : ""}`} key={a.id} onClick={() => { setSelected(a.id); setTag(""); }}><div><span className="asset-icon">{a.kind === "Transformer" ? "ϟ" : "◎"}</span><span className={`pill ${r?.status === "alert" || r?.status === "data_quality" ? "amber" : ""}`}>{status}</span></div><h3>{a.name}</h3><p>{a.kind} <span>·</span> {a.targets.length} predicted tags</p><div className="asset-bottom"><span>Largest deviation</span><strong>{offline || state?.error || r?.score == null ? "—" : `${fmt(r.score)}×`}</strong></div></button>; })}</div>
        {!state?.model && <div className="empty panel"><h2>Start with the data you already have.</h2><p>Import a healthy CSV or Excel export, map the tag columns to assets, and train a baseline. Or load the synthetic demo to explore the full workflow.</p><button onClick={() => setTab("Data & training")}>Import operating history →</button></div>}
        <div className="panel trend-panel"><div className="panel-heading"><div><div className="eyebrow">ACTUAL VS EXPECTED</div><h2>{asset?.name || "Asset behaviour"}</h2></div><select aria-label="Chart tag" value={activeTag} onChange={e => setTag(e.target.value)}>{asset?.targets.map(t => <option key={t}>{t}</option>)}</select></div><div className="legend"><span className="actual">━ Actual</span><span className="expected">┄ Expected</span><span>Shaded: calibrated normal residual band</span></div><Trace history={state?.history || []} assetId={asset?.id || ""} tag={activeTag}/><div className="replay-controls"><span className="pill">{state?.mode === "live" ? "LIVE PI" : state?.source === "synthetic" ? "SYNTHETIC REPLAY" : "FILE REPLAY"}</span><button disabled={!state?.replay_rows || offline || busy || state?.job.status === "training" || state?.mode === "live" || !state?.model?.compatible} onClick={() => setPlaying(!playing)}>{playing ? "Pause" : "Play replay"}</button><button disabled={disabled || !state?.replay_rows || !state?.model?.compatible} onClick={() => act("/replay")}>Step</button><button disabled={disabled || !state?.replay_rows || !state?.model?.compatible} onClick={() => act("/replay", { reset: true })}>Restart</button><span className="muted">{state?.replay_index || 0} / {state?.replay_rows || 0} rows</span></div></div>
        <div className="panel"><div className="panel-heading"><h2>Signal detail</h2><span className="muted">Deviation ≥ 1× exceeds the calibrated band</span></div>{result && <p className="muted">{result.detail}</p>}<div className="table-wrap"><table><thead><tr><th>PI tag</th><th>Actual</th><th>Expected</th><th>Residual</th><th>Band ±</th><th>Deviation</th></tr></thead><tbody>{result?.signals.map(s => <tr key={s.tag}><td className="mono">{s.tag}</td><td>{fmt(s.actual)}</td><td className="mint-text">{fmt(s.expected)}</td><td>{fmt(s.residual)}</td><td>{fmt(s.tolerance)}</td><td className={s.ratio >= 1 ? "amber-text" : ""}>{fmt(s.ratio)}×</td></tr>)}</tbody></table>{!result?.signals.length && <p className="table-empty">No scored signals yet.</p>}</div></div>
      </>}

      {tab === "Data & training" && <>
        <div className="two-columns"><div className="panel"><div className="step-label">01 / IMPORT</div><h2>Operating history</h2><p className="muted">One row per timestamp. Any number of numeric tag columns, with their original names. Use timezone-bearing timestamps, or choose the timezone for Excel/local timestamps below.</p><label>Timestamp column<input value={timestampColumn} onChange={e => setTimestampColumn(e.target.value)}/></label><label>Source timezone for timestamps without an offset<input value={sourceTimezone} onChange={e => setSourceTimezone(e.target.value)} placeholder="America/Vancouver"/></label><label>Excel sheet name <span className="muted">(blank = first sheet)</span><input value={sheet} onChange={e => setSheet(e.target.value)}/></label><label className="upload">Choose CSV or Excel workbook<input type="file" accept=".csv,.xlsx" disabled={disabled} onChange={e => { loadFile(e.target.files?.[0]).catch(error => setNotice(String(error))); e.target.value = ""; }}/></label><p className="mono muted">{fileName || "No file selected"}</p><button disabled={disabled || !fileData} onClick={async () => { const r = await act<Inspection>("/csv", { ...fileData, purpose: "inspect", timestamp_column: timestampColumn, source_timezone: sourceTimezone, sheet }); if (r) setInspection(r); }}>Inspect columns</button></div>
        <div className="panel"><div className="step-label">03 / TRAIN</div><h2>Learn expected behaviour</h2><p className="muted">60% fit · 20% calibration · 20% held-out evaluation. Each asset learns its own relationships from the columns you select.</p><p><strong>{state?.training_rows.toLocaleString() || 0}</strong> training rows loaded</p><label className="checkbox"><input type="checkbox" checked={healthy} onChange={e => setHealthy(e.target.checked)}/> This history represents known healthy operation, with faults and maintenance removed.</label><button className="primary" disabled={disabled || !healthy || !state?.training_rows} onClick={() => act("/train", { healthy_confirmed: true }, "Training started.")}>Train asset models</button><p className="fine-print">New readings update predictions. Retraining is explicit, so developing faults do not automatically become the new normal.</p></div></div>
        {inspection && <div className="panel"><div className="step-label">02 / MAP YOUR TAGS</div><h2>{inspection.rows.toLocaleString()} rows · {inspection.columns.length - 1} available columns</h2><p className="muted">Select operating inputs such as load or cooling, then the tags you want to predict. Names are kept exactly as supplied. Inputs and targets cannot overlap.</p><label>Asset name<input value={assetName} onChange={e => setAssetName(e.target.value)}/></label><div className="mapping-list">{inspection.columns.filter(t => t !== timestampColumn).map(t => <div key={t}><span className="mono">{t}</span><label className="checkbox"><input type="checkbox" checked={inputTags.includes(t)} onChange={e => updateTag(t, e.target.checked, "input")}/>Input</label><label className="checkbox"><input type="checkbox" checked={targetTags.includes(t)} onChange={e => updateTag(t, e.target.checked, "target")}/>Predict</label></div>)}</div><div className="actions"><button disabled={disabled} onClick={() => addAsset(true)}>Replace assets with this mapping</button><button disabled={disabled} onClick={() => addAsset(false)}>Add asset</button></div><div className="actions"><button className="primary" disabled={disabled || !fileData} onClick={() => act("/csv", { ...fileData, sheet, source_timezone: sourceTimezone, purpose: "training" }, "Training file loaded.")}>Use file for training</button><button disabled={disabled || !fileData} onClick={() => act("/csv", { ...fileData, sheet, source_timezone: sourceTimezone, purpose: "replay" }, "Replay file loaded. Open Monitor to play.")}>Use file for replay</button></div></div>}
        {state?.model && <div className="panel"><div className="panel-heading"><h2>Held-out model results</h2><span className="muted">Trained {new Date(state.model.created_at).toLocaleString()}</span></div><p className="muted">Flag rate measures deviations in the held-out data; it is not failure-detection accuracy. Errors and bands use each tag’s original units.</p>{state.model.reports.map(r => <div className="report" key={r.id}><h3>{r.name}</h3><p className="muted">{r.fit_rows} fit / {r.calibration_rows} calibration / {r.test_rows} test · {r.excluded_rows} excluded · {(r.test_flag_rate * 100).toFixed(1)}% flagged · {(r.test_out_of_domain_rate * 100).toFixed(1)}% outside learned operating range</p><div className="table-wrap"><table><thead><tr><th>Predicted tag</th><th>Mean absolute error</th><th>Normal residual band ±</th></tr></thead><tbody>{r.tags.map(t => <tr key={t.tag}><td className="mono">{t.tag}</td><td>{fmt(t.mae)}</td><td>{fmt(t.tolerance)}</td></tr>)}</tbody></table></div></div>)}</div>}
      </>}

      {tab === "Assets & tags" && <div className="two-columns"><div className="panel"><h2>Asset configuration</h2><p className="muted">Use the column picker in Data & training, or edit the configuration here. Tag names can contain dots, spaces, slashes or other characters.</p><label>Configuration JSON<textarea className="config-editor" spellCheck={false} value={configText} onChange={e => setConfigText(e.target.value)}/></label><button className="primary" disabled={disabled} onClick={async () => { try { const next = JSON.parse(configText); const r = await act("/config", next, "Configuration saved. Re-import data and retrain."); if (r) setTimestampColumn(next.timestamp_column); } catch { setNotice("Configuration must be valid JSON."); } }}>Save configuration</button></div><div className="panel help"><h2>A model for each asset</h2><h3>inputs</h3><p>Operating conditions used to calculate the expected values. Choose signals such as load, speed, cooling temperature or ambient temperature.</p><h3>targets</h3><p>Any numeric tags you want to monitor. Their expected values change with the incoming operating inputs.</p><h3>running (optional)</h3><p>A tag and minimum value that gate training and monitoring to running conditions. Use separate assets/models for distinct regimes.</p><h3>Timing and alerts</h3><p>Configure polling, maximum data age, permitted timestamp skew, and how many fresh deviations must persist before an alert.</p><h3>Units and scaling</h3><p>Values stay in the units of the source file. PI must use those same units. The model learns the values; tag names carry no built-in meaning.</p></div></div>}

      {tab === "PI connection" && <div className="two-columns"><div className="panel"><h2>PI Web API settings</h2><p className="muted">Enter these locally when available. The connector reads snapshots and never writes back to PI.</p><label>PI Web API base URL<input placeholder="https://your-host/piwebapi" value={connection.url || ""} onChange={e => setConnection({ ...connection, url: e.target.value })}/></label><label>PI Data Archive server<input placeholder="Your PI server name" value={connection.data_server || ""} onChange={e => setConnection({ ...connection, data_server: e.target.value })}/></label><label>Authentication<select value={connection.auth || "basic"} onChange={e => setConnection({ ...connection, auth: e.target.value })}><option value="basic">Basic over HTTPS</option><option value="windows">Windows integrated (Windows service host)</option><option value="bearer">Bearer token (compatible gateway)</option><option value="anonymous">Anonymous (if enabled on server)</option></select></label>{connection.auth === "basic" && <label>Username<input autoComplete="off" value={connection.username || ""} onChange={e => setConnection({ ...connection, username: e.target.value })}/></label>}{(connection.auth === "basic" || connection.auth === "bearer") && <label>{connection.auth === "bearer" ? "Access token" : "Password"}<input type="password" autoComplete="new-password" value={secret} placeholder={state?.connection.has_secret ? "Saved in service memory for this session" : "Enter when connecting"} onChange={e => setSecret(e.target.value)}/></label>}<label>Custom CA certificate bundle <span className="muted">(optional local path)</span><input value={connection.ca_bundle || ""} onChange={e => setConnection({ ...connection, ca_bundle: e.target.value })}/></label><label>Tag → Web ID overrides <span className="muted">(optional JSON)</span><textarea spellCheck={false} value={webIds} onChange={e => setWebIds(e.target.value)} rows={4}/></label><button className="primary" disabled={disabled} onClick={async () => { try { const r = await act("/connection", { ...connection, web_ids: JSON.parse(webIds), secret }, "Connection settings saved locally. Secret retained only in service memory."); if (r) setSecret(""); } catch { setNotice("Web ID overrides must be a JSON object."); } }}>Save connection</button></div><div><div className="panel"><div className="step-label">CONNECTION & MONITORING</div><h2>Bring the baseline online.</h2><p className="muted">Train on your plant data first, test tag access, then start monitoring. CSV headers are resolved as PI point names unless you provide Web IDs.</p><div className="stack-actions"><button disabled={disabled || !state?.connection.url} onClick={() => act("/connection/test")}>Test connection & tags</button><button className="primary" disabled={disabled || !state?.model?.compatible || state.source === "synthetic" || !state.connection.url} onClick={() => act("/live/start", {}, "PI polling started.")}>Start live prediction</button><button disabled={state?.mode !== "live" || busy} onClick={() => act("/live/stop", {}, "Live monitoring stopped.")}>Stop live prediction</button></div><p className="fine-print">Passwords and tokens are never saved to disk. Re-enter them after restarting the Python service. TLS certificate validation stays enabled.</p></div><div className="panel help"><h2>Data quality comes first.</h2><p>Bad, questionable, substituted, missing, stale or misaligned PI values are marked as data-quality issues. Repeated snapshots do not advance alert persistence.</p><p>Predictions describe expected values now, given current operating inputs. They are not a forecast of the next hour or a diagnosis of remaining asset life.</p></div></div></div>}
      <footer><span>HYDROSENSE <span className="slash">/</span> Local asset intelligence</span><span>Advisory monitoring · Validate on plant history before operational use</span></footer>
    </section>
  </main>;
}
