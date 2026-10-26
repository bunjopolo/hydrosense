"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type Range = [number, number];
type Preview = { revision: string; rows: number; start: number; end: number; start_time: string; end_time: string; reduced: boolean; traces: { tag: string; points: [number, number | null][] }[] };

function merge(ranges: Range[]): Range[] {
  const result: Range[] = [];
  for (const [start, end] of [...ranges].sort((a, b) => a[0] - b[0])) {
    const last = result[result.length - 1];
    if (last && start <= last[1]) last[1] = Math.max(end, last[1]);
    else result.push([start, end]);
  }
  return result;
}
function subtract(ranges: Range[], [start, end]: Range): Range[] {
  return ranges.flatMap(([a, b]): Range[] => {
    if (end <= a || start >= b) return [[a, b]];
    const remaining: Range[] = [];
    if (a < start) remaining.push([a, start]);
    if (end < b) remaining.push([end, b]);
    return remaining;
  });
}
const number = (value: number) => value.toLocaleString(undefined, { maximumFractionDigits: 3 });
const utc = (value: string) => new Date(value).toISOString().replace("T", " ").replace(".000Z", " UTC");

export default function TrainingReview({ api, revision, rows, initialRanges, disabled, onDirty, onApplied }: {
  api: string; revision: string; rows: number; initialRanges: Range[]; disabled: boolean;
  onDirty: (dirty: boolean) => void; onApplied: () => Promise<unknown>;
}) {
  const [ranges, setRanges] = useState<Range[]>(initialRanges);
  const [saved, setSaved] = useState<Range[]>(initialRanges);
  const [candidate, setCandidate] = useState<Range>([0, rows]);
  const [view, setView] = useState<Range>([0, rows]);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const anchor = useRef<number | null>(null);
  const dirty = JSON.stringify(ranges) !== JSON.stringify(saved);
  const count = ranges.reduce((sum, [start, end]) => sum + end - start, 0);
  const locked = disabled || loading || saving;
  const traces = useMemo(() => preview?.traces.map(trace => {
    const finite = trace.points.flatMap(([, value]) => value === null ? [] : [value]);
    const min = finite.length ? Math.min(...finite) : 0, max = finite.length ? Math.max(...finite) : 1;
    const span = Math.max(max - min, Math.abs(max) * .001, .001);
    const x = (row: number) => 50 + (row - preview.start) / (preview.end - preview.start) * 530;
    const y = (value: number) => 148 - (value - min) / span * 120;
    const line = trace.points.map(([row, value], i) => value === null ? "" : `${i === 0 || trace.points[i - 1][1] === null ? "M" : "L"}${x(row)},${y(value)}`).join(" ");
    return { tag: trace.tag, min, span, line, hasValues: finite.length > 0 };
  }) || [], [preview]);
  useEffect(() => { onDirty(dirty); }, [dirty, onDirty]);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setError("");
    fetch(api + "/training/preview", { method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal,
      body: JSON.stringify({ revision, start: view[0], end: view[1] }) })
      .then(async response => { const body = await response.json() as Preview & { error?: string }; if (!response.ok) throw new Error(body.error); return body as Preview; })
      .then(setPreview).catch(reason => { if (!controller.signal.aborted) setError(String(reason)); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [api, revision, view]);
  async function apply() {
    setSaving(true); setError("");
    try {
      const response = await fetch(api + "/training/selection", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ revision, ranges }) });
      const body = await response.json() as { ranges: Range[]; error?: string };
      if (!response.ok) throw new Error(body.error);
      await onApplied();
      setRanges(body.ranges); setSaved(body.ranges);
    } catch (reason) { setError(String(reason)); }
    finally { setSaving(false); }
  }
  const choose = (next: Range[]) => setRanges(merge(next));
  function rowAt(event: React.PointerEvent<SVGSVGElement>) {
    const box = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - box.left) / box.width * 600;
    return Math.min(view[1] - 1, Math.max(view[0], Math.floor(view[0] + (x - 50) / 530 * (view[1] - view[0]))));
  }
  function brush(event: React.PointerEvent<SVGSVGElement>) {
    if (anchor.current === null) return;
    const row = rowAt(event);
    setCandidate([Math.min(anchor.current, row), Math.max(anchor.current, row) + 1]);
  }
  return <section className="training-review" aria-label="Training data selection">
    <div className="panel review-controls">
      <div className="panel-heading"><div><div className="step-label">REVIEW / CHOOSE HEALTHY PERIODS</div><h2>Select training data visually</h2></div><strong className="mint-text" role="status">{count.toLocaleString()} / {rows.toLocaleString()} rows included</strong></div>
      <p className="muted">Drag across any graph to mark a range, then include or exclude it. The same selection applies to every tag. Green shading shows included rows; amber outlines show the range you are editing.</p>
      <div className="review-actions">
        <button disabled={locked} onClick={() => choose([[0, rows]])}>Include all rows</button>
        <button disabled={locked} onClick={() => choose([])}>Clear inclusion</button>
        <button disabled={locked || !dirty} onClick={() => setRanges(saved)}>Undo unsaved changes</button>
        <button className="primary" disabled={locked || !dirty} onClick={apply}>{saving ? "Applying…" : "Apply training selection"}</button>
        {!dirty && count > 0 && !locked && <a className="button-link" href={api + "/export/training?revision=" + encodeURIComponent(revision)}>Export included CSV</a>}
        <span className="muted">{dirty ? "Unsaved selection — apply before training" : "Selection applied"}</span>
      </div>
      {count < 250 && <p className="amber-text">Include at least 250 rows. Each training split also needs long enough continuous periods for the model’s history and forecast windows.</p>}
      <div className="range-editor">
        <label>Range start row<input type="number" min={1} max={candidate[1]} value={candidate[0] + 1} disabled={locked} onChange={e => { const n = Number(e.target.value); if (Number.isInteger(n) && n >= 1 && n <= candidate[1]) setCandidate([n - 1, candidate[1]]); }}/></label>
        <label>Range end row<input type="number" min={candidate[0] + 1} max={rows} value={candidate[1]} disabled={locked} onChange={e => { const n = Number(e.target.value); if (Number.isInteger(n) && n > candidate[0] && n <= rows) setCandidate([candidate[0], n]); }}/></label>
        <span className="muted">{(candidate[1] - candidate[0]).toLocaleString()} rows marked</span>
        <button disabled={locked} onClick={() => choose([candidate])}>Include only this range</button>
        <button disabled={locked} onClick={() => choose([...ranges, candidate])}>Add range</button>
        <button disabled={locked} onClick={() => choose(subtract(ranges, candidate))}>Exclude range</button>
        <button disabled={locked} onClick={() => setView([...candidate])}>Zoom to range</button>
        <button disabled={locked || (view[0] === 0 && view[1] === rows)} onClick={() => setView([0, rows])}>Show full file</button>
      </div>
      <div className="included-ranges" aria-label="Included training ranges">{ranges.length ? ranges.map(([start, end]) => <span key={start}>Rows {(start + 1).toLocaleString()}–{end.toLocaleString()}<button aria-label={`Remove included rows ${start + 1} to ${end}`} disabled={locked} onClick={() => choose(ranges.filter(r => r[0] !== start))}>×</button></span>) : <span>No rows included. Mark a healthy period and choose Add range.</span>}</div>
      <p className="fine-print">Selections use original row positions after sorting by timestamp. Excluded gaps stay separate during training. The chronological 60 / 20 / 20 split is applied to included rows only.</p>
      {error && <p role="alert" className="amber-text">{error}</p>}
    </div>
    <div className="section-heading"><h2>Imported training graphs <span className="muted">/ {preview?.traces.length || 0}</span></h2><span>{loading ? "Loading graphs…" : `Rows ${view[0] + 1}–${view[1]}`}</span></div>
    {preview && <><p className="muted review-time">{utc(preview.start_time)} → {utc(preview.end_time)} · Horizontal axis: original row position. {preview.reduced ? "Overview retains bucket minima and maxima; zoom in for individual readings." : "Every reading is shown."}</p>
      <div className="review-brush-bar"><span>Marked rows <strong>{candidate[0] + 1}–{candidate[1]}</strong></span><button disabled={locked} onClick={() => choose([...ranges, candidate])}>Include marked rows</button><button disabled={locked} onClick={() => choose(subtract(ranges, candidate))}>Exclude marked rows</button><button className="primary" disabled={locked || !dirty} onClick={apply}>Apply selection</button><small>{count.toLocaleString()} included{dirty ? " · unsaved" : " · applied"}</small></div>
      <div className={`tag-grid training-graphs ${loading ? "loading" : ""}`}>{traces.map(trace => {
        const { min, span, line } = trace;
        const x = (row: number) => 50 + (row - preview.start) / (preview.end - preview.start) * 530;
        const highlight = ([a, b]: Range, key: string, selected: boolean) => {
          const start = Math.max(a, preview.start), end = Math.min(b, preview.end);
          return start < end && <rect key={key} x={x(start)} y="20" width={Math.max(1, x(end) - x(start))} height="136" fill={selected ? "#63d8c4" : "#efc17c"} fillOpacity={selected ? .13 : .06} stroke={selected ? "none" : "#efc17c"} strokeWidth="2"/>;
        };
        return <article className="panel tag-card training-tag" key={trace.tag}><h3 title={trace.tag}>{trace.tag}</h3>
          <svg className="training-trace" viewBox="0 0 600 185" preserveAspectRatio="none" role="img" aria-label={`Training data graph for ${trace.tag}. Drag to mark rows; use row inputs for keyboard selection.`}
            onPointerDown={event => { if (locked || event.button !== 0) return; event.currentTarget.setPointerCapture(event.pointerId); anchor.current = rowAt(event); setCandidate([anchor.current, anchor.current + 1]); }}
            onPointerMove={brush} onPointerUp={event => { brush(event); anchor.current = null; }} onPointerCancel={() => { anchor.current = null; }}>
            <rect x="50" y="20" width="530" height="136" fill="#0b1720"/>
            {ranges.map((range, i) => highlight(range, `included-${i}`, true))}
            {[0, 1, 2].map(i => <g key={i}><line x1="50" x2="580" y1={28 + 60 * i} y2={28 + 60 * i} stroke="#28404b" strokeDasharray="3 5"/><text x="44" y={32 + 60 * i} textAnchor="end">{number(min + span * (1 - i / 2))}</text></g>)}
            <path d={line} fill="none" stroke="#8abaff" strokeWidth="1.7"/>
            {highlight(candidate, "candidate", false)}
            <text x="50" y="178">Row {preview.start + 1}</text><text x="580" y="178" textAnchor="end">Row {preview.end}</text>
          </svg>
          {!trace.hasValues && <p className="muted">No numeric readings in this window.</p>}
        </article>;
      })}</div></>}
  </section>;
}
