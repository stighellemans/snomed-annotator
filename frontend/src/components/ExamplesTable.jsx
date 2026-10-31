import React, { useEffect, useState } from 'react';
import { fetchExamples, annotateText, fetchExampleResults } from '../api.js';

const CONCURRENCY = 4;

// Distinct pastel per linked span (by order within the row), so the same color
// ties a highlighted span in the text to its SNOMED term chip.
function spanColor(i) {
  const hue = (i * 57 + 15) % 360;
  return { bg: `hsl(${hue} 72% 88%)`, accent: `hsl(${hue} 50% 42%)` };
}

// FSN stacked over concept_id; tinted to match its span highlight.
function StackedCode({ fsn, conceptId, color }) {
  const style = color
    ? { background: color.bg, borderColor: color.accent, borderLeft: `3px solid ${color.accent}` }
    : undefined;
  return (
    <span className="code-cell" style={style}>
      <span className="code-fsn">{fsn}</span>
      <span className="code-id">{conceptId}</span>
    </span>
  );
}

// Render the description with each linked span highlighted in its matching color.
function HighlightedText({ text, spans }) {
  const ordered = spans
    .filter((s) => s.start >= 0 && s.end > s.start)
    .sort((a, b) => a.start - b.start || a.end - b.end);
  const nodes = [];
  let cursor = 0;
  ordered.forEach((s, k) => {
    const begin = Math.max(cursor, Math.min(s.start, text.length));
    const end = Math.max(begin, Math.min(s.end, text.length));
    if (begin > cursor) nodes.push(<span key={`t${cursor}`}>{text.slice(cursor, begin)}</span>);
    nodes.push(
      <mark
        key={`m${k}`}
        className="hl-span"
        style={{ background: s.color.bg, borderBottom: `2px solid ${s.color.accent}` }}
        title={`${s.snomed_term} (${s.concept_id})`}
      >
        {text.slice(begin, end)}
      </mark>,
    );
    cursor = end;
  });
  if (cursor < text.length) nodes.push(<span key="tend">{text.slice(cursor)}</span>);
  return <>{nodes}</>;
}

export default function ExamplesTable({ embedder = 'biolord' }) {
  const [examples, setExamples] = useState([]);
  const [useLlm, setUseLlm] = useState(true);
  const [results, setResults] = useState({}); // key `${row}|${mode}` -> annotate result
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState({ done: 0, total: 0 });
  const [error, setError] = useState('');

  useEffect(() => {
    fetchExamples().then(setExamples).catch((e) => setError(e.message));
  }, []);

  const mode = useLlm ? 'llm' : 'top1';
  const keyOf = (i) => `${i}|${mode}|${embedder}`;

  // Rehydrate from the disk cache whenever the example set or (mode, embedder)
  // changes, so previously-annotated examples appear immediately on load instead
  // of starting blank after every reload.
  useEffect(() => {
    if (!examples.length) return undefined;
    let cancelled = false;
    fetchExampleResults(useLlm, embedder)
      .then((cached) => {
        if (cancelled) return;
        setResults((prev) => {
          const next = { ...prev };
          cached.forEach((res, i) => {
            if (res) next[`${i}|${mode}|${embedder}`] = res;
          });
          return next;
        });
      })
      .catch(() => {}); // non-fatal: table still works, just starts empty
    return () => {
      cancelled = true;
    };
  }, [examples, useLlm, embedder]);
  const ranCount = examples.filter((_, i) => results[keyOf(i)]).length;

  async function runRow(i) {
    const ex = examples[i];
    try {
      const data = await annotateText(ex.description, useLlm, embedder);
      setResults((prev) => ({ ...prev, [keyOf(i)]: data }));
    } catch (e) {
      setResults((prev) => ({ ...prev, [keyOf(i)]: { error: e.message } }));
    }
  }

  async function runAll() {
    setRunning(true);
    setError('');
    const indices = examples.map((_, i) => i).filter((i) => !results[keyOf(i)]);
    const total = indices.length;
    let done = 0;
    setProgress({ done: 0, total });
    const queue = [...indices];
    async function worker() {
      while (queue.length) {
        const i = queue.shift();
        await runRow(i);
        done += 1;
        setProgress({ done, total });
      }
    }
    await Promise.all(Array.from({ length: CONCURRENCY }, worker));
    setRunning(false);
  }

  if (error) return <div className="error">{error}</div>;
  if (!examples.length) return <p className="muted">Loading examples…</p>;

  return (
    <section className="examples">
      <div className="examples-toolbar">
        <button className="primary-btn" onClick={runAll} disabled={running}>
          {running
            ? `Running… ${progress.done}/${progress.total}`
            : `Run all (${useLlm ? 'LLM arbitrator' : 'top-1 vector'})`}
        </button>
        <label className="toggle" title="Switch between the two cached result sets. NER is reused across both.">
          <input
            type="checkbox"
            checked={useLlm}
            onChange={(e) => setUseLlm(e.target.checked)}
            disabled={running}
          />
          LLM arbitrator
        </label>
        <span className="muted small">
          {examples.length} examples · {ranCount} shown for this mode
        </span>
      </div>

      <table className="examples-table">
        <thead>
          <tr>
            <th className="col-id">Source</th>
            <th className="col-text">Description</th>
            <th className="col-codes">SNOMED CT ({useLlm ? 'LLM' : 'top-1'})</th>
            <th className="col-icd">ICD (source)</th>
            <th className="col-run"></th>
          </tr>
        </thead>
        <tbody>
          {examples.map((ex, i) => {
            const res = results[keyOf(i)];
            // Use the results already in state; tie each linked span to a color.
            const linked = res && !res.error
              ? res.spans
                  .filter((s) => s.status === 'linked' && s.concept_id)
                  .map((s, k) => ({ ...s, color: spanColor(k) }))
              : [];
            return (
              <tr key={`${ex.id}-${ex.nr}-${i}`}>
                <td className="col-id muted small">{ex.id}<br />#{ex.nr}</td>
                <td className="col-text">
                  {res && !res.error
                    ? <HighlightedText text={res.text} spans={linked} />
                    : ex.description}
                </td>
                <td className="col-codes">
                  {!res && <span className="muted small">not run</span>}
                  {res && res.error && <span className="error small">{res.error}</span>}
                  {res && !res.error && linked.length === 0 && <span className="muted small">— none —</span>}
                  {res && !res.error && linked.length > 0 && (
                    <div className="code-chips">
                      {linked.map((s, k) => (
                        <StackedCode key={k} fsn={s.snomed_term} conceptId={s.concept_id} color={s.color} />
                      ))}
                    </div>
                  )}
                </td>
                <td className="col-icd muted small">{ex.icd || '—'}</td>
                <td className="col-run">
                  <button className="ghost-btn small" onClick={() => runRow(i)} disabled={running}>
                    {res ? '↻' : 'Run'}
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}
