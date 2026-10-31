import React, { useEffect, useState } from 'react';
import { searchConcepts } from '../api.js';

export default function SpanInspector({ span, onUpdate, embedder = 'biolord' }) {
  const [query, setQuery] = useState('');
  const [extra, setExtra] = useState([]);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    setQuery(span ? span.search_term || span.text : '');
    setExtra([]);
    setError('');
  }, [span, embedder]);

  if (!span) {
    return (
      <aside className="inspector">
        <p className="muted">Click a highlighted span to inspect and correct its SNOMED CT code.</p>
      </aside>
    );
  }

  // Merge pipeline candidates with any manual-search extras, dedup by concept_id.
  const byId = new Map();
  for (const c of [...(span.candidates || []), ...extra]) {
    if (!byId.has(c.concept_id)) byId.set(c.concept_id, c);
  }
  const candidates = [...byId.values()].sort((a, b) => b.vector_score - a.vector_score);

  async function runSearch(e) {
    e.preventDefault();
    if (!query.trim()) return;
    setSearching(true);
    setError('');
    try {
      const res = await searchConcepts(query.trim(), 10, embedder);
      setExtra(res.candidates || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setSearching(false);
    }
  }

  function choose(c) {
    onUpdate(span.index, {
      concept_id: c.concept_id,
      snomed_term: c.fsn,
      status: 'linked',
    });
  }

  return (
    <aside className="inspector">
      <div className="inspector-head">
        <div className="span-chip">{span.text}</div>
        <div className="muted small">{span.category}</div>
      </div>

      <div className="current">
        {span.status === 'linked' && span.concept_id ? (
          <>
            <div className="current-label">Selected</div>
            <div className="current-term">{span.snomed_term || '(unknown)'}</div>
            <div className="muted small">
              {span.concept_id}
              {typeof span.confidence === 'number' ? ` · conf ${span.confidence.toFixed(2)}` : ''}
            </div>
          </>
        ) : (
          <div className="muted">No SNOMED code chosen</div>
        )}
        <button className="ghost-btn" onClick={() => onUpdate(span.index, { concept_id: '', snomed_term: '', status: 'none' })}>
          Clear code
        </button>
      </div>

      <form className="search-row" onSubmit={runSearch}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search SNOMED CT…"
        />
        <button type="submit" disabled={searching}>{searching ? '…' : 'Search'}</button>
      </form>

      {error && <div className="error">{error}</div>}

      <ul className="candidate-list">
        {candidates.map((c) => {
          const active = c.concept_id === span.concept_id && span.status === 'linked';
          return (
            <li key={c.concept_id}>
              <button className={`candidate ${active ? 'active' : ''}`} onClick={() => choose(c)}>
                <div className="cand-term">{c.fsn}</div>
                <div className="cand-meta">
                  <span className="cand-id">{c.concept_id}</span>
                  <span className="cand-tag">{c.semantic_tag}</span>
                  <span className="cand-score">{c.vector_score.toFixed(3)}</span>
                </div>
              </button>
            </li>
          );
        })}
        {candidates.length === 0 && <li className="muted small">No candidates.</li>}
      </ul>
    </aside>
  );
}
