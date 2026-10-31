import React, { useEffect, useRef, useState } from 'react';
import { searchConcepts } from '../api.js';

// Plain vector search — no NER, no LLM. Type a term, get top-k SNOMED candidates
// straight from the BioLORD + FAISS index. Multilingual, so Dutch works directly.
export default function VectorSearch({ embedder = 'biolord' }) {
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(10);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const timer = useRef(null);
  const seq = useRef(0); // guard against out-of-order responses

  useEffect(() => {
    const q = query.trim();
    if (timer.current) clearTimeout(timer.current);
    if (!q) {
      setResults([]);
      setLoading(false);
      setError('');
      return;
    }
    setLoading(true);
    timer.current = setTimeout(async () => {
      const mine = ++seq.current;
      try {
        const res = await searchConcepts(q, topK, embedder);
        if (mine === seq.current) {
          setResults(res.candidates || []);
          setError('');
        }
      } catch (e) {
        if (mine === seq.current) setError(e.message);
      } finally {
        if (mine === seq.current) setLoading(false);
      }
    }, 250);
    return () => timer.current && clearTimeout(timer.current);
  }, [query, topK, embedder]);

  return (
    <section className="vsearch">
      <div className="vsearch-bar">
        <input
          className="vsearch-input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search SNOMED CT directly (e.g. lumbalgie, atrial fibrillation)…"
          autoFocus
        />
        <label className="vsearch-topk">
          top‑k
          <input
            type="number"
            min={1}
            max={50}
            value={topK}
            onChange={(e) => setTopK(Math.max(1, Math.min(50, Number(e.target.value) || 1)))}
          />
        </label>
        <span className="muted small">{loading ? 'searching…' : query.trim() ? `${results.length} results` : 'no NER · pure vector'}</span>
      </div>

      {error && <div className="error">{error}</div>}

      <ul className="candidate-list vsearch-list">
        {results.map((c) => (
          <li key={c.concept_id}>
            <div className="candidate vsearch-cand">
              <span className="code-cell">
                <span className="code-fsn">{c.fsn}</span>
                <span className="code-id">{c.concept_id}</span>
              </span>
              <span className="cand-meta">
                <span className="cand-tag">{c.semantic_tag}</span>
                <span className="cand-score">{c.vector_score.toFixed(3)}</span>
              </span>
            </div>
          </li>
        ))}
        {!loading && query.trim() && results.length === 0 && !error && (
          <li className="muted small">No matches.</li>
        )}
      </ul>
    </section>
  );
}
