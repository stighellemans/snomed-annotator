import React, { useState } from 'react';
import { annotateText } from './api.js';
import AnnotatedText from './components/AnnotatedText.jsx';
import SpanInspector from './components/SpanInspector.jsx';
import ExamplesTable from './components/ExamplesTable.jsx';
import VectorSearch from './components/VectorSearch.jsx';
import PromptsView from './components/PromptsView.jsx';
import EmbedderSelect from './components/EmbedderSelect.jsx';

// Authored fictional example; no patient data is shipped with the prototype.
const SAMPLE = `Patiënt met hypertensie en diabetes mellitus type 2. Sinds drie dagen lumbalgie. Geen koorts.`;

export default function App() {
  const [text, setText] = useState(SAMPLE);
  const [result, setResult] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [useLlm, setUseLlm] = useState(true);
  const [embedder, setEmbedder] = useState('biolord');
  const [view, setView] = useState('annotate'); // 'annotate' | 'search' | 'examples' | 'pipeline'

  async function onAnnotate() {
    setLoading(true);
    setError('');
    setResult(null);
    setSelectedIndex(null);
    try {
      const data = await annotateText(text, useLlm, embedder);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function updateSpan(index, patch) {
    setResult((prev) => {
      if (!prev) return prev;
      const spans = prev.spans.map((s, i) => (i === index ? { ...s, ...patch } : s));
      return { ...prev, spans };
    });
  }

  const selectedSpan =
    result && selectedIndex != null
      ? { ...result.spans[selectedIndex], index: selectedIndex }
      : null;

  const linkedCount = result ? result.spans.filter((s) => s.status === 'linked' && s.concept_id).length : 0;

  return (
    <div className="app">
      <header className="app-header">
        <h1>ZAS · SNOMED CT Annotator</h1>
        <span className="muted small">Multilingual vector search + LLM selection</span>
        <nav className="tabs">
          <button className={view === 'annotate' ? 'tab active' : 'tab'} onClick={() => setView('annotate')}>
            Annotate
          </button>
          <button className={view === 'search' ? 'tab active' : 'tab'} onClick={() => setView('search')}>
            Search
          </button>
          <button className={view === 'examples' ? 'tab active' : 'tab'} onClick={() => setView('examples')}>
            Examples
          </button>
          <button className={view === 'pipeline' ? 'tab active' : 'tab'} onClick={() => setView('pipeline')}>
            Pipeline
          </button>
        </nav>
        {view !== 'pipeline' && (
          <EmbedderSelect value={embedder} onChange={setEmbedder} disabled={loading} />
        )}
      </header>

      {view === 'pipeline' && (
        <section className="pipeline-view">
          <img src="/pipeline.svg" alt="SNOMED CT annotation pipeline data flow" />
          <PromptsView />
        </section>
      )}

      {view === 'search' && <VectorSearch embedder={embedder} />}

      {view === 'examples' && <ExamplesTable embedder={embedder} />}

      {view === 'annotate' && (
      <>
      <section className="input-panel">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste a Dutch clinical snippet…"
          rows={5}
        />
        <div className="actions">
          <button className="primary-btn" onClick={onAnnotate} disabled={loading || !text.trim()}>
            {loading ? 'Annotating…' : 'Annotate'}
          </button>
          <label className="toggle" title="When off, each span takes the top-1 vector match with no LLM arbitration — faster and deterministic.">
            <input
              type="checkbox"
              checked={useLlm}
              onChange={(e) => setUseLlm(e.target.checked)}
              disabled={loading}
            />
            LLM arbitrator
          </label>
          {result && (
            <span className="muted small">
              {result.spans.length} spans · {linkedCount} linked
            </span>
          )}
        </div>
        {error && <div className="error">{error}</div>}
      </section>

      {result && (
        <section className="workspace">
          <AnnotatedText
            text={result.text}
            spans={result.spans}
            selectedIndex={selectedIndex}
            onSelect={setSelectedIndex}
          />
          <SpanInspector span={selectedSpan} onUpdate={updateSpan} embedder={embedder} />
        </section>
      )}
      </>
      )}
    </div>
  );
}
