import React, { useEffect, useState } from 'react';
import { fetchPrompts } from '../api.js';

// One labelled, copyable block of prompt text.
function PromptBlock({ label, text }) {
  const [copied, setCopied] = useState(false);
  function copy() {
    navigator.clipboard?.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    });
  }
  return (
    <div className="prompt-block">
      <div className="prompt-block-head">
        <span className="prompt-block-label">{label}</span>
        <button className="ghost-btn small" onClick={copy} type="button">
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre className="prompt-pre">{text}</pre>
    </div>
  );
}

export default function PromptsView() {
  const [prompts, setPrompts] = useState([]);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchPrompts().then(setPrompts).catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="error">{error}</div>;
  if (!prompts.length) return <p className="muted">Loading prompts…</p>;

  return (
    <section className="prompts-view">
      <p className="muted small">
        The exact prompts sent to the LLM at each stage of the pipeline. Per-request content
        (the note, the span, the retrieved candidates) is shown as ‹placeholders›; configured
        values are rendered as their current settings.
      </p>
      {prompts.map((p) => (
        <article className="prompt-card" key={p.id}>
          <header className="prompt-card-head">
            <h2>{p.title}</h2>
            <div className="prompt-tags">
              <span className="prompt-tag">{p.model}</span>
              <span className="prompt-tag">structured: {p.structured_output}</span>
              <span className="prompt-tag">prompt v{p.version}</span>
            </div>
          </header>
          <p className="muted small prompt-purpose">{p.purpose}</p>
          <PromptBlock label="System" text={p.system} />
          <PromptBlock label="User" text={p.user} />
          <PromptBlock label="Output schema (JSON)" text={JSON.stringify(p.schema, null, 2)} />
        </article>
      ))}
    </section>
  );
}
