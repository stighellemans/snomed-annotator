import React, { useEffect, useState } from 'react';
import { fetchEmbedders } from '../api.js';

// Global embedder picker. Lists backends from /api/embedders; disables any whose
// index isn't available on disk.
export default function EmbedderSelect({ value, onChange, disabled }) {
  const [options, setOptions] = useState([]);

  useEffect(() => {
    fetchEmbedders()
      .then((d) => setOptions(d.embedders || []))
      .catch(() => setOptions([]));
  }, []);

  return (
    <label className="embedder-select" title="Embedding backend used for vector retrieval">
      <span className="muted small">embedder</span>
      <select value={value} onChange={(e) => onChange(e.target.value)} disabled={disabled}>
        {options.map((o) => (
          <option key={o.id} value={o.id} disabled={!o.available}>
            {o.label}
            {o.available ? '' : ' (unavailable)'}
          </option>
        ))}
      </select>
    </label>
  );
}
