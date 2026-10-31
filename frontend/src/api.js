export async function fetchEmbedders() {
  const res = await fetch('/api/embedders');
  if (!res.ok) throw new Error(`Embedders failed (${res.status})`);
  return res.json(); // { default, embedders: [{id,label,available}] }
}

export async function annotateText(text, useLlmSelection = true, embedder = 'biolord') {
  const res = await fetch('/api/annotate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, use_llm_selection: useLlmSelection, embedder }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `Annotate failed (${res.status})`);
  }
  return res.json();
}

export async function fetchExamples() {
  const res = await fetch('/api/examples');
  if (!res.ok) throw new Error(`Examples failed (${res.status})`);
  const data = await res.json();
  return data.examples || [];
}

// Persisted annotate results per example (from the backend disk cache), aligned
// with the examples order; entries are null where not yet computed for this mode.
export async function fetchExampleResults(useLlmSelection = true, embedder = 'biolord') {
  const res = await fetch(
    `/api/examples/results?embedder=${encodeURIComponent(embedder)}&use_llm_selection=${useLlmSelection}`,
  );
  if (!res.ok) throw new Error(`Example results failed (${res.status})`);
  const data = await res.json();
  return data.results || [];
}

// The system/user prompt templates + output schemas for the NER and arbitrator
// LLM stages (for the Prompts tab).
export async function fetchPrompts() {
  const res = await fetch('/api/prompts');
  if (!res.ok) throw new Error(`Prompts failed (${res.status})`);
  const data = await res.json();
  return data.prompts || [];
}

export async function searchConcepts(query, topK = 10, embedder = 'biolord') {
  const res = await fetch(
    `/api/search?q=${encodeURIComponent(query)}&top_k=${topK}&embedder=${encodeURIComponent(embedder)}`,
  );
  if (!res.ok) throw new Error(`Search failed (${res.status})`);
  return res.json();
}
