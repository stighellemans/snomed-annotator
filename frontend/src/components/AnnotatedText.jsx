import React from 'react';

// Deterministic pastel color per category (ported from custom-annotation-platform).
function hashString(value) {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash * 31 + value.charCodeAt(i)) | 0;
  }
  return Math.abs(hash);
}

export function categoryColor(label) {
  if (!label) return '#fff4d6';
  const hue = (hashString(label) * 47) % 360;
  return `hsl(${hue} 74% 88%)`;
}

// Drop the trailing SNOMED semantic tag (e.g. " (disorder)") for the
// at-a-glance label; the full term stays in the tooltip + inspector.
function shortTerm(term) {
  if (!term) return '';
  return term.replace(/\s*\([^)]*\)\s*$/, '');
}

export default function AnnotatedText({ text, spans, selectedIndex, onSelect }) {
  if (!text) return null;

  // Spans carry their original array index; render in document order.
  const ordered = spans
    .map((span, index) => ({ ...span, index }))
    .filter((span) => span.start >= 0 && span.end > span.start)
    .sort((a, b) => a.start - b.start || a.end - b.end);

  const nodes = [];
  let cursor = 0;

  for (const span of ordered) {
    const begin = Math.max(cursor, Math.min(span.start, text.length));
    const end = Math.max(begin, Math.min(span.end, text.length));
    if (begin > cursor) {
      nodes.push(<span key={`t-${cursor}`}>{text.slice(cursor, begin)}</span>);
    }
    const linked = span.status === 'linked' && span.concept_id;
    nodes.push(
      <span
        key={`s-${span.index}-${begin}`}
        className={`inline-span ${selectedIndex === span.index ? 'selected' : ''} ${linked ? 'linked' : 'unlinked'}`}
        style={{ '--span-color': categoryColor(span.category) }}
        role="button"
        tabIndex={0}
        title={linked ? `${span.snomed_term} (${span.concept_id})` : 'No SNOMED code chosen'}
        onClick={() => onSelect(span.index)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onSelect(span.index);
          }
        }}
      >
        <span className="inline-span-label" aria-hidden="true">
          {linked ? (
            <>
              <span className="lbl-fsn">{shortTerm(span.snomed_term)}</span>
              <span className="lbl-id">{span.concept_id}</span>
            </>
          ) : (
            <span className="lbl-fsn">{span.category || 'span'}</span>
          )}
        </span>
        <span className="inline-span-text">{text.slice(begin, end)}</span>
      </span>,
    );
    cursor = end;
  }
  if (cursor < text.length) {
    nodes.push(<span key={`t-end`}>{text.slice(cursor)}</span>);
  }
  return <div className="text-content">{nodes}</div>;
}
