(function () {
  const form = document.getElementById('search-form');
  const resultsSection = document.getElementById('results');

  if (!form || !resultsSection) {
    return;
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const formData = new FormData(form);
    const payload = Object.fromEntries(formData.entries());

    try {
      const response = await fetch('/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error('Search failed');
      }

      const data = await response.json();
      renderResults(data.results || {});
    } catch (error) {
      console.error(error);
      resultsSection.innerHTML = '<article class="card"><p class="muted">Search failed. Check console for details.</p></article>';
    }
  });

  function renderResults(results) {
    const variants = Object.keys(results);
    if (!variants.length) {
      resultsSection.dataset.hasResults = 'false';
      resultsSection.innerHTML = '<article class="card empty-state"><p>No results yet. Try another query.</p></article>';
      return;
    }

    resultsSection.dataset.hasResults = 'true';
    resultsSection.innerHTML = variants
      .map((variant) => renderVariant(variant, results[variant]))
      .join('');
  }

  function renderVariant(variant, hits) {
    if (!Array.isArray(hits)) {
      hits = [];
    }
    const items = hits
      .map((hit, index) => {
        const best = hit.best_utterance || {};
        const speakers = Array.isArray(best.speakers) ? best.speakers : [];
        const speakerHtml = speakers
          .map((sp) => `<li>${sp.label || sp.id}</li>`)
          .join('');
        const locationHtml = hit.location && hit.location.latitude != null && hit.location.longitude != null
          ? `<p class="muted">Location: ${hit.location.latitude.toFixed(4)}, ${hit.location.longitude.toFixed(4)}</p>`
          : '';
        const audioLink = hit.media_path
          ? `<a class="link" href="/media?path=${encodeURIComponent(hit.media_path)}" target="_blank" rel="noopener">Audio</a>`
          : '';
        return `
          <li class="result-item">
            <div class="result-meta">
              <span class="badge">${index + 1}</span>
              <span class="mono">${hit.key || hit.id || ''}</span>
              <span class="muted">score ${(hit.score || 0).toFixed(4)}</span>
            </div>
            ${best.text ? `<p class="utterance-text">${best.text}</p>` : ''}
            ${best.start != null && best.end != null ? `<p class="muted">${best.start.toFixed(2)}s – ${best.end.toFixed(2)}s</p>` : ''}
            ${speakerHtml ? `<ul class="speaker-tags">${speakerHtml}</ul>` : ''}
            <div class="result-actions">
              <a class="link" href="/transcripts/${encodeURIComponent(hit.id)}">Open transcript</a>
              ${audioLink}
            </div>
            ${locationHtml}
          </li>
        `;
      })
      .join('');
    return `
      <article class="card">
        <header class="result-header">
          <h3>${variant.toUpperCase()}</h3>
          <p class="muted">${hits.length} hits</p>
        </header>
        <ol class="result-list">${items}</ol>
      </article>
    `;
  }
})();
