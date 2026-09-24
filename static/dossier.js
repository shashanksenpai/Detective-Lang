/* Shared dossier rendering for person.html (one case) and combined_dossier.html (pooled across cases).
 * Both pages show the same profile object, so the rendering lives here once - and so does the escaping:
 * person names, source labels, case names and words all originate in uploaded chat exports, which are
 * untrusted, so nothing that comes from a file reaches innerHTML except through esc().
 * The page must provide the element ids used below (sourceChips, sentimentBar, pctPos, pctNeu, pctNeg,
 * avgCompound, volatility, traits, traitDisclaimer, topics, topWords, distinctiveWords, phrases,
 * comparisonSection, compareGrid, insights). */
(function () {
  'use strict';

  function esc(value) {
    return String(value ?? '').replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function pct(n) { return (n === undefined || n === null) ? '—' : n.toFixed(1) + '%'; }
  function signed(n) { return (n >= 0 ? '+' : '') + n.toFixed(2); }
  function frac(n) { return Math.max(0, Math.min(1, Number(n) / 100 || 0)); }
  function $(id) { return document.getElementById(id); }

  function renderSources(bySource) {
    $('sourceChips').innerHTML = Object.entries(bySource).map(function (e) {
      var label = e[0], s = e[1];
      return '<span class="source-chip">' + esc(label) + ' <span class="ctx-tag">' + esc(s.context) + '</span> · ' + esc(s.message_count) + ' msgs</span>';
    }).join('');
  }

  function renderSentiment(stats) {
    var s = stats.sentiment;
    $('sentimentBar').innerHTML =
      '<div class="neg" style="width:' + (Number(s.pct_negative) || 0) + '%"></div>' +
      '<div class="neu" style="width:' + (Number(s.pct_neutral) || 0) + '%"></div>' +
      '<div class="pos" style="width:' + (Number(s.pct_positive) || 0) + '%"></div>';
    $('pctPos').textContent = pct(s.pct_positive);
    $('pctNeu').textContent = pct(s.pct_neutral);
    $('pctNeg').textContent = pct(s.pct_negative);
    $('avgCompound').textContent = signed(s.avg_compound);
    $('volatility').textContent = s.volatility.toFixed(2);
  }

  function renderTraits(traits) {
    $('traits').innerHTML = Object.entries(traits.scores).map(function (e) {
      var name = e[0], score = e[1];
      return '<div class="trait-row"><div class="trait-head"><span class="trait-name">' + esc(name) + '</span>' +
        '<span class="trait-score">' + esc(score) + '/100</span></div>' +
        '<span class="meter wide" style="--v:' + frac(score) + '"></span>' +
        '<div class="trait-basis">Based on: ' + esc(traits.basis[name]) + '</div></div>';
    }).join('');
    $('traitDisclaimer').textContent = traits.disclaimer;
  }

  function renderTopics(dist) {
    var entries = Object.entries(dist).sort(function (a, b) { return b[1] - a[1]; });
    $('topics').innerHTML = entries.map(function (e) {
      return '<div class="topic-row"><div class="label">' + esc(e[0]) + '</div>' +
        '<span class="meter wide" style="--v:' + frac(e[1]) + '"></span>' +
        '<div class="pct">' + esc(e[1].toFixed(0)) + '%</div></div>';
    }).join('');
  }

  function renderVocab(vocab) {
    var none = '<span class="empty-note">not enough data yet</span>';
    $('topWords').innerHTML = vocab.top_words.length
      ? vocab.top_words.map(function (w) { return '<span class="word-chip">' + esc(w.word) + '<span class="n">×' + esc(w.count) + '</span></span>'; }).join('')
      : none;
    $('distinctiveWords').innerHTML = vocab.distinctive_words.length
      ? vocab.distinctive_words.map(function (w) { return '<span class="word-chip">' + esc(w.word) + '<span class="n">' + esc(w.ratio) + '×</span></span>'; }).join('')
      : none;
    $('phrases').innerHTML = vocab.frequent_phrases.length
      ? vocab.frequent_phrases.map(function (p) { return '<span class="word-chip">"' + esc(p.phrase) + '"<span class="n">×' + esc(p.count) + '</span></span>'; }).join('')
      : none;
  }

  function statRows(stats) {
    function row(label, value) { return '<div class="stat-row"><span class="label">' + label + '</span><span class="value">' + esc(value) + '</span></div>'; }
    return row('Messages', stats.message_count) + row('Avg tone', signed(stats.sentiment.avg_compound)) +
      row('Emoji / msg', stats.avg_emoji_per_message) + row('Self-reference', stats.self_reference_ratio + '%') +
      row('Volatility', stats.sentiment.volatility);
  }

  // The group-vs-DM panel only exists for someone who was seen in both contexts.
  function renderComparison(profile) {
    var section = $('comparisonSection'), ctx = profile.by_context;
    if (!ctx.group || !ctx.dm) { section.style.display = 'none'; return; }
    section.style.display = 'block';
    $('compareGrid').innerHTML =
      '<div class="compare-col"><h4>Group</h4>' + statRows(ctx.group) + '</div>' +
      '<div class="compare-col"><h4>DM</h4>' + statRows(ctx.dm) + '</div>';
    $('insights').innerHTML = (profile.context_comparison || []).map(function (i) { return '<div class="insight">' + esc(i) + '</div>'; }).join('');
  }

  function renderProfile(profile) {
    renderSources(profile.by_source);
    renderSentiment(profile.overall);
    renderTraits(profile.traits);
    renderTopics(profile.overall.topic_distribution);
    renderVocab(profile.vocabulary);
    renderComparison(profile);
  }

  window.Dossier = { esc: esc, pct: pct, signed: signed, frac: frac, renderProfile: renderProfile };
})();
