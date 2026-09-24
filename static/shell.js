/* Detective Lang workstation shell (BACKLOG N-3 / Phase 7).
 * Builds the top bar every page shares: brand, the case you are in, navigation, and live
 * status indicators. Every indicator is read from the running server (GET /status,
 * /cases/{id}, /merge-suggestions) - nothing here is decorative or hard-coded state.
 * Case names come from users and uploaded files, so the DOM is built with textContent only.
 * If the server cannot be reached the bar still renders and says so (amber), it never
 * shows a stale "online". */
(function () {
  'use strict';

  // The same address every page already uses for its API calls (see BACKLOG S-2).
  var API = 'http://127.0.0.1:8000';
  var POLL_MS = 20000;

  var params = new URLSearchParams(location.search);
  var caseId = params.get('case_id') || params.get('case');
  var page = (location.pathname.split('/').pop() || '').toLowerCase();

  var GLOBAL_NAV = [['cases.html', 'Cases'], ['merge_review.html', 'Identity review']];
  var CASE_NAV = [
    ['detective_lang.html', 'Investigate'], ['person.html', 'Dossiers'],
    ['investigation_board.html', 'Board'], ['workspace.html', 'Workspace'],
  ];
  var PAGE_LABEL = {
    'combined_dossier.html': 'Combined dossier',   // pages the nav does not name; the nav names the rest
  };

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function getJSON(path) {
    var ctl = new AbortController();
    var timer = setTimeout(function () { ctl.abort(); }, 4000);
    return fetch(API + path, { signal: ctl.signal }).then(function (r) {
      clearTimeout(timer);
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    });
  }

  // one status indicator: a dot, a small label and a value; updated in place
  function indicator(label) {
    var wrap = el('div', 'wb-ind');
    var dot = el('span', 'dot');
    var lab = el('span', null, label);
    var val = el('b', null, '…');
    wrap.append(dot, lab, val);
    return {
      wrap: wrap,
      set: function (dotClass, value, warn, title) {
        dot.className = 'dot ' + (dotClass || '');
        val.textContent = value;
        wrap.classList.toggle('warn', !!warn);
        if (title) wrap.title = title; else wrap.removeAttribute('title');
      },
    };
  }

  function build() {
    var bar = el('header', 'wb-bar');
    bar.setAttribute('role', 'banner');

    var brand = el('a', 'wb-brand');
    brand.href = 'cases.html';
    brand.append(el('i'), el('span', null, 'DETECTIVE·LANG'));
    bar.append(brand);

    var crumb = null;
    if (caseId || PAGE_LABEL[page]) {
      crumb = el('div', 'wb-crumb');
      if (caseId) crumb.append('CASE ', el('b', null, String(caseId).padStart(2, '0')), ' // …');
      else crumb.textContent = PAGE_LABEL[page].toUpperCase();
      bar.append(crumb);
    }

    var nav = el('nav', 'wb-nav');
    nav.setAttribute('aria-label', 'Workstation');
    var reviewBadge = null;
    function link(href, text, withCase) {
      var a = el('a', null, text);
      a.href = withCase ? href + '?case_id=' + encodeURIComponent(caseId) : href;
      if (page === href) a.setAttribute('aria-current', 'page');
      nav.append(a);
      return a;
    }
    GLOBAL_NAV.forEach(function (n) {
      var a = link(n[0], n[1], false);
      if (n[0] === 'merge_review.html') { reviewBadge = el('span', 'badge'); reviewBadge.hidden = true; a.append(reviewBadge); }
    });
    if (caseId) CASE_NAV.forEach(function (n) { link(n[0], n[1], true); });
    bar.append(nav, el('div', 'wb-spacer'));

    var api = indicator('API'), sent = indicator('NLP'), llm = indicator('LLM'), db = indicator('DB');
    bar.append(api.wrap, sent.wrap, llm.wrap, db.wrap);
    document.body.insertBefore(bar, document.body.firstChild);

    function refresh() {
      getJSON('/status').then(function (s) {
        api.set('cyan', 'online', false);
        var fb = s.sentiment.engine !== 'hinglish-model';
        sent.set(fb ? 'amber' : 'blue', fb ? 'vader fallback' : 'hinglish', fb,
          fb ? 'The trained Hinglish model is not loaded: sentiment is plain VADER (English only).' : 'Trained Hinglish sentiment model loaded.');
        var on = s.external_llm.state !== 'off';
        llm.set(on ? 'amber' : '', on ? 'configured' : 'off', on, s.external_llm.detail || '');
        db.set('blue', s.counts.messages.toLocaleString() + ' msgs', false,
          s.counts.cases + ' cases, ' + s.counts.sources_ready + ' ready sources, ' + s.counts.messages + ' messages stored');
      }).catch(function () {
        api.set('amber', 'unreachable', true, 'The server did not answer. Start it: python -m uvicorn server:app --port 8000');
        sent.set('', '–'); llm.set('', '–'); db.set('', '–');
      });
      getJSON('/merge-suggestions?status=pending').then(function (list) {
        if (reviewBadge) { reviewBadge.hidden = !list.length; reviewBadge.textContent = String(list.length); reviewBadge.title = list.length + ' identity matches awaiting review'; }
      }).catch(function () { /* the API indicator already says so */ });
    }
    refresh();
    setInterval(refresh, POLL_MS);
    document.addEventListener('visibilitychange', function () { if (!document.hidden) refresh(); });

    if (caseId && crumb) {
      getJSON('/cases/' + encodeURIComponent(caseId)).then(function (c) {
        crumb.textContent = '';
        crumb.append('CASE ', el('b', null, String(caseId).padStart(2, '0')), ' // ' + String(c.name).toUpperCase());
        crumb.title = c.name;
      }).catch(function () { /* leave the placeholder; API indicator reports the outage */ });
    }
  }

  if (document.body) build(); else document.addEventListener('DOMContentLoaded', build);
})();
