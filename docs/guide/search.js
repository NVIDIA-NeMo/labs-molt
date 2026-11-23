/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Search for the docs: a palette over every section and every flag row of the pages in pages.js.
// MiniSearch ranks the matches in the browser, so there is no server and no build step. The palette
// also hands the question to an AI assistant, which reads the repository.
// A page gets the palette by loading site.js, pages.js, and this file, and by having a [data-search] button.
(function () {
  'use strict';
  var d = document, el = window.Molt.el;
  var GUIDE = new URL('.', d.currentScript.src).href;  // this file sits in the guide folder
  d.body.insertAdjacentHTML('beforeend',
    '<dialog class="search" id="search" aria-label="Search the docs">' +
    '<label class="search-box">' +
    '<svg viewBox="0 0 16 16" aria-hidden="true"><use href="#search-icon"/></svg>' +
    '<input type="search" placeholder="Search the docs, or ask a question" autocomplete="off"' +
    ' spellcheck="false" aria-label="Search the docs" aria-controls="search-results">' +
    '<button class="search-close" type="button" aria-label="Close search"><kbd>esc</kbd></button>' +
    '</label>' +
    '<div class="search-list" id="search-results" role="listbox" aria-label="Results"></div>' +
    '<p class="search-foot"><span><kbd>↑</kbd><kbd>↓</kbd> move</span><span><kbd>↵</kbd> open</span>' +
    '<span><kbd>esc</kbd> close</span></p>' +
    '</dialog>');
  var dialog = d.getElementById('search'), input = dialog.querySelector('input');
  var list = d.getElementById('search-results');
  var MINISEARCH = 'https://cdn.jsdelivr.net/npm/minisearch@7.1.2/dist/umd/index.js';
  var MINISEARCH_SRI = 'sha384-g0EqC0ztAmrHQRoIxEGBacCSpVWXXiY+Vtg/twou+HeyvDp3B7440gAaiHnj9GJQ';
  var STOP = new RegExp('^(a|an|and|are|as|be|can|do|does|for|how|i|in|is|it|my|of|on|or|should|the|to|' +
    'what|when|where|which|why|with)$');
  var EXAMPLES = ['ep_size', 'partial rollout', 'ChatAgent', 'how do I evaluate a checkpoint'];
  var REPO = 'NVIDIA-NeMo/labs-molt';
  var ASSISTANTS = [
    ['Ask ChatGPT', 'https://chatgpt.com/?hints=search&q='],
    ['Ask Claude', 'https://claude.ai/new?q=']
  ];
  var ICONS = {
    page: 'M4 1.5h5.5L13 5v9.5H4z M9.5 1.5V5H13',
    section: 'M6.5 2 5 14M11 2 9.5 14M3 6h11M2 10h11',
    row: 'M5.5 4 2 8l3.5 4M10.5 4 14 8l-3.5 4',
    ask: 'M8 1.5 9.6 6.4 14.5 8 9.6 9.6 8 14.5 6.4 9.6 1.5 8 6.4 6.4z'
  };
  var RULE = /^\|[\s:|-]+$/;  // the line under a table's header row
  var entries = [], index, ready, status, active = 0;

  // Words, plus the parts of CamelCase names, so "runner" finds StepEnvRunner.
  function tokenize(text) {
    var words = text.match(/[\p{L}\p{N}]+/gu) || [];
    words.forEach(function (w) {
      if (/[a-z][A-Z]/.test(w)) words.push.apply(words, w.match(/[A-Z]?[a-z]+|[A-Z]+(?![a-z])/g));
    });
    return words;
  }

  // Markdown to the text a reader sees: link targets, emphasis marks, and escapes go; code stays.
  function plain(s) {
    return s.replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1').replace(/^>\s?/, '').split('`').map(function (part, i) {
      return i % 2 ? part : part.replace(/\\(.)|\*+/g, '$1');
    }).join('').trim();
  }

  // Index one page: an entry for the page, one per section, and one per table row whose first cell
  // is code, such as a flag, a field, or a metric. `at` is the text of the heading or the first cell,
  // which is where the result lands.
  function parse(md, page, group) {
    var fence = false, h2 = '', heads = [], lines = md.split('\n');
    function add(kind, title, crumb, at) {
      var entry = { id: entries.length, kind: kind, page: page[0], title: title, crumb: crumb, at: at, text: '' };
      entries.push(entry);
      return entry;
    }
    var section = add('page', page[1], group, '');
    section.text = page[2];
    lines.forEach(function (line, i) {
      if (/^```/.test(line)) { fence = !fence; return; }
      var h = !fence && line.match(/^(#{2,3}) +(.*)/);
      // A table line, split into cells. "\|" is a literal pipe inside a cell.
      var cells = !fence && line[0] === '|' && line.replace(/\\\|/g, '\0').split('|').slice(1, -1);
      if (cells) cells = cells.map(function (cell) { return plain(cell.replace(/\0/g, '|')); });
      if (h) {
        var title = plain(h[2]);
        section = add('section', title, h[1] === '##' ? page[1] : page[1] + ' › ' + h2, title);
        if (h[1] === '##') h2 = title;
      } else if (cells && RULE.test(lines[i + 1])) {
        heads = cells;  // the header row names the columns; it is not content
      } else if (cells && /^\|\s*`/.test(line)) {
        var crumb = section.kind === 'page' ? page[1] : page[1] + ' › ' + section.title;
        add('row', cells[0], crumb, cells[0]).text = cells.slice(1).map(function (cell, j) {
          return cell && heads[j + 1] === 'Default' ? 'default ' + cell : cell;
        }).filter(Boolean).join(' · ');
      } else if (cells && !RULE.test(line)) {
        section.text += ' ' + cells.filter(Boolean).join(' · ') + '.';
      } else if (!cells && line.trim() && (fence || !/^# /.test(line))) {  // "# Title" is the page entry itself
        section.text += ' ' + (fence ? line.trim() : plain(line));
      }
    });
  }

  // Fetch MiniSearch and every page once, on the first search.
  function load() {
    if (ready) return ready;
    var pages = [], lib = new Promise(function (ok, fail) {
      var js = el('script');
      js.src = MINISEARCH; js.integrity = MINISEARCH_SRI; js.crossOrigin = 'anonymous';
      js.onload = ok; js.onerror = fail;
      d.head.appendChild(js);
    });
    window.MOLT_DOCS.forEach(function (group) {
      group.pages.forEach(function (page) { pages.push([page, group.section]); });
    });
    ready = Promise.all([lib].concat(pages.map(function (p) {
      return fetch(GUIDE + 'pages/' + p[0][0] + '.md').then(function (r) {
        return r.ok ? r.text() : Promise.reject();
      });
    }))).then(function (got) {
      pages.forEach(function (p, i) { parse(got[i + 1], p[0], p[1]); });
      index = new window.MiniSearch({
        fields: ['title', 'crumb', 'text'],
        tokenize: tokenize,
        searchOptions: {
          boost: { title: 4, crumb: 1.5 },
          prefix: function (term) { return term.length > 1; },
          fuzzy: function (term) { return term.length > 4 ? 0.2 : false; },
          weights: { prefix: 0.7, fuzzy: 0.3 }
        }
      });
      index.addAll(entries);
    });
    return ready;
  }

  // The eight best entries for `q`. MiniSearch scores them, and three rules adjust the order.
  function search(q) {
    var words = tokenize(q).map(function (w) { return w.toLowerCase(); });
    var terms = words.filter(function (w) { return !STOP.test(w); });
    if (!terms.length) terms = words;
    // A query that looks like a flag or a name favours the rows; a question favours pages and sections.
    var code = /--|[_.]\w|[a-z][A-Z]|\(\)/.test(q), seen = {};
    var weight = { page: 2, section: 1.3, row: code ? 1.5 : 0.7 };
    var hits = index.search(terms.join(' '));
    hits.forEach(function (h) {
      var e = entries[h.id];
      // Matching only some of the words counts for less; a title that contains the whole query, for more.
      h.score *= weight[e.kind] * h.queryTerms.length / terms.length;
      if (e.title.toLowerCase().indexOf(q.toLowerCase()) >= 0) h.score *= 2;
    });
    hits.sort(function (a, b) { return b.score - a.score; });
    return hits.filter(function (h) {
      var key = entries[h.id].page + ' ' + entries[h.id].title;  // a page can document a flag twice
      return !seen[key] && (seen[key] = true);
    }).slice(0, 8);
  }

  // About 170 characters of `text`, starting just before its first matched word.
  function snippet(text, terms) {
    var m = new RegExp('(^|[^\\p{L}\\p{N}])(' + terms.join('|') + ')(?![\\p{L}\\p{N}])', 'iu').exec(text);
    var from = Math.max(0, (m ? m.index : 0) - 50), cut = text.slice(from, from + 170).trim();
    if (from) cut = '…' + cut.replace(/^\S*\s/, '');
    return from + 170 < text.length ? cut.replace(/\s\S*$/, '') + '…' : cut;
  }

  // Append `text` to `parent`, with the matched words marked.
  function mark(parent, text, terms) {
    text.split(/([\p{L}\p{N}]+)/u).forEach(function (part, i) {
      var matched = i % 2 && terms.indexOf(part.toLowerCase()) >= 0;
      parent.appendChild(matched ? el('mark', null, part) : d.createTextNode(part));
    });
  }

  // One row of the list: an icon, a title with a short note beside it, and for a docs entry its text.
  function hit(href, kind, title, note, text, terms) {
    var a = el('a', 'hit'), badge = el('span', 'hit-icon'), head = el('span', 'hit-head');
    a.href = href;
    a.setAttribute('role', 'option');
    if (kind === 'ask') { a.target = '_blank'; a.rel = 'noopener'; }
    badge.innerHTML = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="' + ICONS[kind] + '"/></svg>';
    mark(head.appendChild(el('b', kind === 'row' ? 'mono' : null)), title, terms || []);
    head.appendChild(el('small', null, note));
    a.appendChild(badge);
    a.appendChild(head);
    if (text) mark(a.appendChild(el('p')), text, terms);
    list.appendChild(a);
  }

  function select(i) {
    [].forEach.call(list.querySelectorAll('.hit'), function (a, j) { a.setAttribute('aria-selected', j === i); });
    active = i;
  }

  function show() {
    var q = input.value.trim(), hits = q && index ? search(q) : [];
    list.textContent = '';
    if (!index) {
      list.appendChild(el('p', 'search-note', status));
    } else if (!q) {
      var tries = list.appendChild(el('p', 'search-note', 'Search every page, section, and flag. Try '));
      EXAMPLES.forEach(function (text) { tries.appendChild(el('button', null, text)).type = 'button'; });
    } else if (!hits.length) {
      list.appendChild(el('p', 'search-note', 'Nothing in the docs matches “' + q + '”. An assistant may know.'));
    } else {
      list.appendChild(el('h4', null, 'In the docs'));
    }
    hits.forEach(function (h) {
      var e = entries[h.id];
      var href = GUIDE + '?p=' + e.page + (e.at && '&at=' + encodeURIComponent(e.at));
      hit(href, e.kind, e.title, e.crumb, snippet(e.text, h.terms), h.terms);
    });

    // An assistant cannot run this site's script, so it reads the repository, which holds these pages.
    var prompt = (q ? 'Answer this question about Molt: ' + q : 'I have questions about Molt.') +
      '\n\nMolt is a framework for agentic reinforcement learning. Read its repository first: ' +
      'https://github.com/' + REPO + ' (the documentation is in docs/guide/pages).';
    list.appendChild(el('h4', null, 'Ask an assistant'));
    ASSISTANTS.forEach(function (a) {
      hit(a[1] + encodeURIComponent(prompt), 'ask', a[0], 'opens with a link to the repository');
    });
    hit('https://deepwiki.com/' + REPO, 'ask', 'Ask DeepWiki', 'answers from the source code');
    select(0);
    list.scrollTop = 0;
  }

  function open() {
    status = 'Loading the docs…';
    dialog.showModal();
    input.select();
    show();
    load().then(show, function () {
      ready = null;  // the next open tries again
      status = 'Search could not load. Check the connection, then open it again.';
      show();
    });
  }

  input.addEventListener('input', show);
  input.addEventListener('keydown', function (e) {
    var hits = list.querySelectorAll('.hit'), step = { ArrowDown: 1, ArrowUp: -1 }[e.key];
    if (step) {
      e.preventDefault();
      select((active + step + hits.length) % hits.length);
      hits[active].scrollIntoView({ block: 'nearest' });
    }
    if (e.key === 'Enter' && !e.isComposing) hits[active].click();  // Enter also confirms an input method
  });
  list.addEventListener('mousemove', function (e) {
    var a = e.target.closest('.hit');
    if (a && a.getAttribute('aria-selected') !== 'true') select([].indexOf.call(list.querySelectorAll('.hit'), a));
  });
  list.addEventListener('click', function (e) {
    if (e.target.tagName === 'BUTTON') { input.value = e.target.textContent; input.focus(); show(); }
  });
  d.addEventListener('click', function (e) {
    if (e.target.closest('[data-search]')) open();
    else if (e.target === dialog || e.target.closest('.search-close')) dialog.close();  // the backdrop, or "esc"
  });
  d.addEventListener('keydown', function (e) {
    var typing = /^(INPUT|TEXTAREA)$/.test(d.activeElement.tagName);
    if ((e.key === 'k' && (e.metaKey || e.ctrlKey)) || (e.key === '/' && !typing)) {
      e.preventDefault();
      if (!dialog.open) open();
    }
  });
  if (!/Mac|iPhone|iPad/.test(navigator.platform)) {
    [].forEach.call(d.querySelectorAll('[data-search] kbd'), function (k) { k.textContent = 'Ctrl K'; });
  }
})();
