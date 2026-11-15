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

// Shared by the homepage, the blog, and the docs: post lists and Markdown rendering.
(function () {
  'use strict';

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  // "2026-11" -> "November 2026"; "2026-11-15" -> "November 15, 2026".
  function fmtDate(s) {
    return new Date(s).toLocaleDateString('en-US',
      { timeZone: 'UTC', year: 'numeric', month: 'long', day: s.length > 7 ? 'numeric' : undefined });
  }

  // One link per post: a compact row, or with opts.cards a card with a cover.
  // opts.base is the path from the page to the blog folder; opts.limit caps the count.
  function render(container, opts) {
    window.MOLT_POSTS.slice(0, opts.limit).forEach(function (p) {
      var a = el('a', opts.cards ? 'post' : 'row');
      a.href = p.url || opts.base + '?p=' + encodeURIComponent(p.slug);
      if (p.url) { a.target = '_blank'; a.rel = 'noopener'; }
      if (p.tone) a.setAttribute('data-tone', p.tone);

      var time = el('time', null, fmtDate(p.date));
      time.setAttribute('datetime', p.date);
      var body = el('div', opts.cards ? 'post-body' : null);
      body.appendChild(el('span', 'tag', p.tag));
      body.appendChild(el('h3', null, p.title));
      body.appendChild(el('p', null, p.summary));

      if (opts.cards) {
        var cover = el('div', 'post-cover');
        if (p.image) {
          var img = el('img');
          img.src = opts.base + p.image; img.alt = '';
          cover.appendChild(img);
        } else {
          cover.appendChild(el('span', 'cover-big', p.cover[0]));
          cover.appendChild(el('span', 'cover-small', p.cover[1]));
        }
        body.appendChild(time);
        a.appendChild(cover);
        a.appendChild(body);
      } else {
        a.appendChild(time);
        a.appendChild(body);
        a.appendChild(el('span', 'row-go', '›'));
      }
      container.appendChild(a);
    });
  }

  // Light syntax coloring for Python and shell samples; other languages stay plain.
  // Each rule's groups are, in order: comment, string, then two language-specific classes.
  var PYTHON = /(#.*)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')|\b(False|None|True|and|as|async|await|break|class|continue|def|elif|else|except|finally|for|from|if|import|in|is|lambda|not|or|pass|raise|return|try|while|with|yield)\b|\b(\d[\w.]*)\b/g;
  var SHELL = /((?:^|\s)#.*)|("[^"\n]*"|'[^'\n]*')|(^[A-Za-z_.\/][\w.\/-]*)|((?:^|\s)--?[A-Za-z][\w.\-]*)/gm;
  var RULES = {
    python: [PYTHON, 't-com', 't-str', 't-key', 't-num'],
    bash: [SHELL, 't-com', 't-str', 't-cmd', 't-flag']
  };
  function highlight(code, lang) {
    var rule = RULES[lang];
    if (!rule) return;
    code.innerHTML = code.innerHTML.replace(rule[0], function (match, com, str, third) {
      var cls = rule[com != null ? 1 : str != null ? 2 : third != null ? 3 : 4];
      return '<span class="' + cls + '">' + match + '</span>';
    });
  }

  var KATEX = 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/';
  var KATEX_JS = 'sha384-7zkQWkzuo3B5mTepMUcHkMB5jZaolc2xDwL6VFqjFALcbeS9Ggm/Yr2r3Dy4lfFg';
  var KATEX_CSS = 'sha384-nB0miv6/jRmo5UMMR1wu3Gz6NLsoTkbqJghGIsx//Rlm+ZU03BU6SQNC66uf4l5+';

  // Render the Markdown file at `url` into `prose`. The page must load marked first.
  function markdown(md, url, prose) {
    // Lift math out before Markdown runs, so underscores and backslashes survive:
    // $inline$, $$display$$, and GitHub-style ```math fences. Code is left alone.
    var math = [];
    function keep(tex, display) {
      math.push({ tex: tex, display: display });
      var token = '@@MATH' + (math.length - 1) + '@@';
      return display ? '\n\n' + token + '\n\n' : token;
    }
    md = md.split(/(^```[^\n]*\n[\s\S]*?\n```[ \t]*$)/m).map(function (seg, i) {
      if (i % 2) {
        var fence = seg.match(/^```math[ \t]*\n([\s\S]*?)\n```[ \t]*$/);
        return fence ? keep(fence[1].trim(), true) : seg;
      }
      return seg.split(/(`[^`\n]*`)/).map(function (text, j) {
        if (j % 2) return text;
        return text
          .replace(/\$\$([\s\S]+?)\$\$/g, function (_, tex) { return keep(tex.trim(), true); })
          .replace(/(^|[^\\$\w])\$(?!\s)([^$\n]*?[^\s$\\])\$(?![\w$])/g, function (_, pre, tex) { return pre + keep(tex, false); });
      }).join('');
    }).join('');
    prose.innerHTML = window.marked.parse(md)
      .replace(/<p>\s*@@MATH(\d+)@@\s*<\/p>/g, '<div class="math-block" data-math="$1"></div>')
      .replace(/@@MATH(\d+)@@/g, '<span data-math="$1"></span>');

    [].forEach.call(prose.querySelectorAll('pre code[class]'), function (code) {
      highlight(code, code.className.replace('language-', ''));
    });

    // Resolve relative links and images against the Markdown file, so the same
    // paths work here and when the file is viewed on GitHub.
    [].forEach.call(prose.querySelectorAll('[src], a[href]'), function (n) {
      var attr = n.hasAttribute('src') ? 'src' : 'href', v = n.getAttribute(attr);
      if (!/^([a-z][a-z0-9+.\-]*:|\/|#)/i.test(v)) n.setAttribute(attr, new URL(v, url).href);
      if (attr === 'href' && n.host !== location.host) { n.target = '_blank'; n.rel = 'noopener'; }
    });

    // KaTeX loads only for pages that contain math. Until it does, the TeX source shows.
    var nodes = [].slice.call(prose.querySelectorAll('[data-math]'));
    if (!nodes.length) return;
    nodes.forEach(function (n) { n.textContent = math[n.getAttribute('data-math')].tex; });
    var css = el('link'), js = el('script');
    css.rel = 'stylesheet'; css.href = KATEX + 'katex.min.css'; css.integrity = KATEX_CSS; css.crossOrigin = 'anonymous';
    js.src = KATEX + 'katex.min.js'; js.integrity = KATEX_JS; js.crossOrigin = 'anonymous';
    js.onload = function () {
      nodes.forEach(function (n) {
        var m = math[n.getAttribute('data-math')];
        window.katex.render(m.tex, n, { displayMode: m.display, throwOnError: false });
      });
    };
    document.head.appendChild(css);
    document.head.appendChild(js);
  }

  window.Molt = { el: el, fmtDate: fmtDate, render: render, highlight: highlight, markdown: markdown };
})();
