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

// Shared by the homepage and the blog: renders the entries of blog/posts.js.
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

  window.Molt = { el: el, fmtDate: fmtDate, render: render };
})();
