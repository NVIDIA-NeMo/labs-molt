# Molt docs

Pages are Markdown files in `pages/`. One page, `index.html`, lists them and renders them
in the browser, so there is no build step.

| URL | Shows |
|---|---|
| `guide/` | the library: every page in `pages.js`, by section |
| `guide/?p=<name>` | the page in `pages/<name>.md` |

## Add or edit a page

1. Write `pages/<name>.md`, starting with `# Title`.
2. Add `[name, title, summary]` to a section in `pages.js`. The order there is the reading
   order of the sidebar and of the previous and next links.

Search needs no step: it reads the pages listed in `pages.js`.

Link to another page by file name, such as `[Recipes](recipes.md)`, so the link also works
on GitHub. Link to files in the repository by their full GitHub URL.

## Generated pages

`cli-rl.md`, `cli-sft.md`, and `api-agents.md` are written from the source code. Do not
edit them by hand.

```bash
python docs/guide/build_reference.py
```

```bash
python docs/guide/build_reference.py --check
```

The first command rewrites the three pages from the argument parsers and from
`molt/agents`. The second fails if they are stale, or if any page names a flag that
neither CLI defines. Run it after changing a flag.

## Search

`search.js` runs in the browser, on the homepage and the blog as well as here: a page gets the
palette by loading `pages.js` and `search.js` and having a `data-search` button. On the first
search it fetches every page in `pages.js` and indexes one entry per page, per section, and per
table row whose first cell is code, such as a flag. [MiniSearch](https://github.com/lucaong/minisearch), loaded from jsDelivr, ranks the
matches. A result links to `guide/?p=<name>&at=<text>`, and the page scrolls to the heading or
the table row with that text.

The palette also offers to ask an AI assistant. That link carries the question and the address
of the repository, because an assistant cannot run the site's script but can read the Markdown
pages on GitHub.

## Preview

```bash
python3 -m http.server 4173 --directory docs
```

Then open <http://localhost:4173/guide/>.
