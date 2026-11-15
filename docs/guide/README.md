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

## Preview

```bash
python3 -m http.server 4173 --directory docs
```

Then open <http://localhost:4173/guide/>.
