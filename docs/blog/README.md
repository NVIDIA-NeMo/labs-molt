# Molt blog

Posts are Markdown files. One page, `index.html`, lists them and renders them in the
browser, so there is no build step.

| URL | Shows |
|---|---|
| `blog/` | every entry in `posts.js`, newest first |
| `blog/?p=<slug>` | the post in `posts/<slug>.md` |

The homepage lists the newest eight entries of `posts.js` and puts the latest one in the
chip above the title.

## Add a post

1. Copy `posts/_template.md` to `posts/<slug>.md` and write the post. Use letters,
   digits, `-`, and `_` in the slug.
2. Add an entry at the top of `posts.js`.

| Field | Meaning |
|---|---|
| `title` | headline on the card and at the top of the post |
| `date` | `"YYYY-MM-DD"` or `"YYYY-MM"` |
| `tag` | category label: Release, Research, Blog, Tutorial |
| `summary` | one or two sentences for the card and the post's intro |
| `slug` | file name in `posts/` without `.md`, for a post on this site |
| `url` | full link, for an item hosted elsewhere (a paper, a project page); no Markdown file is needed |
| `tone` | accent color: `green`, `violet`, `blue`, or `amber` |
| `cover` | `[big text, small text]` on the card cover in `blog/` |
| `image` | cover image instead of `cover`, relative to `blog/`, such as `posts/images/my-post.jpg` |

Every entry needs `title`, `date`, `tag`, `summary`, one of `slug` or `url`, and one of
`cover` or `image`.

## Writing

- Start the file with `# Title`. It is replaced by the `title` in `posts.js` on the site
  and keeps the file readable on GitHub.
- Math: `$inline$`, `$$display$$`, or a `math` code fence. Rendered with KaTeX.
- Images: put them in `posts/images/` and link them relative to the post, for example
  `![Alt text](images/figure.png)`.

## Preview

```bash
python3 -m http.server 4173 --directory docs
```

Then open <http://localhost:4173/blog/>. The template renders at
<http://localhost:4173/blog/?p=_template>.
