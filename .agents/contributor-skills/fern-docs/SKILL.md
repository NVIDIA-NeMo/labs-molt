---
name: fern-docs
description: Maintain the NeMo AutoModel Fern docs site under docs/ (MDX content) + docs/fern/ (infra) — add, update, move, or remove pages; manage redirects, slugs, navigation, and version aliases; run validation and previews.
when_to_use: Editing or adding documentation pages, fixing broken links, renaming a slug, updating the sidebar, adding a redirect, regenerating the API library reference, debugging fern check / broken-link errors, cutting a new version train, 'edit docs', 'add doc page', 'fern check failing', 'preview fails locally'.
---

# Fern Docs Maintenance — NeMo AutoModel

Unified skill for adding, updating, moving, and removing pages on the NeMo AutoModel Fern documentation site at `docs.nvidia.com/nemo/automodel`.

## Scope rule

**Nightly MDX content lives at the top level of `docs/`** (e.g. `docs/index.mdx`, `docs/guides/llm/finetune.mdx`). **`docs/fern/` holds only Fern build infrastructure** — config, theme, and components. New pages, release notes, migration guides → add as a top-level `.mdx` under `docs/`.

**Only the nightly tree is kept on `main`.** Frozen backward-version snapshots live on the `docs-archive` branch and are restored at build time — see *Archived backward versions* below.

**Two real content trees, plus a GA alias YAML.**

- `docs/` — bleeding-edge (nightly) tree. Every PR lands here. Mounted at the `nightly` URL slug via `docs/fern/versions/nightly.yml` (paths reach back up via `../../<rel>.mdx`).
- `docs/fern/versions/v0.4/pages/` — frozen 0.4.0 GA snapshot. **Not on `main`**: it lives on the `docs-archive` branch and is restored under this path by `make docs-stitch` (local) or the `stitch-fern-versions` CI action (build). Mounted at the `v0.4` URL slug via `v0.4.yml`. Only changes via deliberate back-port (on `docs-archive`).
- `docs/fern/versions/latest.yml` — GA alias. Its `path:` lines mount the current GA's content (today: `./v0.4/pages/...`). Repointed at the next GA's tree when one is cut.

The nightly and v0.4 trees were byte-for-byte identical at the moment 0.4.0 shipped, but they will diverge as nightly accumulates post-release edits and v0.4 stays frozen. **Default editing target is `docs/` top-level.** Back-ports to a frozen version happen on the `docs-archive` branch, not here — call out the divergence in the PR description.

### Archived backward versions

Fern has no native way to source a version train's prose from another git ref: `fern generate --docs` reads the single local working tree, and publish is a full-site snapshot (a train missing from the tree is *unpublished*). So frozen GA pages are kept off `main` on the **`docs-archive` branch** and restored before every Fern build.

- **The registry** lives inline in each `fern-docs-*` workflow as `archived-versions: |` lines of `<version-dir>=<git-ref>` (today: `v0.4=docs-archive`). The `<git-ref>` is opaque: a **branch** (default — all frozen versions in one place, easy back-ports) or a **tag** like `docs/v0.4.0` (immutable snapshot). `latest` is an alias of an existing pages tree and needs **no** entry.
- **The mechanism** is the `.github/actions/stitch-fern-versions` composite action: `git fetch --depth=1 origin <ref>` then `git restore --source=FETCH_HEAD -- docs/fern/versions/<vdir>/pages`. It runs in `publish-fern-docs.yml`, `fern-docs-ci.yml`, and `fern-docs-preview-build.yml`. `docs.yml` + the nav YAMLs always come from the live checkout, never the archive ref, so frozen prose can't drift into the wrong train.
- **Locally**, `make docs` / `docs-check` / `docs-preview` depend on `make docs-stitch`, which does the same restore. Override the ref with `make docs-check ARCHIVE_REF=docs/v0.4.0`.
- The restored path is gitignored on `main`, so a local stitch won't show the pages as untracked.

**Sidebar fidelity rule.** Section captions, page titles, and Model Coverage child ordering must match the **published v0.4.0 sidebar at docs.nvidia.com/nemo/automodel/latest** verbatim. Don't silently shorten a title or reorder siblings — the docs PM and content engineers diff against the published site and any drift is treated as a regression. If you want a shorter sidebar label, change the toctree-derived display name in the source — never just retitle in the MDX.

## Layout at a glance

```
docs/                                ← nightly MDX (top level)
├── index.mdx, breaking-changes.mdx, release-notes.mdx, ...
├── about/, guides/, model-coverage/, launcher/, api-reference/
├── *.png / *.jpg                    ← page-scoped images
└── fern/                            ← infra only
    ├── fern.config.json             # Org slug + Fern CLI pin (5.29.0+)
    ├── docs.yml                     # Site config + global-theme: nvidia (inherits
    │                                #   logos / footer / theme CSS / fonts / OneTrust JS
    │                                #   from NVIDIA/fern-components)
    ├── components/                  # BadgeLinks.tsx, Tag.tsx
    │                                #   (repo-specific; NVIDIA footer ships in global theme)
    ├── versions/
    │   ├── nightly.yml              # Nav for nightly — paths → ../../<rel>.mdx (up into docs/)
    │   ├── v0.4.yml                 # Nav for frozen 0.4.0 — paths → ./v0.4/pages/
    │   ├── v0.4/pages/              # Frozen 0.4.0 MDX (back-ports only)
    │   └── latest.yml               # GA alias — paths → ./v0.4/pages/ today; repointed at next GA cut
    └── product-docs/                # GENERATED Python API reference (gitignored)
```

```
File                                                          URL
─────────────────────────────────────────────────────────────  ────────────────────────────────────────────
docs/get-started/installation.mdx                              /nightly/get-started/installation
docs/fern/versions/v0.4/pages/get-started/installation.mdx     /latest/get-started/installation
                                                               /v0.4/get-started/installation
```

## Operations

### Add a page

1. Gather: title, target section, filename (kebab-case `.mdx`), subdirectory under `docs/`.
2. Create the MDX at `docs/<subdir>/<filename>.mdx` with frontmatter:

   ```mdx
   ---
   title: "<Page Title>"
   description: "One-line SEO description (or empty string)"
   position: 4
   ---

   <body — typically no leading `# H1`; Fern renders the title automatically>
   ```

3. Add a `- page:` entry to `docs/fern/versions/nightly.yml` under the right `section:`, with `path:` reaching up into `docs/` via `../../`:

   ```yaml
   - page: "<Page Title>"
     path: ../../<subdir>/<filename>.mdx
     slug: <short-url-segment>
   ```

4. `make docs-check` (runs `fern check`) and verify URL resolves on `make docs` preview. There is no `nightly.yml` ↔ `latest.yml` alias-sync step under this layout — `latest.yml` mounts the frozen v0.4 tree, not nightly, so it's intentionally out of sync.

### Update a page

1. Locate by path, title, or keyword: `grep -rn "<keyword>" docs/ --include="*.mdx" --exclude-dir=fern`.
2. **Content only** — edit the single MDX file at `docs/<...>.mdx`.
3. **Title change** — update the frontmatter `title:` and update the `- page:` entry's display label in `docs/fern/versions/nightly.yml`.
4. **Section move** — `git mv` the file within `docs/`, update `path:` in `nightly.yml`, fix incoming links.
5. **Slug change** — change `slug:` in the YAML (or rename the file and let the default slug update). Add a `redirects:` entry in `docs/fern/docs.yml` so the old URL keeps working.

### Redirect quirks

Four things to watch when editing `redirects:` in `docs/fern/docs.yml`:

1. **`:path*` does NOT match the empty-path case.** `/<basepath>/v0.4/:path*/index.html` will *not* match `/<basepath>/v0.4/index.html` (where `:path*` would have to be empty). Each version-root `index.html` needs its own explicit rule. NeMo Curator (NVIDIA-NeMo/Curator#1938) discovered this when their version-root URLs 404'd. AutoModel ships explicit rules for `latest`, `v0.4`, `nightly`, and the legacy `0.4` form — when you add a new version slug, add four new explicit rules: `<slug>/index.html`, `<slug>/index`, plus the same two for any legacy form (e.g. `0.5` → `v0.5`).
2. **Older un-migrated versions need a fallback.** Whatever versions the published Sphinx site exposed (check the version-switcher dropdown on `docs.nvidia.com/nemo/<product>/latest/`) but you didn't migrate into Fern still need to resolve. The pattern: redirect each old slug's URLs to the equivalent path under `/latest/` so external bookmarks and search results land on the closest current page instead of 404ing. Five rules per old version: `<slug>/index.html`, `<slug>/index`, `<slug>/:path*/index.html`, `<slug>/:path*`, `<slug>/:path*.html` — all destinations `/latest/...`. AutoModel ships these for `0.3.0`, `0.2.0`, `0.1.0`.
3. **Order matters.** Specific rules must come before catch-alls — Fern uses first-match. Slot new rules *before* the `:path*/index.html` and `:path*.html` catch-alls.
4. **Don't ship `redirects: []`** then re-run the redirect generator on top — it replaces the whole `redirects:` block. Edit by hand or back up the existing rules first.

### Remove a page

1. Find incoming links: `grep -rn "<filename>" docs/ --include="*.mdx" --exclude-dir=fern`.
2. `git rm docs/<...>.mdx`.
3. Remove the `- page:` block from `docs/fern/versions/nightly.yml`.
4. Fix or delete incoming links.
5. Add a redirect in `docs/fern/docs.yml` if the URL was public.

### Worked example: add a guide

Request: *"Add a fine-tuning guide for Qwen3.6 under Recipes & E2E Examples."*

1. Create `docs/guides/llm/qwen3-6-finetune.mdx`:

   ```mdx
   ---
   title: "Fine-Tune Qwen3.6"
   description: "End-to-end SFT and PEFT recipes for Qwen3.6 on NeMo AutoModel"
   ---

   This guide walks through fine-tuning Qwen3.6 with NeMo AutoModel...
   ```

2. Add to `docs/fern/versions/nightly.yml` under the `Recipes & E2E Examples` section, slotted in publication-order with the other fine-tune entries:

   ```yaml
   - page: "Fine-Tune Qwen3.6"
     path: ../../guides/llm/qwen3-6-finetune.mdx
     slug: qwen3-6-finetune
   ```

3. `make docs-check` then `make docs` to preview at `http://localhost:3002/nightly/recipes-e2e-examples/qwen3-6-finetune`.

### Worked example: rename a slug with a redirect

Request: *"Rename `/recipes-e2e-examples/sft-peft` to `/recipes-e2e-examples/fine-tuning`."*

1. Edit `docs/fern/versions/nightly.yml`, change the `slug:` on the SFT & PEFT entry from `sft-peft` to `fine-tuning`.
2. Add a redirect to `docs/fern/docs.yml`:

   ```yaml
   redirects:
     - source: "/:version/recipes-e2e-examples/sft-peft"
       destination: "/:version/recipes-e2e-examples/fine-tuning"
   ```

3. `grep -rn "/recipes-e2e-examples/sft-peft" docs/ --include="*.mdx" --exclude-dir=fern` and update incoming body links.

## Content guidelines

NeMo AutoModel uses **Fern-native MDX components**. Don't use GitHub `> [!NOTE]` syntax — it doesn't render in MDX.

| Purpose | Component |
|---|---|
| Neutral aside | `<Note>...</Note>` |
| Helpful tip | `<Tip>...</Tip>` |
| Informational callout | `<Info>...</Info>` |
| Warning | `<Warning>...</Warning>` |
| Error / danger | `<Error>...</Error>` |
| Card grid on landing pages | `<Cards>` with `<Card title="..." href="...">` children |
| Card chips ("start here", "5 min") | `<Tag variant="primary">label</Tag>` — sphinx-design `{bdg-*}` mapping |
| Header badge rows (PyPI, license, GitHub) | `<BadgeLinks badges={[{href, src, alt}, ...]} />` |

Required imports when using `<Tag>` or `<BadgeLinks>` (`landing_badges.py` adds these in the post stage; in hand-written pages add them yourself):

```mdx
import { Tag } from "@/components/Tag";
import { BadgeLinks } from "@/components/BadgeLinks";
```

`<Tag variant="...">` accepts: `primary`, `secondary`, `success`, `warning`, `danger`, `info`, `light`, `dark` (1:1 with sphinx-design `{bdg-*}` variants).

Page-scoped images live alongside the MDX file (e.g. `docs/guides/audio/qwen_omni_asr.png`). Reference them with relative paths (`./image.png`), not absolute (`/image.png`) — Fern's path resolver doesn't normalize root-relative image paths the same way as link targets. The NVIDIA logos and favicon come from the `nvidia` global theme; do not add them locally.

## Frontmatter

```yaml
---
title: "<Page Title>"        # required — Fern renders this as the page H1
description: ""              # required (may be "") — SEO meta description
position: 1                  # optional — orders auto-discovered pages within a folder
---
```

**Don't repeat the title as a leading `# H1` in the body.** Fern already renders `title:` at the top of the page, and a duplicate creates a double heading. The post-stage `remove_duplicate_h1.py` strips them when title and H1 match exactly, but it can't catch near-duplicates (e.g. `title: "About"` vs `# About NeMo AutoModel`) — keep the body H1-free, or promote the descriptive form to `subtitle:` if you want both visible.

## Internal links

Use **version-agnostic** paths — no `/latest/`, `/v0.4/`, or `/nightly/` prefix:

```mdx
[Install NeMo AutoModel](/get-started/installation)
[Llama coverage](/model-coverage/large-language-models/llama)
```

The same MDX backs every version slug; a hard-coded prefix sends readers across versions unintentionally. URL slugs come from explicit `slug:` overrides in the version YAML (set during the migration so URLs stay short while sidebar titles match the verbose published H1s) — so `Install NeMo AutoModel` is at `/get-started/installation`, not `/get-started/install-nemo-automodel`.

For cross-repo references (yaml configs, Python source), use absolute GitHub URLs:

```mdx
[mistral4_medpix.yaml](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/vlm_finetune/mistral4/mistral4_medpix.yaml)
```

## Validate

```bash
make docs-check          # MDX syntax validation + `fern check`
```

Run from `docs/fern/` (`cd docs/fern && make docs-check`) or anywhere with `make -C docs/fern docs-check`.

`make docs-check` must pass before commit. It runs a no-secret MDX parser before `fern check`, so raw HTML must be valid JSX (for example, use `<img ... />`, not `<img ...>`). The dev server's broken-link warnings for version-prefixed routes (e.g. `/latest/get-started/installation` from MDX that uses `/get-started/installation`) are **false positives** — Fern's strict validator doesn't resolve version-agnostic links. The published site renders them correctly. The URLMap-based `validate_fern_internal_links.py` (under the convert-to-fern toolkit) is authoritative.

To regenerate the autodoc library reference (gitignored under `docs/fern/product-docs/`):

```bash
make docs                # runs `fern docs md generate` then `fern docs dev`
```

`fern docs md generate` populates `docs/fern/product-docs/` from the `nemo_automodel` package source declared in `docs.yml` `libraries:` block. Without this step, a cold `fern docs dev` fails with `Folder not found: ./product-docs/...`.

## Preview and publish

| Goal | Command |
|---|---|
| Local preview at `http://localhost:3002` | `make docs` |
| Validation only (no server) | `make docs-check` |
| Shared preview URL on `*.docs.buildwithfern.com` (needs `DOCS_FERN_TOKEN`) | `make docs-preview` |
| Trigger production publish workflow on `origin/main` | `make docs-publish` |

PRs that touch `docs/**` get an automatic Fern preview URL posted as a 🌿 comment by `fern-docs-preview-comment.yml`. No manual step.

Every job below first runs the `stitch-fern-versions` action to restore the archived
backward-version pages (the `docs-archive` branch) into the working copy — the frozen
trees are not on `main`.

```
                    ┌─ fern-docs-ci.yml                  → stitch → MDX syntax → fern check (push to pull-request/<n>)
PR (touches docs/) ─┼─ fern-docs-preview-build.yml       → stitch → upload docs/ artifact (no secrets)
                    └─ fern-docs-preview-comment.yml     → 🌿 preview URL comment (consumes artifact)

Push to main (touches docs/) → publish-fern-docs.yml → stitch → docs.nvidia.com/nemo/automodel
Tag push (docs/v*)           → publish-fern-docs.yml → stitch → docs.nvidia.com/nemo/automodel
Manual dispatch              → publish-fern-docs.yml → stitch → docs.nvidia.com/nemo/automodel
```

The preview-comment + publish jobs require the `DOCS_FERN_TOKEN` org secret (already wired for `build-docs.yml`).

## Cutting a new version train

When NeMo AutoModel ships a new GA (e.g. `v0.5`):

1. `mkdir -p docs/fern/versions/v0.5/pages && rsync -a --exclude='fern' docs/ docs/fern/versions/v0.5/pages/` — fresh frozen snapshot of nightly at release time.
2. `cp docs/fern/versions/nightly.yml docs/fern/versions/v0.5.yml` and rewrite `../../` path prefixes to `./v0.5/pages/`.
3. Update `docs/fern/versions/latest.yml` to point at the new train: `cp docs/fern/versions/v0.5.yml docs/fern/versions/latest.yml`. (`latest` is the auto-bumping GA alias.)
4. In `docs/fern/docs.yml` `versions:`, add a new frozen-pin entry (`display-name: "0.5.0 · 26.07"`, `slug: v0.5`, `availability: stable`) and keep the previous pin (`v0.4`) for permalink stability.
5. **Archive the frozen tree off `main`** (see *Archived backward versions*): commit `docs/fern/versions/v0.5/pages/` onto the `docs-archive` branch and push it, then `git rm -r docs/fern/versions/v0.5/pages` from `main` and add that path to `.gitignore`. The `v0.5.yml`/`latest.yml`/`docs.yml` config stays on `main`.
6. Add `v0.5=docs-archive` to the `archived-versions:` registry in `publish-fern-docs.yml`, `fern-docs-ci.yml`, and `fern-docs-preview-build.yml` (and the `.gitignore` line). `docs/` keeps moving forward as the bleeding-edge tree; the new frozen snapshot only changes via deliberate back-port on `docs-archive`.
7. Promote `nightly` to `availability: stable` if and when its content tree gets cut over.
8. Tag `docs/v0.5.0` and push to publish.

## Commits and DCO

Every commit needs a `Signed-off-by:` trailer:

```bash
git commit -s -m "docs: add fine-tuning guide for Qwen3.6"
```

If sign-off is missing on a recent commit, amend with `git commit --amend -s`. PR titles follow Conventional Commits: `docs(fern): <short summary>`. See [`AGENTS.md`](../../AGENTS.md) for the full repo commit convention.

## Debugging

| Symptom | Fix |
|---|---|
| `fern check` YAML error | 2-space indent; `- page:` inside `contents:`; `path:` is relative to `nightly.yml`'s location (so nightly entries reach back up via `../../`); `slug:` must not collide with siblings |
| Page 404 in preview | Missing `slug:` override (default slugifies the long display title) or `position:` collision in an auto-discovered folder |
| `Folder not found: ./product-docs/...` on `fern docs dev` | Run `make docs` once to populate the library reference |
| `[ERR_PNPM_IGNORED_BUILDS]` on first `fern docs dev` | pnpm 10+ blocks esbuild's postinstall — `pnpm config set onlyBuiltDependencies '["esbuild"]' --location global`, then `rm -rf ~/.fern/app-preview` and retry |
| Broken-link warning on version-agnostic path | `fern docs broken-links` false-positives; URLMap-based validator is authoritative |
| `JSX expressions must have one parent element` | Wrap multi-element JSX in `<>...</>` or a `<div>` |
| Old Sphinx URL breaks | Add a `redirects:` entry in `docs/fern/docs.yml`; the redirect generator already handles `/index.html` and `.html` legacy forms |
| Image not rendering | Use relative path (`./image.png`) for page-scoped images, not root-relative (`/image.png`) |
| Sidebar caption looks shortened vs published site | Compare against `docs.nvidia.com/nemo/automodel/latest` and restore the verbatim title in `docs/fern/versions/nightly.yml` |
| `path: ../../foo.mdx` doesn't resolve | Confirm the MDX file is at `docs/foo.mdx` (top level), not still under `docs/fern/versions/nightly/pages/` — that legacy tree no longer exists |
| `fern check` fatals on missing `./v0.4/pages/...` paths | The frozen v0.4 tree isn't checked out. Run `make docs-stitch` (or `make docs-check`, which depends on it) to restore it from the `docs-archive` branch |
| `archive ref '...' does not contain '...'` in CI | The `stitch-fern-versions` action couldn't find the version's pages on its registry ref. Confirm the `docs-archive` branch (or the configured tag) still holds `docs/fern/versions/<vdir>/pages` |

## Key references

| File | Purpose |
|---|---|
| `docs/fern/docs.yml` | Site config — `instances`, `versions`, `redirects`, `libraries`, theme |
| `docs/fern/versions/nightly.yml` | Canonical nav tree — paths reach up into `docs/` via `../../` |
| `docs/fern/versions/{latest,v0.4}.yml` | Frozen GA nav (mount `./v0.4/pages/...`) |
| `docs/` (top-level *.mdx) | Nightly MDX content (~140 pages + page-scoped images) |
| `docs/fern/versions/v0.4/pages/` | Frozen 0.4.0 snapshot — **on the `docs-archive` branch, not `main`**; stitched in at build time |
| `docs-archive` branch | Holds all frozen backward-version `pages/` trees; restored by `stitch-fern-versions` / `make docs-stitch` |
| `.github/actions/stitch-fern-versions/` | Composite action that restores archived version pages before any Fern build |
| `docs/fern/components/` | `BadgeLinks.tsx`, `Tag.tsx` (repo-specific; NVIDIA footer ships via `global-theme: nvidia`) |
| `docs/fern/README.md` | Human-facing orientation |
| `docs/fern/Makefile` | `make docs / docs-check / docs-preview / docs-publish` (run from `docs/fern/` or via `make -C docs/fern`) |
| `.github/workflows/fern-docs-*.yml` | CI: check, preview build, preview comment |
| `.github/workflows/publish-fern-docs.yml` | CI: publish to docs.nvidia.com/nemo/automodel |
