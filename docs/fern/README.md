# Molt Fern Docs

This directory holds the Fern configuration for the Molt documentation site.

The docs content lives in `../`. The nightly navigation is defined in `versions/nightly.yml`, and the generated Python API reference is written under `product-docs/`. Run Fern commands from this directory so `docs.yml` can resolve paths correctly.

## Quick Links

| What | Where |
|---|---|
| Published site | https://docs.nvidia.com/nemo/molt |
| Fern dashboard | https://dashboard.buildwithfern.com |
| Fern config | `docs.yml` |
| Nightly navigation | `versions/nightly.yml` |

## Local Setup

Install Node.js 20 or newer. Docker is required for no-auth local API reference
generation.

The Fern CLI is not installed globally. The npm scripts fetch and run the
`fern-api` version pinned in `fern.config.json` with `npm exec`.

Run these commands from this directory:

```bash
cd docs/fern

npm run login
```

If login fails with an organization access error, sign in to https://dashboard.buildwithfern.com with an account that has access to the NVIDIA Fern organization, then run the login command again. The local dev server needs this access to fetch `global-theme: nvidia`.

## Validate

Run Fern's config and docs validation:

```bash
npm run check
```

## Local Preview

For a private checkout, use the local path-backed API reference generator:

```bash
npm run generate:library:local
```

This parses `../../molt` with Docker and writes generated pages to
`product-docs/`. It also runs `npm run sanitize:generated` because Fern's
Python generator can emit MDX-invalid JSX attributes for Pydantic field types.

Before serving locally, temporarily comment out or remove the local library block
from `docs.yml`:

```yaml
  molt-local:
    input:
      path: ../../molt
    output:
      path: ./product-docs/molt/Full-Library-Reference
    lang: python
```

Then start the local docs server:

```bash
npm run dev
```

Fern serves the local preview at http://localhost:3000.

Why this extra step is needed: `fern docs md generate --local` supports
`input.path`, but `fern docs dev` currently rejects path-backed libraries during
config loading. The generated `product-docs/` files remain available after you
comment out the local library entry.

For a full git-backed API reference that matches preview and publish workflows,
sign in with `npm run login`, then run:

```bash
npm run generate:library
```

This requires Fern's remote parser to clone
`https://github.com/NVIDIA-NeMo/labs-molt`, so it fails while the repository is
private unless Fern has GitHub access.

## Hosted Preview

Hosted previews require `DOCS_FERN_TOKEN`:

```bash
export FERN_TOKEN="$DOCS_FERN_TOKEN"
npm run preview
```

## Troubleshooting

| Error | Fix |
|---|---|
| `Failed to fetch global theme "nvidia": HTTP 403` | Run `npm run login` with an account that has access to the NVIDIA Fern organization. |
| `Library 'molt-local' uses 'path' input which is not yet supported` | Comment out the `molt-local` library block before `npm run dev`. |
| `Failed to clone repository ... (CLONE_FAILED)` | Use `npm run generate:library:local` for private checkouts. The git-backed generator requires Fern to clone the GitHub repository. |
| `Unexpected character` in generated `product-docs/.../*.mdx` | Re-run `npm run generate:library:local`; it runs the generated MDX sanitizer after Fern generation. |
