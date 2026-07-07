# Contributing to Molt

## Development Environment

### Prerequisites

- Python 3.10, 3.11, or 3.12
- [uv](https://github.com/astral-sh/uv) for Python package management
- Git for version control

### Setup

1. Install uv:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the repository:

   ```bash
   git clone <repository-url>
   cd labs-molt
   ```

3. Set up the development environment:

   ```bash
   uv sync --all-extras --group test --group dev
   ```

4. Install pre-commit hooks:

   ```bash
   uv run pre-commit install
   ```

## Code Style and Quality

Molt uses Ruff for formatting, import sorting, and linting:

```bash
uv run ruff format .
uv run ruff check . --fix
```

Run all configured pre-commit hooks before submitting a pull request:

```bash
uv run pre-commit run --all-files
```

## Testing

Run the test suite with:

```bash
uv run pytest
```

Run a specific test file with:

```bash
uv run pytest tests/test_specific_module.py
```

## Pull Request Guidelines

Before submitting a pull request:

1. Create an issue for significant changes so maintainers can discuss the approach.
2. Use a descriptive branch name, such as `feature/add-new-agent` or `fix/checkpoint-resume`.
3. Run code quality checks and tests:

   ```bash
   uv run pre-commit run --all-files
   uv run pytest
   ```

Pull requests should include:

1. A clear description of what changed and why.
2. Tests for new functionality or bug fixes.
3. Documentation updates when behavior or user-facing workflows change.
4. Backwards compatibility notes for any breaking changes.

## Signing Your Work

We require that all contributors sign off on their commits. This certifies that the contribution is your original work, or that you have rights to submit it under the same license or a compatible license.

Any contribution that contains commits without a Signed-off-by line will not be accepted.

To sign off on a commit, use the `--signoff` or `-s` option:

```bash
git commit -s -m "Add cool feature"
```

This appends a line like this to your commit message:

```text
Signed-off-by: Your Name <your@email.com>
```

Full text of the DCO:

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Running GitHub CI

There are two ways to trigger CI tests on your pull request.

### Automatic CI Triggering

If your GitHub user is configured to use [signed commits](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification), CI tests will run automatically when you push commits to your pull request.

Signed commits are different from signing off on commits, which uses the `-s` flag described above.

### Manual CI Triggering

If you do not have signed commits set up, trigger CI tests manually by commenting on your pull request:

```text
/ok to test <commit-SHA>
```

For example:

```text
/ok to test a1b2c3d4e5f6
```

Add this comment for each new commit you push so CI runs against the latest changes.
