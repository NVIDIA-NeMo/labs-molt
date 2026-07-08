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
pip install pre-commit
pre-commit install
```

## Sign your work

We require that all contributors "sign-off" on their commits, certifying the
[Developer Certificate of Origin (DCO)](https://developercertificate.org/) —
that the contribution is your original work, or you have rights to pass it on
under the same open-source license:

```
git commit -s -m "Your commit message"
```

This appends a `Signed-off-by: Your Name <your@email.com>` line to the commit
message. Any contribution containing commits that are not signed off will not
be accepted.
