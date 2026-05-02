# Security & Best Practices

This document outlines what must NEVER be committed to the repository and the
practices used to keep secrets, credentials, and personal data out of git.

---

## What is gitignored (and why)

| Path / pattern               | Reason                                                  |
|------------------------------|---------------------------------------------------------|
| `.env`, `.env.*`             | Holds real API keys (Alpha Vantage, EODHD, FRED, etc.)  |
| `config/alpha.yaml`          | Contains absolute filesystem paths for the local data root |
| `cache/`, `_px_cache/`       | Re-derivable backtest caches (large parquet files)      |
| `_px_reports/`, `plots*/`    | Backtest outputs — regenerable, can be GBs              |
| `logs/`, `*.log`, `*.pid`    | Runtime artifacts                                       |
| `*.parquet`, `*.h5`, `*.xlsx`| Large data files (test fixtures explicitly allow-listed) |
| `data/`, `ProjectX_MasterData/` | Raw market-data dumps — usually licensed              |
| `.DS_Store`, `Thumbs.db`     | OS junk                                                 |
| `.venv/`, `venv/`            | Python virtual environment                              |
| `*.pem`, `*.key`, `*.crt`    | TLS / SSH keys                                          |

The full set of patterns lives in `.gitignore`.

---

## Secrets management — required practices

### 1. NEVER hardcode API keys in source files

**Bad** (don't do this):
```python
api_key = "abc123XYZ"   # ← will be committed and grep-able forever
```

**Good** (read from environment):
```python
import os
api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
if not api_key:
    raise SystemExit("ALPHAVANTAGE_API_KEY is not set")
```

### 2. Use `.env` files locally, but never commit them

```bash
cp .env.example .env       # one-time setup
# edit .env with real keys
set -a && source .env && set +a   # load into shell
```

For Python scripts that read it directly, use `python-dotenv`:
```python
from dotenv import load_dotenv
load_dotenv()
```

### 3. Rotate immediately if a secret is committed

A secret pushed to git is permanently exposed even after deletion (GitHub
caches branches, forks, archives, search indexes). Steps if it happens:

1. Treat the secret as compromised — **rotate it immediately** at the provider
2. Use `git filter-repo` or BFG Repo-Cleaner to scrub history
3. Force-push the cleaned history (only if no collaborators) and notify any
4. Add a pre-commit hook to prevent re-occurrence (see below)

### 4. Pre-commit secret scanning

Install `pre-commit` and add a secret-scanner hook:

```bash
pip install pre-commit detect-secrets
pre-commit install
```

`.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.5.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks
```

Then create the baseline:
```bash
detect-secrets scan > .secrets.baseline
git add .secrets.baseline .pre-commit-config.yaml
```

### 5. Use a secrets manager for production

For deployments (not local dev), prefer:
- **AWS Secrets Manager** / **Parameter Store**
- **Google Secret Manager**
- **HashiCorp Vault**
- **1Password / Bitwarden** for human-managed secrets

Never store production API keys in `.env` on shared servers.

---

## Hardcoded paths — keep them out

`config/alpha.yaml` (the live one) contains absolute paths like
`/Users/<you>/Desktop/...` that:
- Leak your username
- Don't work for anyone else
- Are not portable across machines

That's why **only `config/alpha.yaml.example`** is committed. Each user
copies it to `config/alpha.yaml` and edits the `<DATA_ROOT>` placeholder.
The real file is gitignored.

For Python scripts (e.g. `scripts/visualize_iter10.py`), prefer reading
the data path from an environment variable or argparse argument rather
than hardcoding it.

---

## Data licensing & PII

- **Russell 1000 holdings** (`R1000.xlsx`) — likely subject to FTSE Russell's
  licensing terms. Do not commit. Distribute via your data vendor.
- **EODHD price/fundamentals dumps** — licensed; redistribution may violate ToS.
- **News articles / sentiment data** — usually scraped from a third-party
  provider; check their ToS before sharing.
- **Personal trading data** — never commit. Includes account numbers, tax IDs,
  PII, or anything that could identify a real account.

When in doubt, treat data as confidential and keep it outside the repo.

---

## Pre-push checklist

Before every `git push`, run through this:

- [ ] `git status` shows only intended files
- [ ] `git diff --staged` does not contain any keys, tokens, passwords
- [ ] No absolute paths to `/Users/<me>/...` in any new file
- [ ] No `.env`, `*.parquet`, `*.log`, `*.xlsx` staged
- [ ] No `print()` left in code that dumps secrets
- [ ] Tests pass: `pytest`

If you've staged a secret by accident, run:
```bash
git reset HEAD <file>            # unstage
git rm --cached <file>           # if already tracked
# then add to .gitignore and re-commit
```

---

## Audit commands (run periodically)

```bash
# Find any tracked file with /Users/ paths (should return nothing)
git ls-files | xargs grep -l "/Users/" 2>/dev/null

# Find any tracked file containing API keys (broad regex; trim noise)
git grep -nE "(api[_-]?key|secret|token|passw|aws_access|private[_-]?key)\s*=\s*[\"'][^\"']{8,}"

# List the largest tracked files (often indicates accidental data commits)
git ls-files | xargs du -h 2>/dev/null | sort -rh | head -20

# Check pack/object sizes (large objects suggest history bloat)
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | sort -k3 -n -r | head -20
```

---

## If something does leak

1. **Rotate** the credential at the provider
2. **Rewrite history** with `git filter-repo`:
   ```bash
   pip install git-filter-repo
   git filter-repo --invert-paths --path path/to/leaked-file
   git filter-repo --replace-text replacements.txt   # for inline secrets
   ```
3. **Force-push** the cleaned history — coordinate with collaborators first
4. **Invalidate** old clones: collaborators must re-clone after history rewrite
5. **Notify** anyone who may have access to the leaked credential
