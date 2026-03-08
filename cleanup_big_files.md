# Cleaning Up Big Files from Git History

## Context

Accidentally committed large log files via `git add .` and pushed to GitHub:

```
6260445  example/110_gentoo/docker_min_hpz6/logs/setup01_run_20260306_200733.log
4553609  example/110_gentoo/docker_min_hpz6/logs/setup01_run_20260306_192017.log
3480199  example/110_gentoo/docker_min_hpz6/logs/setup01_run_20260307_012725.log
```

## Steps Taken

### 1. Find large files in history

```bash
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | awk '/^blob/ {print $3, $4}' \
  | sort -rn | head -20
```

### 2. Install git-filter-repo

```bash
uv tool install git-filter-repo
export PATH="/home/kiel/.local/bin:$PATH"
```

### 3. Rewrite history to remove the folder

```bash
git filter-repo --invert-paths --path example/110_gentoo/docker_min_hpz6/logs --force
```

This removes the `origin` remote as a safety measure.

### 4. Prevent future accidents

```bash
echo "example/110_gentoo/docker_min_hpz6/logs/" >> .gitignore
git add .gitignore
git commit -m "chore: ignore docker logs"
```

### 5. Re-add remote and force push

```bash
git remote add origin git@github.com:plops/cl-py-generator.git
git push origin --force --all
git push origin --force --tags
```

### 6. Update other machines

On any other computer that has this repo checked out, the safest approach is a fresh clone:

```bash
cd ..
rm -rf cl-py-generator
git clone git@github.com:plops/cl-py-generator.git
```

Alternatively (if you have local changes to preserve):

```bash
git fetch origin
git reset --hard origin/master
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

## Notes

- `git filter-repo` rewrites all commit hashes from the rewritten point onward.
- Force push is required because the remote history no longer matches.
- All collaborators/clones need to fresh-clone or hard-reset after the force push.
