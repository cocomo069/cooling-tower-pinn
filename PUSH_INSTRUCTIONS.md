# Pushing this project to GitHub

I could not drive git from my sandbox because the mounted folder blocks git's
lock files. Everything is ready to push from your own machine, where git works
normally. Run these in the project folder.

There is a half-initialised `.git/` folder I left behind. Delete it first so you
start clean.

## What gets published, and what does NOT

The `.gitignore` already excludes the private and heavy items:

- `outreach.md` (your email and LinkedIn drafts) stays private
- `cfd/cases/`, `cfd/mesh/`, `data/` (large binaries) are excluded, regenerate with the scripts
- training checkpoints and the eval `.npz` are excluded
- the trained model `pinn/pinn_params.npz` IS included so the repo is usable

Public files: the four scripts, the README, the LICENSE, all figures, and the report.

## Commands (Windows PowerShell / Git Bash)

```bash
cd "D:\paid tasks\chiller_airflow_modeling_godela\cooling_tower_plume_pinn"

# 1. start clean
rm -rf .git            # PowerShell: Remove-Item -Recurse -Force .git

# 2. set your identity (once, if not already set globally)
git init
git add README.md LICENSE .gitignore \
        cfd/mesh_gen.py cfd/extract_field.py \
        pinn/pinn_surrogate.py pinn/analyze.py pinn/pinn_params.npz \
        figures/*.png figures/recirculation_table.csv figures/recirculation_table.json \
        report/*
git commit -m "Cooling-tower array CFD sweep and physics-informed surrogate"
```

## Then create the remote and push, either way

**With GitHub CLI (if you have `gh`):**

```bash
gh repo create cooling-tower-pinn --public --source=. --remote=origin --push
```

**Without gh (create an empty repo on github.com first, then):**

```bash
git branch -M main
git remote add origin https://github.com/<your-username>/cooling-tower-pinn.git
git push -u origin main
```

## Before you push, double check

- Open `.gitignore` and confirm `outreach.md` is listed (it is).
- Run `git status` and make sure no `case_*.h5`, `*.msh`, `*_field.npz`, or
  `outreach.md` is staged.
- Update the copyright line in `LICENSE` and the `[Your name]` placeholders.
