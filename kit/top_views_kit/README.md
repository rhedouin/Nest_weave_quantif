# Top-view kit: figures and tables for any subset of nests

This kit recomputes every figure and table of the top-view report on the photos you choose, statistics included. For example, some cohorts only, one round, without the last photo of each nest, or without some pairs. You need Python, not the project repository.

The segmentation of the photos is already done (model v7f, validated at 92 % accuracy on blind points). The kit starts from its result: a table of measurements, one row per top-view photo.

## What is in this folder

```
top_views_kit/
├── README.md          this file
├── top_report.py      the script: one function, make_report(), also usable from the command line
├── guide/
│   └── guide.pdf      how to read the statistics (mixed models, variance shares, time effect)
└── data/
    ├── metrics_top.csv          one row per top photo (173 photos, 49 nests, 25 pairs): colour shares,
    │                            mixing indices, centrality, ... computed from the segmentation
    └── validation_points.csv    blind validation: point, model prediction, human answer
```

**Minimum to run:** `top_report.py` and `data/metrics_top.csv`. The script does not read the photos themselves (the `05_Top` folder): it starts from the measurements already extracted from them. `validation_points.csv` is optional: without it, the validation tables are skipped.

The example-photo figure of the report needs the segmentation class maps, which are not in the kit: it is skipped, and the mixing figure shows the curves without the photos.

Each run writes a new results folder (default `results/`, or the name given with `--out`):

```
results/
├── index.html        everything on one page: options used, all tables, all figures
├── figs/             the 6 figures (PNG)
└── tables/           the tables (CSV, they open in Excel) + photos_used.csv (exact rows used)
```

## Install (once)

Python 3.9 or later with five standard packages:

    pip install numpy pandas matplotlib scipy statsmodels

or, with conda: `conda install numpy pandas matplotlib scipy statsmodels`.

## Run

Open a terminal in this folder, then:

    python top_report.py                                   # all photos -> results/
    python top_report.py --max-session 3 --out no_photo4   # photos 1-3 of each nest only
    python top_report.py --drop-sessions 4 --out no_photo4_only
    python top_report.py --time session --out time_by_photo
    python top_report.py --cohorts 2 3 --out cohorts23
    python top_report.py --rounds 1 --out round1
    python top_report.py --exclude-pairs 110 205 --out without_110_205
    python top_report.py --exclude-photos 205_top_20260403 --out test
    python top_report.py --help

Options can be combined, for example `--cohorts 2 3 --max-session 3`. Open `<out>/index.html` in a browser.

| option | meaning |
|---|---|
| `--cohorts 2 3` | keep these cohorts |
| `--rounds 1` | keep these rounds (1 = first nest of the pair, 2 = second) |
| `--max-session 3` | keep photos 1 to 3 of each nest ("session" = photo number within the nest) |
| `--drop-sessions 4` | remove these photo numbers only |
| `--pairs` / `--exclude-pairs` | keep only / remove these pairs |
| `--exclude-photos` | remove photos by name (e.g. `205_top_20260403`) |
| `--time days` / `--time session` | time in the models: straight line in days (default, as in the report), or one effect per photo number (see guide/guide.pdf) |
| `--n-boot 200` | fewer bootstrap resamples: faster, less precise repeatability intervals |
| `--out name` | results folder |

From Python or a notebook:

    from top_report import make_report
    tables = make_report(cohorts=[2, 3], max_session=3, time="session", out="test")
    tables["models_coefficients"]          # every table is a pandas DataFrame

A full run takes about a minute, mostly the bootstrap.

## The figures and tables

| file | content |
|---|---|
| `examples.png` | photos and their class maps (only if the class maps are in `data/overlays/`) |
| `mixing_example.png` | mixing index and cell entropy against scale: two example photos and the spread of the selection |
| `composition.png` | share of each colour in the visible string, per nest and photo |
| `trends.png` | string area, mixing at 12 mm and order-centre correlation across photos |
| `centrality.png` | where each colour sits (0 = edge, 1 = centre) |
| `pair_signature.png` | 1st vs 2nd nest of each pair |
| `groups.csv` | photos, nests and pairs per cohort and round |
| `key_numbers.csv` | medians and other summary values |
| `models_coefficients.csv` | mixed models: effect of cohort, round and time (how to read: guide/guide.pdf) |
| `models_variance.csv` | share of variance between pairs, between nests of a pair, between photos of a nest |
| `round_scales.csv` | round effect at every scale of both mixing indices |
| `repeatability.csv` | is a pair similar across its two nests? |
| `validation_*.csv` | blind validation of the segmentation, restricted to the selected photos |

## Good to know

- **Mixing indices.** `mix_*` and `cellH_*` exist only when the 2nd colour makes at least 10 % of the visible string. Very uniform photos are therefore absent from those models.
- **Variables that no longer vary.** When a subset removes a variable (`--rounds 1`, one cohort), its row disappears from the models.
  - With a single round, each pair has one nest, so the pair share means pair + nest.
- **Repeatability and the pair-signature figure** need both rounds and at least 3 pairs with both nests.
- **When something cannot be estimated** (too few pairs, for example), it is listed in yellow at the top of `index.html` and the script carries on.
