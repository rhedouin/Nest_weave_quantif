"""Build the standalone top-view kit for collaborators without the repo: kit/top_views_kit.zip.

    python kit/build_top_kit.py

Layout of kit/ (gitignored: the kit is sent by hand, never pushed):
    build_top_kit.py     this script
    guide/               source of GUIDE.pdf (guide.tex, make_guide_fig.py)
    top_views_kit/       the folder Marie gets (README, GUIDE.pdf, top_report.py, data/)
    top_views_kit.zip    zip of that folder (unzips to top_views_kit/)
    old_results/         earlier runs, not shipped

Refreshes top_views_kit/data from the repo (per-photo metrics v7f, blind-validation points with
prediction and answer, half-size example overlays), rebuilds the guide figure and PDF, and zips.
"""
import json
import subprocess
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "kit/top_views_kit"
GUIDE = ROOT / "kit/guide"
sys.dont_write_bytecode = True        # no __pycache__ in the zip
sys.path.insert(0, str(KIT))
import top_report  # noqa: E402


def main():
    data = KIT / "data"
    (data / "overlays").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "outputs/top/metrics_v7f.csv", data / "metrics_top.csv")
    v = pd.read_csv(ROOT / "validation_points_top/points.csv").merge(
        pd.read_csv(ROOT / "validation_points_top/predictions.csv"), on="id")
    ans = json.loads((ROOT / "annotations/validations_top_RH_20261126.json").read_text())["answers"]
    v["truth"] = v.id.map(ans)
    v[v.truth.notna()].to_csv(data / "validation_points.csv", index=False)
    for s in top_report.EXAMPLES + top_report.MIX_EXAMPLES:
        im = Image.open(ROOT / f"outputs/top/seg_v7f/{s}_overlay.jpg")
        im.resize((im.width // 2, im.height // 2), Image.LANCZOS).save(data / "overlays" / f"{s}_overlay.jpg",
                                                                        quality=88)
    subprocess.run([sys.executable, "-B", GUIDE / "make_guide_fig.py"], check=True)
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-quiet", "guide.tex"], cwd=GUIDE, check=True,
                   stdout=subprocess.DEVNULL)
    shutil.copy(GUIDE / "guide.pdf", KIT / "GUIDE.pdf")
    for junk in [*KIT.glob("results*"), KIT / "__pycache__"]:
        if junk.exists():
            print("not zipped, please move:", junk) if junk.name != "__pycache__" else shutil.rmtree(junk)
    z = shutil.make_archive(str(ROOT / "kit/top_views_kit"), "zip", ROOT / "kit", "top_views_kit")
    print("->", z)


if __name__ == "__main__":
    main()
