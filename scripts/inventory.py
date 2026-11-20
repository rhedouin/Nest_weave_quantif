"""S0.2 — inventory of the photo folders: name fields, EXIF, duplicates, overlap with old/.

    python scripts/inventory.py 05_Top 02_Left 03_Right --out outputs/inventory.csv

File names are <code>_<view>_<YYYYMMDD>[ (2)].JPG. From cohort 1 round 2 on, <code>
is the pair number (cohort = first digit). In cohort 1 round 1 (Oct.-Nov. 2025) it
is a random 3-digit photo code that can collide with a pair number (207, 104):
metadata/c1r1_photocodes.csv maps (code, day) to the nest. A round is a run of
sessions of the same pair with gaps of at most ROUND_GAP_DAYS days; nest_id is
MIA_<pair><round:02d>, as in Marie's table.
"""
import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd
from PIL import ExifTags, Image

ROUND_GAP_DAYS = 45  # 304: 02/05 -> 13/06 is one round (42 d), 13/06 -> 02/08 is not (50 d)
NAME = re.compile(r"(\d+)_([a-z]+)_(\d{8})(?: \((\d+)\))?$")


def exif(path):
    im = Image.open(path)
    e = im.getexif()
    sub = {ExifTags.TAGS.get(k, k): v for k, v in e.get_ifd(0x8769).items()}
    return {"width": im.size[0], "height": im.size[1], "orientation": e.get(0x0112, 1),
            "exposure": float(sub.get("ExposureTime", "nan")), "fnumber": float(sub.get("FNumber", "nan")),
            "focal_mm": float(sub.get("FocalLength", "nan")), "iso": sub.get("ISOSpeedRatings")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("folders", nargs="+")
    p.add_argument("--old", default="old")
    p.add_argument("--out", default="outputs/inventory.csv")
    p.add_argument("--c1r1", default="metadata/c1r1_photocodes.csv")
    args = p.parse_args()
    md5 = lambda f: hashlib.md5(f.read_bytes()).hexdigest()  # noqa: E731
    old = {md5(f): str(f) for f in Path(args.old).rglob("*.JPG")}
    rows = []
    for d in args.folders:
        for f in sorted(Path(d).glob("*.JPG")):
            m = NAME.match(f.stem)
            rows.append({"path": str(f), "nest": m and m.group(1), "view": m and m.group(2),
                         "date": m and m.group(3), "take": int(m.group(4)) if m and m.group(4) else 1,
                         **exif(f), "md5": md5(f)})
    df = pd.DataFrame(rows)
    df["same_file_as"] = df.groupby("md5").path.transform(lambda s: ";".join(s) if len(s) > 1 else "")
    df["in_old"] = df.md5.map(old).fillna("")
    # a "(2)" take is the entrance view of a non-dome nest, shot along another view's axis:
    # entrance views are excluded from the analyses, so keep the plain take only
    df["entrance_dup"] = df["take"] > 1
    # cohort 1 round 1: photo code + day -> nest
    c1 = pd.read_csv(args.c1r1, dtype=str)
    code_day = {}
    for r in c1.itertuples():
        for code in (r.photocode, r.photocode2):
            for day in (r.photoday1, r.photoday2, r.photoday3, r.photoday4):
                if isinstance(code, str) and isinstance(day, str):
                    code_day[(code, day)] = r.nest_id
    def c1_nest(code, day):
        # one left view of 349 is dated the day after its session (23/10 vs 22/10): allow ±1 day
        d = pd.Timestamp(day)
        for k in (0, -1, 1):
            nid = code_day.get((code, (d + pd.Timedelta(days=k)).strftime("%Y%m%d")))
            if nid:
                return nid, (d + pd.Timedelta(days=k)).strftime("%Y%m%d")
        return "", day
    found = [c1_nest(n, d) for n, d in zip(df.nest, df.date)]
    df["nest_id"] = [f[0] for f in found]
    df["date_file"] = df.date
    df["date"] = [f[1] for f in found]
    df["pair"] = [nid[4:7] if nid else n for nid, n in zip(df.nest_id, df.nest)]
    unknown = df[(df.date < "20251201") & (df.nest_id == "")]
    if len(unknown):
        print("ATTENTION codes photo de C1R1 absents de la table :", sorted(set(unknown.path)))
    # the same file filed under two codes of one nest (104 / 207): keep one copy
    df["dup_copy"] = df.duplicated("md5", keep="first")
    df["date"] = pd.to_datetime(df.date)
    df = df.sort_values(["pair", "date", "view", "take"])
    sess = df.drop_duplicates(["pair", "date"])[["pair", "date"]].copy()
    sess["round"] = sess.groupby("pair").date.transform(
        lambda d: (d.diff().dt.days > ROUND_GAP_DAYS).cumsum() + 1)
    sess["session"] = sess.groupby(["pair", "round"]).cumcount() + 1
    df = df.merge(sess, on=["pair", "date"])
    df["cohort"] = df.pair.str[0].astype(int)
    df["nest_id"] = [f"MIA_{p}{r:02d}" for p, r in zip(df.pair, df["round"])]
    df["use"] = ~df.entrance_dup & ~df.dup_copy
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    u = df[df.use]
    print(f"{len(df)} photos ({len(u)} utilisables), {u.pair.nunique()} paires, {u.nest_id.nunique()} nids, "
          f"{u.groupby(['pair', 'date']).ngroups} séances -> {args.out}")
    print("fichiers identiques sous deux noms :", (df.same_file_as != "").sum())
    print("rotation EXIF ≠ 1 :", (df.orientation != 1).sum(), "· prises multiples :", (df["take"] > 1).sum(),
          "· déjà dans old/ :", (df.in_old != "").sum())


if __name__ == "__main__":
    main()
