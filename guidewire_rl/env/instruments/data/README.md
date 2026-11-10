# Instrument data

## `simmons.csv` — traced Simmons 2 catheter

| | |
| --- | --- |
| What | Centreline of a photographed Cordis 5F Simmons 2 (Sidewinder II) catheter, traced point by point; the catheter as Qiu 2024 describe it (the paper itself was not re-read when this file was vendored) |
| Source | Qiu M. 2024, *Energy barriers govern catheter herniation during endovascular procedures: a 2.5D vascular flow model analysis*, Zenodo record 11203294, doi:10.5281/zenodo.11203294 (published 2024-05-16) |
| Licence | CC BY 4.0 (as the Zenodo record states) |
| Where in the record | `2.5D Aortic Arch.zip` (57 732 701 bytes, md5 `cae0029878e4213b772017766b2377d9`, checked on download 2026-10-30) → `2.5D Aortic Arch/Herniation/simmons.csv` |
| This file | unchanged copy; md5 `6978de93499064bc16f3c3eb75378b62`, sha256 `1f930d2c5657c65ac22b2439cde737e24f01ece4e5096a30c415d06a25d10e80`; 3328 points |

Columns used: `X-Coordinate (um)`, `Y-Coordinate (um)` (image coordinates; divided by 1000 they are mm, as
the record's own `getsimmons.m` does). Points are about 0.05 mm apart; the whole trace is 165.8 mm long.
The first row is a stretch of straight shaft, the last row the catheter tip. Columns `x`, `y` in rows 2–4
are the record's landmarks `Tip`, `Simmons A`, `Simmons B` (also image mm); `Tip` is 0.4 mm from the last
row. The `Curve Length (um)` column (16 593) does not match the coordinates (165.8 mm); it is not used.

Used by `../simmons2.py` (the rebuilt head and how) and `tests/test_instruments.py` (md5, shape against
this trace).
