# Phase-Controlled Bistability and Oscillation Suppression in the Morris–Lecar Model under High-Frequency Biharmonic Forcing

Manuscript prepared in the IJBC (World Scientific) LaTeX style.

* `paper/main.tex` — manuscript source (class `ws-ijbc`, bibliography style `ws-ijbc`), `paper/refs.bib`, `paper/numbers.tex` (auto-generated numerical values).
  Compile: `pdflatex main; bibtex main; pdflatex main; pdflatex main` (figures are read from `../figs`).
* `figs/` — all figures (PDF, produced by `code/make_figs.py`).
* `code/` — reproducible Python code (numpy, scipy, numba, matplotlib):
  * `ml_core.py` — model, exact transformation, averaging, Poincaré map, Floquet/stroboscopic map, Lyapunov coefficient.
  * `run_avg.py`, `run_avg2.py` — averaged system: Hopf curve, l1, Bautin points, fold of cycles, one-parameter diagrams, harmonic-ratio and parameter sensitivity.
  * `run_zoom.py` — fine one-parameter scans around the bistable band.
  * `run_full.py` — full forced system: `floquet`, `fold`, `basins`, `switch`, `sweep` stages (`python3 run_full.py <stage>`).
  * `run_basins_zoom.py` — zoomed basins.
  * `make_figs.py`, `make_numbers.py` — figures and LaTeX macros with numerical values.
* `data/` — computed results (`.npz`, `.json`) used by the figures and tables.
