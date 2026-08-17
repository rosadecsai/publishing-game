#The Publishing Game: How Editors, Reviewers, and Authors Shape Informetric Dynamics

The Publishing Game — reference implementation
Simulation code and archived numerical output for:
> J. A. García, R. Rodriguez-Sánchez, J. Fdez-Valdivia,
> **"The Publishing Game: How Editors, Reviewers, and Authors Shape
> Informetric Dynamics"**, *Journal of Informetrics* (under review,
> manuscript JOI-D-26-00430, first revision).
The model is a tripartite sequential game (editor–reviewer–author) whose
equilibrium is computed as a fixed point of the agents' coupled best
responses by damped best-response iteration, and evaluated by Monte
Carlo simulation.
Contents
File	Description
`final_run_v3_4.py`	Complete reference implementation (model, fixed-point solver, all experiments, sensitivity and robustness appendices, figure generation).
`results.json`	Archived numerical output of the full pipeline (all experiments, appendices, and Monte Carlo standard errors).
`requirements.txt`	Python dependencies.
Requirements
Python 3.9+ with `numpy`, `scipy`, and `matplotlib`
(`pip install -r requirements.txt`).
How to run
Full recomputation (recreates `results.json` and every PNG figure from
scratch; this runs all experiments and appendices and takes a while):
```bash
python final_run_v3_4.py
```
Figures only (rebuilds every PNG figure of the paper byte-for-byte from
the archived `results.json`, in seconds):
```bash
python final_run_v3_4.py --figures-only
```
Reproducibility
Base seed: 42. Every reported statistic is the mean over ten
independent replications per configuration, each with
N = 10,000 manuscripts.
Replication seeds are derived deterministically from the base seed
and the configuration tuple — replication index, reputational weight
(1−φ), quality reward β_r, and friction k₀ — via NumPy's
`SeedSequence`, so every table and figure of the manuscript
(including the appendices) is exactly reproducible without further
input.
Monte Carlo standard errors are computed across replications and
stored in `results.json` alongside every headline quantity
(`*_se` keys).
Mapping of outputs to the manuscript
`results.json` key	Manuscript artifact
`exp1`	Table 4; Figure 2 (`figphase1.png`)
`exp2`, `exp2_forced`	Table 5; Figure 3 (`figphase2.png`); Figure 4 (`figexp2.png`)
`exp2_WJ`	Welfare-optimal contract paragraph (Sec. 4.4)
`exp3`, `exp3_curves`	Table 6; Figure 5 (`figsimulation.png`)
`exp4`	Table 7; Figure 6 (`figphase4_integration.png`)
`exp5`, `exp5_forced`	Table 8; Figure 7, Panel A (pgfplots in TeX)
`sens_tipping`	Table B.9
`sens_bstar_k0`, `sens_bstar_wr`	Figure B.8 (`figsens.png`)
`exp1_dro`, `exp4_dro`	Table C.10 (desk-reject-only institution)
`structD`	Table D.11 (structural perturbations)
Figure 1 and both panels of Figure 7 are drawn directly in the LaTeX
source (TikZ/pgfplots) and have no PNG counterpart.
License
<!-- TODO: choose a license (e.g., MIT) before publishing the repository. -->
To be specified by the authors.
Citation
If you use this code, please cite the paper above. A citable archive of
this repository is deposited on Zenodo (DOI: 10.5281/zenodo.XXXXXXX
<!-- TODO: replace with the minted DOI -->).# The Publishing Game: How Editors, Reviewers, and Authors Shape Informetric Dynamics
