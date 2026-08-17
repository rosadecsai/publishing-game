"""
Reference implementation (v3.4, first revision for JOI-D-26-00430) for:
  "The Publishing Game: How Editors, Reviewers, and Authors Shape Informetric Dynamics".

Python 3.9+, numpy, scipy, matplotlib.  Base seed = 42.
Replication seeds are derived deterministically from the base seed and the
configuration tuple (replication index, 1-phi, beta_r, k0) via SeedSequence,
so every table and figure of the manuscript is exactly reproducible.

Changes vs. v3.3 (figure generation only; the simulation pipeline and every
number in results.json are byte-identical):
  * figphase1.png now also plots the Type II error series (it absorbs the
    former panel A of figexp1and2.png, removing a duplicated artifact).
  * figexp2.png replaces figexp1and2.png: the Experiment II contract /
    participation panel is now a standalone figure.
  * figphase3.png (Experiment III bar chart) is no longer generated; the
    manuscript reports those quantities in Table 4 only.
  * figsimulation.png no longer hard-codes an in-figure title or the
    equation number in the axis label (both live in the caption).

Run "python final_run_v3_4.py" to recompute everything (writes results.json
and all figures), or "python final_run_v3_4.py --figures-only" to rebuild
the PNG figures from the archived results.json.
"""

import json
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------- baseline parameters (Table 1) -----------------
P = dict(
    N=10_000,
    pi0=0.2, pi1=0.6,             # quality link p(e) = pi0 + pi1 e
    rho_h=0.85, rho_l=0.60,       # editor diagnostic precision
    BJ=10.0, d=5.0, D=3.0,        # payoffs / classification costs
    c=1.0,                        # external review cost
    cu=5.0, cb=1.0,               # mismatch penalties (ratio 5)
    alpha_r=1.0, w_r=0.3,         # reviewer contract
    gA=1.05, gE=1.05, A=1.0,      # author payoff scalars
    lam=1.5,                      # alignment sensitivity
    k0=1.5,                       # reviewer friction (baseline)
    kappa_E=2.0, r=5.0,           # editor logit precision, reputation
    sigma=0.25,                   # quality noise half-width
    seed=42,
)
NSEED = 10                        # MC replications per configuration

ALIGNED = dict(sE=0.90, sA_lo=0.86, sA_hi=0.94)     # |bE-bA| < 0.05
FORCED = dict(sE=0.90, sA_lo=0.60, sA_hi=0.75)      # gap in [0.15,0.30]
MISALIGNED = dict(sE=0.95, sA_lo=0.50, sA_hi=0.55)  # gap > 0.40

QMAX = 1.0                        # Assumption 1 domain


# ------------------- structural variants (Appendix D) -------------------
# struct=None reproduces the baseline expressions verbatim (bit-for-bit).
def _rev_cost(Q, k0, struct=None):
    """Reviewer evaluation disutility."""
    if struct and "eta_r" in struct:
        er = struct["eta_r"]
        return k0 * Q ** (1 + er) / (1 + er)
    return k0 * Q * Q / 2


def _author_effort(omega, bA, bE, h, p, struct=None):
    """Closed-form author best response e* under cost A e^(1+m)/(1+m)."""
    base = omega * p["gA"] * p["gE"] * bA * bE * h / p["A"]
    if struct and "m_a" in struct:
        return np.minimum(1.0, np.maximum(base, 0.0) ** (1.0 / struct["m_a"]))
    return np.minimum(1.0, base)


def _author_cost(e, p, struct=None):
    """Author effort disutility (enters the welfare accounting)."""
    if struct and "m_a" in struct:
        m = struct["m_a"]
        return p["A"] * e ** (1 + m) / (1 + m)
    return p["A"] * e ** 2 / 2


def _quality_link(e, p, struct=None):
    """Pr(X_true = 1 | e)."""
    if struct and struct.get("q_link") == "sqrt":
        return p["pi0"] + p["pi1"] * np.sqrt(e)
    return p["pi0"] + p["pi1"] * e


def reviewer_best_response(q_pop, beta_r, k0, p=P, struct=None):
    def negU(Q):
        C = p["cu"] * np.maximum(Q - q_pop, 0) + \
            p["cb"] * np.maximum(q_pop - Q, 0)
        Y = np.minimum(Q, q_pop)
        return -(p["w_r"] - p["alpha_r"] * C.mean()
                 + beta_r * Y.mean() - _rev_cost(Q, k0, struct))
    Q = float(minimize_scalar(negU, bounds=(0, QMAX), method="bounded").x)
    C = p["cu"] * np.maximum(Q - q_pop, 0) + \
        p["cb"] * np.maximum(q_pop - Q, 0)
    EY = float(np.minimum(Q, q_pop).mean())
    EC = float(C.mean())
    Pi = p["w_r"] - p["alpha_r"] * EC + beta_r * EY - _rev_cost(Q, k0, struct)
    return Q, EC, EY, float(Pi)


def bayes(alpha, rho, X):
    b1 = alpha * rho / (alpha * rho + (1 - alpha) * (1 - rho))
    b0 = alpha * (1 - rho) / (alpha * (1 - rho) + (1 - alpha) * rho)
    return np.where(X == 1, b1, b0)


def run_config(one_minus_phi, beta_r, sE, sA_lo, sA_hi, k0=None, N=None,
               p=P, seed=None, max_iter=400, tol=1e-4, damp=0.4,
               participation=True, desk_reject_only=False,
               review_available=True, struct=None):
    """Solve one configuration.

    participation=True : if the forced-participation fixed point yields
      Pi_r < 0, re-solve with the review channel unavailable (the reviewer
      declines ex ante); the returned dict flags `participates`.
    desk_reject_only=True : the desk pathway can only reject (a = 0);
      desk acceptance is not available.
    review_available=False : review channel removed (used internally).
    """
    k0 = p["k0"] if k0 is None else k0
    N = p["N"] if N is None else N
    rng = np.random.default_rng(p["seed"] if seed is None else seed)
    phi = 1 - one_minus_phi

    sA = rng.uniform(sA_lo, sA_hi, N)
    eps = rng.uniform(-p["sigma"], p["sigma"], N)
    tau_h = rng.random(N) < 0.5
    rho = np.where(tau_h, p["rho_h"], p["rho_l"])
    u_th, u_X, u_t, u_Z = (rng.random(N) for _ in range(4))
    bA = np.maximum(sA, 1 - sA)
    bE = max(sE, 1 - sE)
    h = np.maximum(0.0, 1 - p["lam"] * np.abs(bE - bA))

    def desk_utility(b, X):
        if desk_reject_only:
            return -b * p["D"]                       # a = 0 forced
        return np.where(X == 1, b * p["BJ"] - (1 - b) * p["d"], -b * p["D"])

    omega, Q, mu0, mu1 = 0.9, 0.5, 0.6, 0.4
    for _ in range(max_iter):
        e = _author_effort(omega, bA, bE, h, p, struct)
        q = np.clip(e + eps, 0.0, 1.0 + p["sigma"])
        if review_available:
            Qn, EC, EY, Pi = reviewer_best_response(q, beta_r, k0, p, struct)
        else:
            Qn, EC, EY, Pi = 0.0, 0.0, 0.0, 0.0
        pe = _quality_link(e, p, struct)
        alpha = float(pe.mean())
        theta = (u_th < pe).astype(int)
        X = np.where(u_X < rho, theta, 1 - theta)
        b = bayes(alpha, rho, X)
        rR = (1 + Qn) / 2
        UJ1 = b * (rR * p["BJ"] - (1 - rR) * p["D"]) \
            - (1 - b) * (1 - rR) * p["d"] - p["c"]
        UJ0 = desk_utility(b, X)
        u1 = phi * UJ1 + (1 - phi) * p["r"] * mu1
        u0 = phi * UJ0 + (1 - phi) * p["r"] * mu0
        if review_available:
            P_t0 = 1 / (1 + np.exp(-p["kappa_E"] * (u0 - u1)))
        else:
            P_t0 = np.ones(N)
        if desk_reject_only:
            # desk pathway = rejection of every manuscript routed there
            omega_n = float(1 - P_t0.mean())
        else:
            omega_n = float(1 - (P_t0 * (X == 0)).mean())
        t0 = u_t < P_t0
        m0 = float(tau_h[t0].mean()) if t0.any() else mu0
        m1 = float(tau_h[~t0].mean()) if (~t0).any() else mu1
        delta = max(abs(omega_n - omega), abs(Qn - Q),
                    abs(m0 - mu0), abs(m1 - mu1))
        omega += damp * (omega_n - omega)
        Q += damp * (Qn - Q)
        mu0 += damp * (m0 - mu0)
        mu1 += damp * (m1 - mu1)
        if delta < tol:
            break

    # participation check at the candidate fixed point
    if participation and review_available and Pi < 0:
        out = run_config(one_minus_phi, beta_r, sE, sA_lo, sA_hi, k0=k0,
                         N=N, p=p, seed=seed, max_iter=max_iter, tol=tol,
                         damp=damp, participation=False,
                         desk_reject_only=desk_reject_only,
                         review_available=False, struct=struct)
        out["participates"] = False
        out["Pi_r_forced"] = float(Pi)   # utility if participation forced
        out["Q_forced"] = float(Q)
        return out

    # realized epoch at the fixed point
    e = _author_effort(omega, bA, bE, h, p, struct)
    q = np.clip(e + eps, 0.0, 1.0 + p["sigma"])
    if review_available:
        Q, EC, EY, Pi = reviewer_best_response(q, beta_r, k0, p, struct)
    else:
        Q, EC, EY, Pi = 0.0, 0.0, 0.0, 0.0
    pe = _quality_link(e, p, struct)
    alpha = float(pe.mean())
    theta = (u_th < pe).astype(int)
    X = np.where(u_X < rho, theta, 1 - theta)
    b = bayes(alpha, rho, X)
    rR = (1 + Q) / 2
    UJ1 = b * (rR * p["BJ"] - (1 - rR) * p["D"]) \
        - (1 - b) * (1 - rR) * p["d"] - p["c"]
    UJ0 = desk_utility(b, X)
    u1 = phi * UJ1 + (1 - phi) * p["r"] * mu1
    u0 = phi * UJ0 + (1 - phi) * p["r"] * mu0
    if review_available:
        P_t0 = 1 / (1 + np.exp(-p["kappa_E"] * (u0 - u1)))
    else:
        P_t0 = np.ones(N)
    t0 = u_t < P_t0
    Z = np.where(u_Z < rR, theta, 1 - theta)
    if desk_reject_only:
        published = (~t0) & (Z == 1)
    else:
        published = (t0 & (X == 1)) | (~t0 & (Z == 1))
    C_i = p["cu"] * np.maximum(Q - q, 0) + p["cb"] * np.maximum(q - Q, 0)
    effort_costs = _author_cost(e, p, struct)
    review_costs = np.where(~t0, p["c"], 0.0)
    transfer = np.where(~t0, p["alpha_r"] * C_i, 0.0)
    value = np.where(published & (theta == 1), p["BJ"], 0.0) \
        - np.where(published & (theta == 0), p["d"], 0.0)
    denom = N * alpha * p["BJ"]
    GER = float((value.sum() - effort_costs.sum() - review_costs.sum())
                / denom)
    GER_old = float((value.sum() - effort_costs.sum() - review_costs.sum()
                     - transfer.sum()) / denom)
    good = theta == 1
    if desk_reject_only:
        desk_reject_rate = float(t0.mean())
    else:
        desk_reject_rate = float((t0 & (X == 0)).mean())
    return dict(
        omega=float(omega), Q=float(Q), EC=EC, EY=EY, Pi_r=float(Pi),
        e_mean=float(e.mean()), alpha_pool=alpha,
        desk_rate_h=float(t0[tau_h].mean()),
        desk_rate_pop=float(t0.mean()),
        desk_reject_rate=desk_reject_rate,
        review_share=float((~t0).mean()),
        pub_prob=float(published.mean()),
        typeII=float((~published[good]).mean()) if good.any() else 0.0,
        UJ=float(np.where(t0, UJ0, UJ1).mean()),
        Ysys=float(value.sum() / N), GER=GER, GER_old=GER_old,
        mu0=float(mu0), mu1=float(mu1),
        participates=bool(review_available),
    )


def replicate(one_minus_phi, beta_r, sc, k0=None, nseed=NSEED, N=None,
              **kw):
    """Run a configuration across nseed seeds; return means + s.e."""
    outs = []
    for i in range(nseed):
        seed = int(np.random.SeedSequence(
            [P["seed"], i, int(one_minus_phi * 1000), int(beta_r * 1000),
             int((k0 or P["k0"]) * 100)]).generate_state(1)[0])
        outs.append(run_config(one_minus_phi, beta_r, **sc, k0=k0, N=N,
                               seed=seed, **kw))
    num_keys = [k for k, v in outs[0].items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)]
    agg = {k: float(np.mean([o[k] for o in outs])) for k in num_keys}
    for k in ("GER", "desk_reject_rate", "typeII", "Q", "e_mean", "Pi_r",
              "Ysys", "UJ", "omega", "EC", "EY"):
        v = np.array([o[k] for o in outs], dtype=float)
        agg[k + "_se"] = float(v.std(ddof=1) / np.sqrt(len(v)))
    agg["participation_share"] = float(
        np.mean([o["participates"] for o in outs]))
    if not all(o["participates"] for o in outs):
        pf = [o.get("Pi_r_forced") for o in outs if "Pi_r_forced" in o]
        qf = [o.get("Q_forced") for o in outs if "Q_forced" in o]
        if pf:
            agg["Pi_r_forced"] = float(np.mean(pf))
            agg["Q_forced"] = float(np.mean(qf))
    return agg


import sys

FIGURES_ONLY = "--figures-only" in sys.argv

if FIGURES_ONLY:
    with open("results.json") as f:
        R = json.load(f)
else:
    R = {}

    # Experiment I: signaling sweep (participation equilibrium) -------------
    print("Experiment I ...")
    sweep = {round(w, 2): replicate(round(w, 2), 0.40, ALIGNED)
             for w in np.arange(0.0, 1.001, 0.05)}
    R["exp1"] = sweep
    g0 = sweep[0.0]["GER"]
    for o in sweep.values():
        o["eff_loss"] = max(0.0, (g0 - o["GER"]) / g0)

    # Experiment I-R: desk-reject-only robustness ---------------------------
    print("Experiment I-R (desk-reject only) ...")
    sweepR = {round(w, 2): replicate(round(w, 2), 0.40, ALIGNED,
                                     desk_reject_only=True)
              for w in np.arange(0.0, 1.001, 0.1)}
    R["exp1_dro"] = sweepR
    g0R = sweepR[0.0]["GER"]
    for o in sweepR.values():
        o["eff_loss"] = max(0.0, (g0R - o["GER"]) / g0R)

    # Experiment II: contract sweep, participation + forced counterfactual --
    print("Experiment II ...")
    R["exp2"] = {br: replicate(0.1, br, ALIGNED)
                 for br in (0.0, 0.20, 0.40, 0.60)}
    R["exp2_forced"] = {br: replicate(0.1, br, ALIGNED, participation=False)
                        for br in (0.0, 0.20, 0.40, 0.60)}
    # fine grid for the welfare-optimal reward (participation equilibrium)
    WJ = {}
    for br in np.round(np.arange(0.0, 0.801, 0.05), 2):
        o = replicate(0.1, float(br), ALIGNED, nseed=5)
        WJ[float(br)] = dict(
            WJ=o["Ysys"] - br * o["EY"] * o["review_share"],
            Q=o["Q"], Pi=o["Pi_r"], part=o["participation_share"])
    R["exp2_WJ"] = WJ

    # Experiment III --------------------------------------------------------
    print("Experiment III ...")
    R["exp3"] = {n: replicate(0.1, 0.40, sc) for n, sc in
                 (("aligned", ALIGNED), ("forced", FORCED),
                  ("misaligned", MISALIGNED))}
    omega_h = R["exp3"]["aligned"]["omega"]
    bE_grid = np.linspace(0.5, 1.0, 51)
    curves = {}
    for bA in (0.60, 0.75, 0.90):
        hh = np.maximum(0, 1 - P["lam"] * np.abs(bE_grid - bA))
        curves[str(bA)] = np.minimum(
            1, omega_h * P["gA"] * P["gE"] * bA * bE_grid * hh).tolist()
    R["exp3_curves"] = dict(bE=bE_grid.tolist(), curves=curves, omega=omega_h)

    # Experiment IV ---------------------------------------------------------
    print("Experiment IV ...")
    R["exp4"] = dict(
        status_quo=replicate(0.70, 0.00, MISALIGNED),
        partial=replicate(0.70, 0.40, MISALIGNED),
        optimal=replicate(0.10, 0.40, ALIGNED))
    R["exp4_dro"] = dict(
        status_quo=replicate(0.70, 0.00, MISALIGNED, desk_reject_only=True),
        partial=replicate(0.70, 0.40, MISALIGNED, desk_reject_only=True),
        optimal=replicate(0.10, 0.40, ALIGNED, desk_reject_only=True))

    # Experiment V ----------------------------------------------------------
    print("Experiment V ...")
    R["exp5"] = {str(k0): replicate(0.1, 0.40, ALIGNED, k0=k0)
                 for k0 in (1.5, 3.0, 5.0)}
    # Experiment V (Panel B): forced-participation counterfactual ----------
    R["exp5_forced"] = {str(k0): replicate(0.1, 0.40, ALIGNED, k0=k0,
                                           participation=False)
                        for k0 in (1.5, 3.0, 5.0)}

    # Sensitivity S1: tipping point vs (kappa_E, r) -------------------------
    print("Sensitivity S1 ...")


    def tipping_point(kE, r_val):
        p2 = dict(P, kappa_E=kE, r=r_val)
        prev = None
        for w in np.arange(0.0, 1.001, 0.05):
            o = run_config(round(w, 2), 0.40, **ALIGNED, p=p2, N=4000)
            if prev is not None and o["desk_reject_rate"] > 0.20 \
                    and prev <= 0.20:
                return round(w, 2)
            prev = o["desk_reject_rate"]
        return None


    S1 = {}
    for kE in (1.0, 2.0, 4.0):
        for r_val in (3.0, 5.0, 7.0):
            S1[f"kE{kE}_r{r_val}"] = tipping_point(kE, r_val)
    R["sens_tipping"] = S1

    # Sensitivity S2: minimal participation-restoring reward ----------------
    print("Sensitivity S2 ...")


    def beta_star(k0=None, w_r=None):
        p2 = dict(P)
        if w_r is not None:
            p2["w_r"] = w_r
        for br in np.round(np.arange(0.0, 1.501, 0.05), 2):
            o = run_config(0.1, float(br), **ALIGNED, k0=k0, p=p2, N=4000,
                           participation=False)
            if o["Pi_r"] >= 0:
                return float(br)
        return None


    R["sens_bstar_k0"] = {str(k0): beta_star(k0=k0)
                          for k0 in (1.0, 1.5, 2.0, 3.0, 5.0)}
    R["sens_bstar_wr"] = {str(wr): beta_star(w_r=wr)
                          for wr in (0.0, 0.15, 0.3, 0.45, 0.6)}

    # Appendix D: structural robustness ---------------------------------
    # For each structural variant we first re-derive the variant's own
    # IR-constrained optimum beta_r* on the 0.05 grid (as Section 4.4
    # does for the baseline technology), and then re-run the headline
    # diagnostics at that contract.  Seeds follow the same derivation
    # rule as everywhere else, keyed by (one_minus_phi, beta_r, k0).
    print("Appendix D (structural robustness) ...")
    VARIANTS = {
        "baseline": None,
        "cubic_reviewer": dict(eta_r=2),      # reviewer cost k0 Q^3 / 3
        "concave_link": dict(q_link="sqrt"),  # p(e) = pi0 + pi1 sqrt(e)
        "convex_author": dict(m_a=2),         # author cost A e^3 / 3
    }
    SD = {}
    for name, struct in VARIANTS.items():
        kw = {} if struct is None else dict(struct=struct)
        d = {}
        # (ii) participation collapse and restoration (Exp. II):
        # minimal restoring contract on the 0.05 grid (forced-utility scan)
        bstar = None
        for br in np.round(np.arange(0.0, 1.501, 0.05), 2):
            o = run_config(0.1, float(br), **ALIGNED, N=4000,
                           participation=False, **kw)
            if o["Pi_r"] >= 0:
                bstar = float(br)
                break
        d["beta_star"] = bstar
        d["expII"] = {str(br): replicate(0.1, br, ALIGNED, **kw)
                      for br in (0.0, bstar)}
        # (i) hyper-criticism hump (Exp. I) at the variant's own optimum:
        # low / peak / corner signaling weights
        d["expI"] = {str(w): replicate(w, bstar, ALIGNED, **kw)
                     for w in (0.1, 0.65, 1.0)}
        # (iii) fatigue cliff (Exp. V): smallest friction breaching
        # participation at the variant's contract, then the eq-vs-forced gap
        k0_cliff = None
        for k0 in (2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0):
            o = run_config(0.1, bstar, **ALIGNED, k0=k0, N=4000,
                           participation=False, **kw)
            if o["Pi_r"] < 0:
                k0_cliff = k0
                break
        d["k0_cliff"] = k0_cliff
        d["expV_base"] = replicate(0.1, bstar, ALIGNED, k0=1.5, **kw)
        if k0_cliff is not None:
            d["expV_eq"] = replicate(0.1, bstar, ALIGNED, k0=k0_cliff, **kw)
            d["expV_forced"] = replicate(0.1, bstar, ALIGNED, k0=k0_cliff,
                                         participation=False, **kw)
        SD[name] = d
    R["structD"] = SD

    with open("results.json", "w") as f:
        json.dump(R, f, indent=1)
    print("results.json written")


# ======================================================================
#  Figure generation (reads R; run "python final_run_v3_3.py --figures-only"
#  to rebuild all PNG figures from the archived results.json)
# ======================================================================

def _healthy_pool_q(p=P):
    """Reconstruct the equilibrium quality pool of the healthy baseline
    (Experiment II, beta_r = 0.40, first replication seed) for Figure 3."""
    seed = int(np.random.SeedSequence(
        [p["seed"], 0, 100, 400, 150]).generate_state(1)[0])
    out = run_config(0.1, 0.40, **ALIGNED, seed=seed)
    rng = np.random.default_rng(seed)
    sA = rng.uniform(ALIGNED["sA_lo"], ALIGNED["sA_hi"], p["N"])
    eps = rng.uniform(-p["sigma"], p["sigma"], p["N"])
    bA = np.maximum(sA, 1 - sA)
    bE = max(ALIGNED["sE"], 1 - ALIGNED["sE"])
    h = np.maximum(0.0, 1 - p["lam"] * np.abs(bE - bA))
    e = np.minimum(1.0, out["omega"] * p["gA"] * p["gE"] * bA * bE * h
                   / p["A"])
    return np.clip(e + eps, 0.0, 1.0 + p["sigma"])


def make_figures(R, p=P):
    ks = sorted(R["exp1"], key=float)
    xs = [float(k) for k in ks]

    # ---- Figure 2 (figphase1.png): Experiment I sweep -----------------
    fig, ax = plt.subplots(figsize=(8.5, 5.3), dpi=150)
    ax.plot(xs, [R["exp1"][k]["desk_rate_h"] * 100 for k in ks],
            "o-", color="tab:blue", label="Desk decision rate, $h$-type (%)")
    ax.plot(xs, [R["exp1"][k]["desk_reject_rate"] * 100 for k in ks],
            "s--", color="tab:red",
            label="Desk-rejection rate, population (%)")
    ax.plot(xs, [R["exp1"][k]["typeII"] * 100 for k in ks],
            "^-", color="tab:purple", label="Type II error rate (%)")
    ax.plot(xs, [R["exp1"][k]["eff_loss"] * 100 for k in ks],
            "D-", color="black",
            label="Efficiency loss, $\\Delta$GER vs. $(1-\\phi)=0$ (%)")
    ax.set_xlabel("Reputational signaling weight $(1-\\phi)$")
    ax.set_ylabel("Percent")
    ax.grid(ls="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig("figphase1.png")
    plt.close(fig)

    # ---- Figure 3 (figphase2.png): reviewer utility profiles ----------
    q = _healthy_pool_q(p)
    Qg = np.linspace(0, 1, 201)

    def util(Q, br):
        C = p["cu"] * np.maximum(Q - q, 0) + p["cb"] * np.maximum(q - Q, 0)
        return (p["w_r"] - p["alpha_r"] * C.mean()
                + br * np.minimum(Q, q).mean() - p["k0"] * Q * Q / 2)

    fig, ax = plt.subplots(figsize=(8.5, 5.3), dpi=150)
    for br, col in ((0.0, "tab:red"), (0.40, "tab:blue")):
        U = np.array([util(Q, br) for Q in Qg])
        Qopt = float(minimize_scalar(lambda Q: -util(Q, br),
                                     bounds=(0, 1), method="bounded").x)
        Uopt = util(Qopt, br)
        note = ("invitation declined" if Uopt < 0 else "")
        lab = ("$\\beta_r=%.2f$  ($Q^*=%.2f$, $\\Pi_r=%+.2f$%s)"
               % (br, Qopt, Uopt, (": " + note) if note else ""))
        ax.plot(Qg, U, color=col, lw=2, label=lab)
        ax.plot(Qopt, Uopt, "o", color=col, ms=7)
    ax.axhline(0, color="gray", lw=1.2)
    ax.text(0.98, 0.01, "participation threshold", color="gray",
            ha="right", va="bottom", transform=ax.get_yaxis_transform())
    ax.set_xlabel("Reviewer evaluative effort $Q$")
    ax.set_ylabel("Expected reviewer utility $E[\\Pi_r]$")
    ax.grid(ls="--", alpha=0.4)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig("figphase2.png")
    plt.close(fig)


    # ---- Figure (figexp2.png): Experiment II contract & participation --
    brs = sorted(R["exp2"], key=float)
    xb = [float(k) for k in brs]
    Qs = [R["exp2"][k]["Q"] for k in brs]
    Pi = [R["exp2"][k].get("Pi_r_forced", R["exp2"][k]["Pi_r"])
          for k in brs]
    # participation boundary: smallest plotted contract restoring
    # participation (the 0.05-grid threshold beta_r* = 0.40)
    part = [R["exp2"][k].get("participation_share", 1.0) for k in brs]
    cross = next(x for x, pf in zip(xb, part) if pf >= 1.0)
    fig, b = plt.subplots(figsize=(8.5, 5.3), dpi=150)
    b.axvspan(min(xb) - 0.03, cross, color="red", alpha=0.08)
    b.text((min(xb) + cross) / 2, max(Qs) * 0.95,
           "no participation:\nreview channel\ncollapses",
           color="tab:red", ha="center", va="top", fontsize=9)
    b.plot(xb, Qs, "o-", color="tab:blue",
           label="Equilibrium reviewer effort $Q^*$")
    b.plot(xb, Pi, "s--", color="tab:green",
           label="$\\Pi_r$ if participation were forced")
    b.axhline(0, color="gray", lw=1.2)
    b.set_xlim(min(xb) - 0.03, max(xb) + 0.03)
    b.set_xlabel("Quality reward $\\beta_r$")
    b.set_ylabel("Effort / utility")
    b.grid(ls="--", alpha=0.4)
    b.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig("figexp2.png")
    plt.close(fig)

    # ---- Figure 6 (figsimulation.png): alignment catalyst -------------
    cv = R["exp3_curves"]
    fig, ax = plt.subplots(figsize=(8.5, 5.3), dpi=150)
    for bA, col in (("0.6", "tab:red"), ("0.75", "tab:orange"),
                    ("0.9", "tab:blue")):
        ax.plot(cv["bE"], cv["curves"][bA], color=col, lw=2,
                label="$b_A = %s$" % bA)
    ax.set_xlabel("Editorial strictness $b_E$")
    ax.set_ylabel("Optimal author effort $e^*$")
    ax.grid(ls="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig("figsimulation.png")
    plt.close(fig)

    # ---- Figure 7 (figphase4_integration.png): radar + GER bars -------
    names = ["status_quo", "partial", "optimal"]
    disp = ["Status Quo", "Partial Reform", "Optimal Synergies"]
    cols = ["tab:red", "tab:orange", "tab:green"]
    axes_lbl = ["$e^*$", "$Q^*$", "$\\omega$", "Pub. rate", "GER"]
    keys = ["e_mean", "Q", "omega", "pub_prob", "GER"]
    ang = np.linspace(0, 2 * np.pi, len(keys), endpoint=False) + np.pi / 2
    fig = plt.figure(figsize=(12.8, 5.6), dpi=150)
    ar = fig.add_subplot(121, polar=True)
    for n, d, c in zip(names, disp, cols):
        vals = [R["exp4"][n][k] for k in keys]
        ar.plot(np.r_[ang, ang[0]], np.r_[vals, vals[0]], color=c, lw=2,
                label=d)
        ar.fill(np.r_[ang, ang[0]], np.r_[vals, vals[0]], color=c,
                alpha=0.15)
    ar.set_thetagrids(np.degrees(ang) % 360, axes_lbl)
    ar.set_rlim(0, 1)
    ar.set_rlabel_position(22.5)
    ar.set_title("(A) Macro-state performance profile", pad=22)
    ar.legend(loc="lower left", bbox_to_anchor=(-0.12, -0.12), fontsize=9)
    ab = fig.add_subplot(122)
    g = [R["exp4"][n]["GER"] for n in names]
    se = [R["exp4"][n]["GER_se"] for n in names]
    bars = ab.bar(disp, g, color=cols, yerr=se, capsize=4)
    for r, v in zip(bars, g):
        ab.text(r.get_x() + r.get_width() / 2, v + 0.012, "%.3f" % v,
                ha="center", fontsize=10)
    ab.set_ylabel("Global Efficiency Ratio (GER)")
    ab.set_title("(B) GER by macro-state (mean $\\pm$ MC s.e.)")
    fig.subplots_adjust(left=0.05, right=0.97, top=0.86, bottom=0.10,
                        wspace=0.30)
    fig.savefig("figphase4_integration.png")
    plt.close(fig)

    # ---- Figure B.9 (figsens.png): participation threshold ------------
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.0, 4.6), dpi=150)
    k0s = sorted(R["sens_bstar_k0"], key=float)
    a.plot([float(k) for k in k0s],
           [R["sens_bstar_k0"][k] for k in k0s], "o-", color="tab:blue")
    a.axhline(0.40, color="gray", ls="--", lw=1.2)
    a.text(3.0, 0.42, "contract of Exp. II ($\\beta_r = 0.40$)",
           color="gray", fontsize=9)
    a.set_title("(A) $\\beta_r^*$ rises with fatigue")
    a.set_xlabel("Reviewer evaluation friction $k_0$")
    a.set_ylabel("Minimal participation-restoring reward $\\beta_r^*$")
    a.grid(ls="--", alpha=0.4)
    wrs = sorted(R["sens_bstar_wr"], key=float)
    b.plot([float(k) for k in wrs],
           [R["sens_bstar_wr"][k] for k in wrs], "s-", color="tab:green")
    b.set_title("(B) $\\beta_r^*$ falls with the fixed reward")
    b.set_xlabel("Fixed social reward $w_r$")
    b.set_ylabel("$\\beta_r^*$")
    b.grid(ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig("figsens.png")
    plt.close(fig)
    print("figures written")


make_figures(R)
