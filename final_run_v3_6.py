"""
Reference implementation (v3.6, second revision for JOI-D-26-00430) for:
  "The Publishing Game: How Editors, Reviewers, and Authors Shape Informetric Dynamics".

Python 3.9+, numpy, scipy, matplotlib.  Base seed = 42.
Replication seeds are derived deterministically from the base seed and the
configuration tuple (replication index, 1-phi, beta_r, k0) via SeedSequence,
so every table and figure of the manuscript is exactly reproducible.

Changes vs. v3.4 (the release cited in the previous version of the manuscript)
  * Standards.  The fold b = max(s, 1 - s) is removed: the priors s_E and
    s_A are defined on [0.5, 1] and b = s; priors outside that range raise
    a ValueError.  No number changes.
  * Experiment IV is the full 2 x 2 x 2 factorial of reforms, under both
    institutions, on each cell's own seeds and on common random numbers
    (the ten seeds of the Status Quo), with Welch and paired contrasts, the
    conditional effect of each reform given the other two, the three-way
    interaction, the contrasts on absolute net value per manuscript (NV),
    the closed-channel and forced benchmarks of the rationed cells, and
    cells separating the level of the authors' standard from the
    editor-author gap.
  * Convergence diagnostics.  Every fixed-point computation returns the
    iteration count, the stopping statistic, the residual re-evaluated at
    the reported state and a convergence flag; for every replicated
    configuration their mean and maximum over replications and the share of
    converged replications are archived, together with the per-replication
    values of GER, NV and the acceptance rate.
  * Participation-consistent equilibrium.  v3.4 tested the reviewer's
    participation constraint only at the pool induced by full participation
    and, if it failed, removed the review channel.  At the pool of a closed
    channel, however, the reviewer's maximized utility is positive in almost
    all of those configurations, so declining is not a best response there.
    v3.6 solves the participation stage consistently: a share pi of the
    review invitations is accepted (a declined invitation falls back on the
    desk) and pi* satisfies Pi_r(pi*) = 0 whenever neither corner is an
    equilibrium (run_config, participation="equilibrium", the default).
    The v3.4 rule is kept as the closed-channel benchmark
    (participation="closed") and full participation can be imposed
    (participation=False); the v3.4 values of the configurations that are
    now rationed are reproduced bit for bit by the closed-channel benchmark
    (see the README for the exact list).  The editor's logit index compares
    the two pathways conditional on the invitation being accepted (it is
    not rescaled by pi), and beliefs are formed on the realized pathway.
  * Experiment I controls: no reputational motive (r = 0); open-loop
    decomposition (pool and/or reviewer effort frozen at the healthy
    baseline); reputational beliefs conditioned on pathway and disposition,
    mu(t, a).  The l-type desk rate is stored for every configuration.
  * Experiment II: flat-fee benchmark (smallest fixed reward w_r sustaining
    full participation without a quality stake); fine contract grid on
    common random numbers (the ten seeds of the threshold contract).
  * Sensitivity S3: crowding out of the fixed reward by the stake
    (proportional loss kappa, floored at zero, and a fixed loss kappa0 at
    the introduction of a stake).
  * Appendix D: structural variants, among them an author's return tied to
    publication.  Each variant is evaluated at its own threshold, defined as
    the smallest contract on the 0.05 grid at which the mean utility of the
    reviewer under full participation, over the ten replications, is
    non-negative (the single-seed scan only provides the starting point).
  * Appendix E: every main configuration is re-solved to a tolerance of
    1e-7 on the same map and on a continuous version of it; the
    participation stage and the Experiment IV contrasts of both institutions
    are re-solved under each alternative solver; the reviewer's maximized
    utility is tabulated on a grid of imposed acceptance rates for six
    rationed configurations (participation_profile), which is the
    monotonicity that the bisection for pi* relies on.

Run "python final_run_v3_6.py" to recompute everything (writes results.json
and all figures), or "python final_run_v3_6.py --figures-only" to rebuild the
PNG figures from the archived results.json (Figure 3, figphase2.png, re-solves
one configuration at its first replication seed).
"""

import json
import sys

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------- baseline parameters (Table 2) -----------------
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

# Priors s_E, s_A live on [0.5, 1]; the standard is the prior itself (b = s).
ALIGNED = dict(sE=0.90, sA_lo=0.86, sA_hi=0.94)     # |bE-bA| < 0.05
FORCED = dict(sE=0.90, sA_lo=0.60, sA_hi=0.75)      # gap in [0.15,0.30]
MISALIGNED = dict(sE=0.95, sA_lo=0.50, sA_hi=0.55)  # gap > 0.40
# level versus gap of the standards (Experiment IV, v3.6)
ALIGNED_LOW = dict(sE=0.55, sA_lo=0.50, sA_hi=0.55)      # matched, low
MISALIGNED_HIGH = dict(sE=0.50, sA_lo=0.86, sA_hi=0.94)  # high, gap>0.35

QMAX = 1.0                        # Assumption 1 domain

# Experiment IV: the 2 x 2 x 2 factorial.  key -> (1-phi, beta_r, alignment).
# "Reformed" levels: editor 1-phi = 0.10 (E), reviewer beta_r = 0.40 (R),
# author-editor alignment ALIGNED (A).
EXP4_CELLS = {
    "status_quo":         (0.70, 0.00, MISALIGNED),   # no node reformed
    "partial":            (0.70, 0.40, MISALIGNED),   # R  (reviewer only)
    "editor_only":        (0.10, 0.00, MISALIGNED),   # E
    "editor_reviewer":    (0.10, 0.40, MISALIGNED),   # E + R
    "alignment_only":     (0.70, 0.00, ALIGNED),      # A
    "alignment_reviewer": (0.70, 0.40, ALIGNED),      # A + R
    "alignment_editor":   (0.10, 0.00, ALIGNED),      # A + E
    "optimal":            (0.10, 0.40, ALIGNED),      # A + E + R (joint)
}
EXP4_REFORMS = {
    "status_quo": "none", "partial": "R", "editor_only": "E",
    "editor_reviewer": "E+R", "alignment_only": "A",
    "alignment_reviewer": "A+R", "alignment_editor": "A+E",
    "optimal": "A+E+R",
}

# Solver settings of the robustness block (Appendix E)
TIGHT = dict(tol=1e-7, max_iter=3000)
SMOOTH = dict(smooth=True, tol=1e-7, max_iter=3000)


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
    """Pr(theta = 1 | e)."""
    if struct and struct.get("q_link") == "sqrt":
        return p["pi0"] + p["pi1"] * np.sqrt(e)
    return p["pi0"] + p["pi1"] * e


def reviewer_best_response(q_pop, beta_r, k0, p=P, struct=None, xatol=None):
    def negU(Q):
        C = p["cu"] * np.maximum(Q - q_pop, 0) + \
            p["cb"] * np.maximum(q_pop - Q, 0)
        Y = np.minimum(Q, q_pop)
        return -(p["w_r"] - p["alpha_r"] * C.mean()
                 + beta_r * Y.mean() - _rev_cost(Q, k0, struct))
    opts = {} if xatol is None else dict(options={"xatol": xatol})
    Q = float(minimize_scalar(negU, bounds=(0, QMAX), method="bounded",
                              **opts).x)
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


def config_seed(i, one_minus_phi, beta_r, k0=None):
    """Seed of replication i of configuration (1-phi, beta_r, k0)."""
    return int(np.random.SeedSequence(
        [P["seed"], i, int(one_minus_phi * 1000), int(beta_r * 1000),
         int((k0 or P["k0"]) * 100)]).generate_state(1)[0])


def reviewer_utility(Q, q_pop, beta_r, k0, p=P, struct=None):
    """(E[C], E[Y], Pi_r) of the reviewer's contract at a GIVEN effort Q."""
    C = p["cu"] * np.maximum(Q - q_pop, 0) + \
        p["cb"] * np.maximum(q_pop - Q, 0)
    EY = float(np.minimum(Q, q_pop).mean())
    EC = float(C.mean())
    return EC, EY, float(p["w_r"] - p["alpha_r"] * EC + beta_r * EY
                         - _rev_cost(Q, k0, struct))


def _solve(one_minus_phi, beta_r, sE, sA_lo, sA_hi, k0=None, N=None,
           p=P, seed=None, max_iter=400, tol=1e-4, damp=0.4,
           desk_reject_only=False, review_available=True, struct=None,
           smooth=False, pi=1.0, freeze=None, belief_by_action=False):
    """Fixed point of the coupled best responses for a GIVEN acceptance rate
    `pi` of review invitations (no participation logic; see run_config).

    pi : probability that a referred manuscript finds a reviewer.  A declined
      invitation falls back on the desk decision, so the manuscript is
      desk-decided with probability P_t0 + (1 - P_t0)(1 - pi).  pi = 1
      is the full-participation solution.
    desk_reject_only=True : the desk pathway can only reject (a = 0).
    review_available=False : review channel removed (closed-channel
      benchmark; equivalent to pi = 0 with the reviewer's effort set to 0).
    smooth=False : the fixed-point map uses the realized signal and pathway
      draws (the map of the paper; a step function of the state at
      resolution 1/N).  smooth=True integrates those draws out, so the map
      is continuous and can be solved to any tolerance; outcomes are still
      realized with the same draws (solver-robustness check, Appendix E).
    freeze : dict with "omega" and/or "Q".  The author's effort is computed
      at the frozen survival probability and/or the reviewer's effort is
      held at the frozen value, while the remaining agents best-respond
      (open-loop decomposition of Experiment I).
    belief_by_action=True : reputational beliefs condition on the pathway
      AND on the disposition, mu(t, a), instead of on the pathway alone
      (robustness check; realized map, baseline institution only).

    Convergence diagnostics returned with every solution:
      n_iter     : damped best-response iterations performed
      resid_stop : stopping statistic at exit: the largest change of the
                   state variables proposed by the best-response map at the
                   last iteration (undamped; the damped step is 0.4 of it)
      resid      : fixed-point residual re-evaluated at the reported state
      converged  : resid_stop < tol (the cap max_iter did not bind)
    """
    if not (0.5 <= sE <= 1.0 and 0.5 <= sA_lo <= sA_hi <= 1.0):
        raise ValueError("priors s_E and s_A must lie in [0.5, 1]")
    if belief_by_action and (smooth or desk_reject_only):
        raise NotImplementedError("mu(t, a) beliefs: realized map, baseline "
                                  "institution only")
    k0 = p["k0"] if k0 is None else k0
    N = p["N"] if N is None else N
    rng = np.random.default_rng(p["seed"] if seed is None else seed)
    phi = 1 - one_minus_phi
    frz = freeze or {}

    sA = rng.uniform(sA_lo, sA_hi, N)
    eps = rng.uniform(-p["sigma"], p["sigma"], N)
    tau_h = rng.random(N) < 0.5
    rho = np.where(tau_h, p["rho_h"], p["rho_l"])
    u_th, u_X, u_t, u_Z = (rng.random(N) for _ in range(4))
    bA = sA                       # standard = prior (priors on [0.5, 1])
    bE = sE
    h = np.maximum(0.0, 1 - p["lam"] * np.abs(bE - bA))
    xatol = 1e-12 if smooth else None   # Brent's default (1e-5) otherwise

    def desk_utility(b, X):
        if desk_reject_only:
            return -b * p["D"]                       # a = 0 forced
        return np.where(X == 1, b * p["BJ"] - (1 - b) * p["d"], -b * p["D"])

    def eff_desk(P_raw):
        """Probability of a desk decision once the declined invitations
        (share 1 - pi of the referrals) fall back on the desk."""
        if pi >= 1.0:
            return P_raw
        if pi <= 0.0:
            return np.ones(N)
        return P_raw + (1.0 - P_raw) * (1.0 - pi)

    def smooth_image(pe, alpha, Qn, mu0, mu1):
        """Expected-frequency image of (omega, mu0, mu1): the signal X and
        the pathway draw are integrated out analytically."""
        pX1 = pe * rho + (1 - pe) * (1 - rho)        # Pr(X = 1) per paper
        rR = (1 + Qn) / 2
        P_desk = []
        for x in (1, 0):
            Xc = np.full(N, x)
            b = bayes(alpha, rho, Xc)
            UJ1 = b * (rR * p["BJ"] - (1 - rR) * p["D"]) \
                - (1 - b) * (1 - rR) * p["d"] - p["c"]
            UJ0 = desk_utility(b, Xc)
            u1 = phi * UJ1 + (1 - phi) * p["r"] * mu1
            u0 = phi * UJ0 + (1 - phi) * p["r"] * mu0
            if review_available:
                P_desk.append(eff_desk(
                    1 / (1 + np.exp(-p["kappa_E"] * (u0 - u1)))))
            else:
                P_desk.append(np.ones(N))
        P1, P0 = P_desk
        Dk = pX1 * P1 + (1 - pX1) * P0               # Pr(t = 0) per paper
        if desk_reject_only:
            omega_n = float(1 - Dk.mean())
        else:
            omega_n = float(1 - ((1 - pX1) * P0).mean())
        s0, s1 = float(Dk.sum()), float((1 - Dk).sum())
        m0 = float((tau_h * Dk).sum() / s0) if s0 > 1e-12 else mu0
        m1 = float((tau_h * (1 - Dk)).sum() / s1) if s1 > 1e-12 else mu1
        return omega_n, m0, m1

    disc_return = bool(struct) and struct.get("author_return") == "disc"
    if disc_return and smooth:
        raise NotImplementedError("author_return='disc': realized map only")

    def disc_index(P_t0, X, theta, rR):
        """Appendix D, variant V4: the author's return is tied to publication.
        The state variable that enters the author's best response is the
        scaled discriminating power of the journal,
        Pr(pub | theta = 1) - Pr(pub | theta = 0), instead of the
        probability of surviving desk rejection."""
        pubp = (1 - P_t0) * np.where(theta == 1, rR, 1 - rR)
        if not desk_reject_only:
            pubp = pubp + P_t0 * (X == 1)
        g = theta == 1
        if g.all() or not g.any():
            return 0.0
        return float(struct["disc_scale"]
                     * (pubp[g].mean() - pubp[~g].mean()))

    def cell_beliefs(t0, X, Zr, old):
        """mu(t, a): share of h-types in each (pathway, disposition) cell."""
        cells = (t0 & (X == 1), t0 & (X == 0), ~t0 & (Zr == 1), ~t0 & (Zr == 0))
        return [float(tau_h[c].mean()) if c.any() else o
                for c, o in zip(cells, old)]

    omega, Q, mu0, mu1 = 0.9, 0.5, 0.6, 0.4
    mu_ta = [0.6, 0.6, 0.4, 0.4]      # (desk,acc) (desk,rej) (rev,acc) (rev,rej)
    n_iter, delta = 0, float("inf")
    Pi = 0.0
    for _ in range(max_iter):
        e = _author_effort(frz.get("omega", omega), bA, bE, h, p, struct)
        q = np.clip(e + eps, 0.0, 1.0 + p["sigma"])
        if review_available:
            Qn, EC, EY, Pi = reviewer_best_response(q, beta_r, k0, p, struct,
                                                    xatol)
            if "Q" in frz:
                Qn = float(frz["Q"])
        else:
            Qn, EC, EY, Pi = 0.0, 0.0, 0.0, 0.0
        pe = _quality_link(e, p, struct)
        alpha = float(pe.mean())
        new_ta = mu_ta
        if smooth:
            omega_n, m0, m1 = smooth_image(pe, alpha, Qn, mu0, mu1)
        else:
            theta = (u_th < pe).astype(int)
            X = np.where(u_X < rho, theta, 1 - theta)
            b = bayes(alpha, rho, X)
            rR = (1 + Qn) / 2
            UJ1 = b * (rR * p["BJ"] - (1 - rR) * p["D"]) \
                - (1 - b) * (1 - rR) * p["d"] - p["c"]
            UJ0 = desk_utility(b, X)
            if belief_by_action:
                Pacc = b * rR + (1 - b) * (1 - rR)
                u1 = phi * UJ1 + (1 - phi) * p["r"] * (
                    Pacc * mu_ta[2] + (1 - Pacc) * mu_ta[3])
                u0 = phi * UJ0 + (1 - phi) * p["r"] * np.where(
                    X == 1, mu_ta[0], mu_ta[1])
            else:
                u1 = phi * UJ1 + (1 - phi) * p["r"] * mu1
                u0 = phi * UJ0 + (1 - phi) * p["r"] * mu0
            if review_available:
                P_t0 = eff_desk(1 / (1 + np.exp(-p["kappa_E"] * (u0 - u1))))
            else:
                P_t0 = np.ones(N)
            if desk_reject_only:
                # desk pathway = rejection of every manuscript routed there
                omega_n = float(1 - P_t0.mean())
            else:
                omega_n = float(1 - (P_t0 * (X == 0)).mean())
            if disc_return:
                omega_n = disc_index(P_t0, X, theta, rR)
            t0 = u_t < P_t0
            m0 = float(tau_h[t0].mean()) if t0.any() else mu0
            m1 = float(tau_h[~t0].mean()) if (~t0).any() else mu1
            if belief_by_action:
                Zr = np.where(u_Z < rR, theta, 1 - theta)
                new_ta = cell_beliefs(t0, X, Zr, mu_ta)
        delta = max(abs(omega_n - omega), abs(Qn - Q),
                    abs(m0 - mu0), abs(m1 - mu1))
        if belief_by_action:
            delta = max([delta] + [abs(a - o) for a, o in zip(new_ta, mu_ta)])
            mu_ta = [o + damp * (a - o) for a, o in zip(new_ta, mu_ta)]
        omega += damp * (omega_n - omega)
        Q += damp * (Qn - Q)
        mu0 += damp * (m0 - mu0)
        mu1 += damp * (m1 - mu1)
        n_iter += 1
        if delta < tol:
            break
    resid_stop = float(delta)
    converged = bool(delta < tol)
    Pi_loop = float(Pi)               # reviewer utility at the last iterate

    # realized epoch at the fixed point
    Q_state = Q                                    # damped state of the loop
    e = _author_effort(frz.get("omega", omega), bA, bE, h, p, struct)
    q = np.clip(e + eps, 0.0, 1.0 + p["sigma"])
    if review_available:
        Q, EC, EY, Pi = reviewer_best_response(q, beta_r, k0, p, struct,
                                               xatol)
        if "Q" in frz:
            Q = float(frz["Q"])
            EC, EY, Pi = reviewer_utility(Q, q, beta_r, k0, p, struct)
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
    if belief_by_action:
        Pacc = b * rR + (1 - b) * (1 - rR)
        u1 = phi * UJ1 + (1 - phi) * p["r"] * (
            Pacc * mu_ta[2] + (1 - Pacc) * mu_ta[3])
        u0 = phi * UJ0 + (1 - phi) * p["r"] * np.where(
            X == 1, mu_ta[0], mu_ta[1])
    else:
        u1 = phi * UJ1 + (1 - phi) * p["r"] * mu1
        u0 = phi * UJ0 + (1 - phi) * p["r"] * mu0
    if review_available:
        P_raw = 1 / (1 + np.exp(-p["kappa_E"] * (u0 - u1)))
        P_t0 = eff_desk(P_raw)
    else:
        P_raw = np.ones(N)
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

    # fixed-point residual re-evaluated at the reported state
    if smooth:
        omega_fp, m0_fp, m1_fp = smooth_image(pe, alpha, Q, mu0, mu1)
    else:
        if desk_reject_only:
            omega_fp = float(1 - P_t0.mean())
        else:
            omega_fp = float(1 - (P_t0 * (X == 0)).mean())
        if disc_return:
            omega_fp = disc_index(P_t0, X, theta, rR)
        m0_fp = float(tau_h[t0].mean()) if t0.any() else mu0
        m1_fp = float(tau_h[~t0].mean()) if (~t0).any() else mu1
    resid = max(abs(omega_fp - omega), abs(Q - Q_state),
                abs(m0_fp - mu0), abs(m1_fp - mu1))
    if belief_by_action:
        fp_ta = cell_beliefs(t0, X, Z, mu_ta)
        resid = max([resid] + [abs(a - o) for a, o in zip(fp_ta, mu_ta)])

    out = dict(
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
        n_iter=int(n_iter), resid_stop=resid_stop, resid=float(resid),
        converged=converged,
    )
    # added in v3.6 ---------------------------------------------------------
    out.update(
        pi=float(pi) if review_available else 0.0,
        Pi_loop=Pi_loop, Q_state=float(Q_state),
        desk_rate_l=float(t0[~tau_h].mean()),
        desk_accept_rate=(0.0 if desk_reject_only
                          else float((t0 & (X == 1)).mean())),
        refer_intended=float(1 - P_raw.mean()),
        NV=float((value.sum() - effort_costs.sum() - review_costs.sum())
                 / N),
        # discriminating power: Pr(pub | theta = 1) - Pr(pub | theta = 0)
        disc=(float(published[good].mean() - published[~good].mean())
              if good.any() and not good.all() else 0.0),
    )
    if belief_by_action:
        out.update(mu_desk_acc=float(mu_ta[0]), mu_desk_rej=float(mu_ta[1]),
                   mu_rev_acc=float(mu_ta[2]), mu_rev_rej=float(mu_ta[3]))
    return out


def run_config(one_minus_phi, beta_r, sE, sA_lo, sA_hi, k0=None, N=None,
               p=P, seed=None, max_iter=400, tol=1e-4, damp=0.4,
               participation="equilibrium", desk_reject_only=False,
               review_available=True, struct=None, smooth=False,
               pi=None, pi_tol=1e-3, freeze=None, belief_by_action=False):
    """Solve one configuration.

    participation selects how the reviewer's participation stage is solved:

      "equilibrium" (default; True is an alias)
          Participation-consistent equilibrium.  A share pi of the review
          invitations is accepted; a declined invitation falls back on the
          desk.  pi* = 1 if the reviewer's maximized utility Pi_r is >= 0 at
          the pool induced by full participation; pi* = 0 if Pi_r <= 0 at
          the pool induced by a closed channel; otherwise pi* in (0, 1)
          solves Pi_r(pi*) = 0 (indifference), found by bisection to pi_tol.
      "closed"
          rule of v3.4, kept as a benchmark: if Pi_r < 0 under full
          participation the review channel is removed altogether, whether
          or not declining is a best response to the resulting pool.
      False / "forced"
          Full participation imposed (pi = 1).

    pi : impose the acceptance rate instead of solving for it.
    The remaining arguments are documented in _solve.
    """
    kw = dict(k0=k0, N=N, p=p, seed=seed, max_iter=max_iter, tol=tol,
              damp=damp, desk_reject_only=desk_reject_only, struct=struct,
              smooth=smooth, freeze=freeze, belief_by_action=belief_by_action)

    def solve(pi_, available=True):
        return _solve(one_minus_phi, beta_r, sE, sA_lo, sA_hi, pi=pi_,
                      review_available=available, **kw)

    if not review_available:
        return solve(1.0, False)
    if pi is not None:
        return solve(float(pi))
    full = solve(1.0)
    if participation in (False, "forced") or full["Pi_loop"] >= 0:
        return full
    forced = dict(Pi_r_forced=full["Pi_loop"],   # utility under pi = 1
                  Q_forced=full["Q_state"],
                  n_iter_forced=full["n_iter"],  # candidate (forced) solve
                  resid_stop_forced=full["resid_stop"],
                  converged_forced=full["converged"])
    if participation == "closed":
        out = solve(1.0, False)
        out.update(forced)
        return out
    # participation-consistent equilibrium ---------------------------------
    closed = solve(0.0)               # reviewer's best response to the pool
    if closed["Pi_loop"] <= 0:        # of a closed channel: stays out
        out = solve(1.0, False)
        out.update(forced, Pi_r_closed=closed["Pi_loop"], n_bisect=0)
        return out
    lo, hi, n_bisect = 0.0, 1.0, 0
    while hi - lo > pi_tol:
        mid = 0.5 * (lo + hi)
        if solve(mid)["Pi_loop"] > 0:
            lo = mid
        else:
            hi = mid
        n_bisect += 1
    out = solve(0.5 * (lo + hi))
    out.update(forced, Pi_r_closed=closed["Pi_loop"], n_bisect=n_bisect)
    return out


def replicate(one_minus_phi, beta_r, sc, k0=None, nseed=NSEED, N=None,
              seed_cfg=None, **kw):
    """Run a configuration across nseed seeds; return means + s.e.

    seed_cfg=(1-phi, beta_r[, k0]) derives the seeds from ANOTHER
    configuration (common random numbers across configurations)."""
    s_w, s_b, s_k = (one_minus_phi, beta_r, k0)
    if seed_cfg is not None:
        s_w, s_b = seed_cfg[0], seed_cfg[1]
        s_k = seed_cfg[2] if len(seed_cfg) > 2 else None
    outs = []
    for i in range(nseed):
        seed = config_seed(i, s_w, s_b, s_k)
        outs.append(run_config(one_minus_phi, beta_r, **sc, k0=k0, N=N,
                               seed=seed, **kw))
    num_keys = [k for k, v in outs[0].items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
                and all(k in o for o in outs)]
    agg = {k: float(np.mean([o[k] for o in outs])) for k in num_keys}
    for k in ("GER", "desk_reject_rate", "typeII", "Q", "e_mean", "Pi_r",
              "Ysys", "UJ", "omega", "EC", "EY", "pi", "NV", "pub_prob",
              "review_share", "desk_rate_h", "desk_rate_l", "disc"):
        v = np.array([o[k] for o in outs], dtype=float)
        agg[k + "_se"] = float(v.std(ddof=1) / np.sqrt(len(v)))
    agg["participation_share"] = float(
        np.mean([o["participates"] for o in outs]))
    pf = [o.get("Pi_r_forced") for o in outs if "Pi_r_forced" in o]
    qf = [o.get("Q_forced") for o in outs if "Q_forced" in o]
    if pf:                  # runs in which full participation is not IR
        agg["Pi_r_forced"] = float(np.mean(pf))
        agg["Q_forced"] = float(np.mean(qf))
        agg["n_runs_not_ir"] = len(pf)
    pc = [o["Pi_r_closed"] for o in outs if "Pi_r_closed" in o]
    if pc:                  # deviation utility at the closed-channel pool
        agg["Pi_r_closed"] = float(np.mean(pc))
        agg["Pi_r_closed_min"] = float(np.min(pc))
        agg["n_bisect_max"] = int(max(o["n_bisect"] for o in outs
                                      if "n_bisect" in o))
    # convergence diagnostics and per-replication GER
    agg["n_iter_max"] = int(max(o["n_iter"] for o in outs))
    agg["resid_max"] = float(max(o["resid"] for o in outs))
    agg["resid_stop_max"] = float(max(o["resid_stop"] for o in outs))
    agg["converged_share"] = float(np.mean([o["converged"] for o in outs]))
    cf = [o["converged_forced"] for o in outs if "converged_forced" in o]
    if cf:                  # convergence of the full-participation candidate
        agg["converged_forced_share"] = float(np.mean(cf))
    agg["n_runs"] = len(outs)
    agg["GER_runs"] = [float(o["GER"]) for o in outs]
    agg["NV_runs"] = [float(o["NV"]) for o in outs]      # v3.6
    agg["pi_runs"] = [float(o["pi"]) for o in outs]      # v3.6
    return agg


def _regime(pi_runs):
    """Participation regime of a configuration across its replications."""
    a = np.asarray(pi_runs, float)
    if (a >= 1).all():
        return "full"
    if (a <= 0).all():
        return "closed"
    return "rationed"


# ------------------------- contrast statistics -------------------------
def welch(a, b):
    """b - a for independent replication samples (Welch 95% interval)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = float(np.sqrt(va + vb))
    if va + vb > 0:
        df = float((va + vb) ** 2 / (va ** 2 / (len(a) - 1)
                                     + vb ** 2 / (len(b) - 1)))
    else:
        df = float(len(a) + len(b) - 2)
    d = float(b.mean() - a.mean())
    tq = float(stats.t.ppf(0.975, df))
    return dict(delta=d, se=se, df=df, ci95=[d - tq * se, d + tq * se])


def lincomb(runs, weights):
    """Paired linear contrast sum_k w_k GER_k on common-random-number runs."""
    d = sum(w * np.asarray(runs[k], float) for k, w in weights.items())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    tq = float(stats.t.ppf(0.975, len(d) - 1))
    m = float(d.mean())
    return dict(delta=m, se=se, df=len(d) - 1,
                ci95=[m - tq * se, m + tq * se],
                max_abs=float(np.abs(d).max()))


EXP4_BY_SET = {"": "status_quo", "R": "partial", "E": "editor_only",
               "ER": "editor_reviewer", "A": "alignment_only",
               "AR": "alignment_reviewer", "AE": "alignment_editor",
               "AER": "optimal"}


def exp4_contrasts(own, crn, key="GER_runs"):
    """Reform contrasts of the factorial: own-seed (Welch) and CRN (paired).
    key selects the outcome: "GER_runs" (default) or "NV_runs"."""
    g = {k: v[key] for k, v in own.items()}
    c = {k: v[key] for k, v in crn.items()}
    out = dict(vs_status_quo={}, given_alignment={})
    for k in own:
        if k == "status_quo":
            continue
        out["vs_status_quo"][k] = dict(
            own_seeds=welch(g["status_quo"], g[k]),
            crn=lincomb(c, {k: 1.0, "status_quo": -1.0}))
    for k in ("alignment_reviewer", "alignment_editor", "optimal"):
        out["given_alignment"][k] = dict(
            own_seeds=welch(g["alignment_only"], g[k]),
            crn=lincomb(c, {k: 1.0, "alignment_only": -1.0}))
    # joint gain minus the sum of the three single-node gains
    out["superadditivity"] = dict(crn=lincomb(c, {
        "optimal": 1.0, "partial": -1.0, "editor_only": -1.0,
        "alignment_only": -1.0, "status_quo": 2.0}))
    joint = out["vs_status_quo"]["optimal"]["crn"]["delta"]
    out["share_of_joint_gain"] = {
        k: (out["vs_status_quo"][k]["crn"]["delta"] / joint
            if joint != 0 else float("nan"))
        for k in out["vs_status_quo"]}
    # v3.6: effect of each reform conditional on each combination of the
    # other two (increasing differences = complementarity), and the
    # three-way interaction of the 2^3 design
    cell = lambda s: EXP4_BY_SET["".join(x for x in "AER" if x in s)]
    cond = {}
    for x in "ERA":
        rest = [y for y in "ERA" if y != x]
        for others in ("", rest[0], rest[1], rest[0] + rest[1]):
            name = "%s|%s" % (x, "+".join(y for y in "AER" if y in others)
                              or "none")
            cond[name] = lincomb(c, {cell(others + x): 1.0,
                                     cell(others): -1.0})
    out["conditional_effects"] = cond
    out["three_way"] = lincomb(c, {
        "optimal": 1.0, "alignment_editor": -1.0, "alignment_reviewer": -1.0,
        "editor_reviewer": -1.0, "alignment_only": 1.0, "editor_only": 1.0,
        "partial": 1.0, "status_quo": -1.0})
    return out


HEAD_STATS = ("GER", "desk_reject_rate", "typeII", "Q", "e_mean", "omega",
              "pub_prob", "Ysys", "UJ", "pi")


def _headline(o):
    """Compact copy of the baseline aggregate (the full one is in R)."""
    d = {k: o[k] for k in HEAD_STATS}
    d.update(GER_se=o["GER_se"], participation_share=o["participation_share"],
             n_iter=o["n_iter"], n_iter_max=o["n_iter_max"],
             resid=o["resid"], resid_max=o["resid_max"],
             resid_stop_max=o["resid_stop_max"],
             converged_share=o["converged_share"])
    return d


def main():
    figures_only = "--figures-only" in sys.argv
    if figures_only:
        with open("results.json") as f:
            R = json.load(f)
        make_figures(R)
        return

    R = {}
    REG = []          # registry of the main replicated configurations

    def rep(family, label, w, br, sc, **kw):
        o = replicate(w, br, sc, **kw)
        REG.append((family, str(label), (w, br, sc), kw, o))
        return o

    # Experiment I: signaling sweep (participation equilibrium) -------------
    print("Experiment I ...")
    sweep = {round(w, 2): rep("exp1", round(w, 2), round(w, 2), 0.40, ALIGNED)
             for w in np.arange(0.0, 1.001, 0.05)}
    R["exp1"] = sweep
    g0 = sweep[0.0]["GER"]
    for o in sweep.values():      # (the clip at zero never binds)
        o["eff_loss"] = max(0.0, (g0 - o["GER"]) / g0)

    # Experiment I, controls (v3.6) -----------------------------------------
    # (a) no reputational motive (r = 0): what the loss of payoff sensitivity
    #     of the logit rule produces on its own;
    # (b) open-loop decomposition: the pool and/or the reviewer's effort are
    #     frozen at the healthy baseline (1 - phi = 0.10) while the editor
    #     best-responds: what the closed loop adds to the editor-only game;
    # (c) reputational beliefs conditioned on pathway AND disposition.
    print("Experiment I (controls) ...")
    W8 = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0)
    R["exp1_r0"] = {w: replicate(w, 0.40, ALIGNED, p=dict(P, r=0.0))
                    for w in W8}
    fz = dict(omega=sweep[0.1]["omega"], Q=sweep[0.1]["Q"])
    R["exp1_frozen"] = dict(
        frozen_at=dict(one_minus_phi=0.1, **fz),
        editor_only={w: replicate(w, 0.40, ALIGNED, participation=False,
                                  freeze=fz) for w in W8},
        author_loop={w: replicate(w, 0.40, ALIGNED, participation=False,
                                  freeze=dict(Q=fz["Q"])) for w in W8},
        reviewer_loop={w: replicate(w, 0.40, ALIGNED, participation=False,
                                    freeze=dict(omega=fz["omega"]))
                       for w in W8})
    R["exp1_mu_ta"] = {round(w, 2): replicate(round(w, 2), 0.40, ALIGNED,
                                              belief_by_action=True)
                       for w in np.arange(0.0, 1.001, 0.05)}

    # Experiment I-R: desk-reject-only robustness ---------------------------
    print("Experiment I-R (desk-reject only) ...")
    sweepR = {round(w, 2): rep("exp1_dro", round(w, 2), round(w, 2), 0.40,
                               ALIGNED, desk_reject_only=True)
              for w in np.arange(0.0, 1.001, 0.1)}
    R["exp1_dro"] = sweepR
    g0R = sweepR[0.0]["GER"]
    for o in sweepR.values():
        o["eff_loss"] = max(0.0, (g0R - o["GER"]) / g0R)

    # Experiment II: contract sweep -----------------------------------------
    # exp2        : participation-consistent equilibrium (acceptance rate pi*)
    # exp2_closed : closed-channel benchmark (the "collapse" numbers of v3.4)
    # exp2_forced : full participation imposed
    print("Experiment II ...")
    BR2 = (0.0, 0.10, 0.20, 0.30, 0.40, 0.60)
    R["exp2"] = {br: rep("exp2", br, 0.1, br, ALIGNED) for br in BR2}
    R["exp2_closed"] = {br: replicate(0.1, br, ALIGNED,
                                      participation="closed") for br in BR2}
    R["exp2_forced"] = {br: rep("exp2_forced", br, 0.1, br, ALIGNED,
                                participation=False) for br in BR2}
    # fine grid: acceptance rate, effort and journal welfare by contract.
    # Every grid point is solved on the ten seeds of the threshold contract
    # (common random numbers), so differences along the grid are paired.
    WJ, wj_runs = {}, {}
    grid_seeds = [config_seed(i, 0.1, 0.40) for i in range(NSEED)]
    tq = float(stats.t.ppf(0.975, NSEED - 1))
    for br in np.round(np.arange(0.0, 0.801, 0.05), 2):
        br = float(br)
        outs = [run_config(0.1, br, **ALIGNED, seed=s) for s in grid_seeds]
        wj = np.array([o["Ysys"] - br * o["EY"] * o["review_share"]
                       for o in outs])
        wj_runs[br] = wj

        def m(k):
            return float(np.mean([o[k] for o in outs]))

        def se(k):
            v = np.array([o[k] for o in outs], dtype=float)
            return float(v.std(ddof=1) / np.sqrt(len(v)))

        WJ[br] = dict(
            WJ=float(wj.mean()),
            WJ_se=float(wj.std(ddof=1) / np.sqrt(len(wj))),
            Q=m("Q"), Pi=m("Pi_r"),
            part=float(np.mean([o["participates"] for o in outs])),
            pi=m("pi"), pi_se=se("pi"),
            pi_min=float(min(o["pi"] for o in outs)),
            pi_max=float(max(o["pi"] for o in outs)),
            GER=m("GER"), GER_se=se("GER"),
            desk_reject_rate=m("desk_reject_rate"), typeII=m("typeII"),
            review_share=m("review_share"),
            Pi_forced=float(np.mean([o.get("Pi_r_forced", o["Pi_r"])
                                     for o in outs])))
    for br, wj in wj_runs.items():      # paired difference from beta_r = 0.40
        d = wj - wj_runs[0.4]
        dse = float(d.std(ddof=1) / np.sqrt(len(d)))
        WJ[br]["dWJ_vs_040"] = dict(
            delta=float(d.mean()), se=dse,
            ci95=[float(d.mean() - tq * dse), float(d.mean() + tq * dse)])
    R["exp2_WJ"] = WJ

    # flat fee: smallest fixed reward w_r (0.01 grid) that sustains full
    # participation with no quality stake, and the outcome it delivers
    wr_star = None
    for wr in np.round(np.arange(P["w_r"], 0.801, 0.01), 2):
        o = run_config(0.1, 0.0, **ALIGNED, p=dict(P, w_r=float(wr)),
                       N=4000, participation=False)
        if o["Pi_r"] >= 0:
            wr_star = float(wr)
            break
    R["exp2_flatfee"] = dict(
        w_r_star=wr_star,
        at_w_r_star=replicate(0.1, 0.0, ALIGNED, p=dict(P, w_r=wr_star)),
        stake_expected_payment=0.40 * R["exp2"][0.40]["EY"])

    # Experiment III --------------------------------------------------------
    print("Experiment III ...")
    R["exp3"] = {n: rep("exp3", n, 0.1, 0.40, sc) for n, sc in
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

    # Experiment IV: full 2 x 2 x 2 factorial -------------------------------
    print("Experiment IV (2 x 2 x 2 factorial) ...")
    R["exp4"] = {k: rep("exp4", k, w, br, sc)
                 for k, (w, br, sc) in EXP4_CELLS.items()}
    R["exp4_dro"] = {k: rep("exp4_dro", k, w, br, sc, desk_reject_only=True)
                     for k, (w, br, sc) in EXP4_CELLS.items()}
    # common random numbers: every cell on the Status Quo's ten seeds
    sq = EXP4_CELLS["status_quo"][:2]
    R["exp4_crn"] = {k: replicate(w, br, sc, seed_cfg=sq)
                     for k, (w, br, sc) in EXP4_CELLS.items()}
    R["exp4_dro_crn"] = {k: replicate(w, br, sc, seed_cfg=sq,
                                      desk_reject_only=True)
                         for k, (w, br, sc) in EXP4_CELLS.items()}
    R["exp4_contrasts"] = exp4_contrasts(R["exp4"], R["exp4_crn"])
    R["exp4_dro_contrasts"] = exp4_contrasts(R["exp4_dro"], R["exp4_dro_crn"])
    # the same contrasts on absolute net value per manuscript (not divided
    # by the cell's own potential alpha * B_J)
    R["exp4_contrasts_nv"] = exp4_contrasts(R["exp4"], R["exp4_crn"],
                                            key="NV_runs")
    R["exp4_dro_contrasts_nv"] = exp4_contrasts(
        R["exp4_dro"], R["exp4_dro_crn"], key="NV_runs")
    R["exp4_design"] = {k: dict(one_minus_phi=w, beta_r=br,
                                aligned=bool(sc is ALIGNED),
                                reformed=EXP4_REFORMS[k])
                        for k, (w, br, sc) in EXP4_CELLS.items()}
    # closed-channel and forced benchmarks of the cells with pi* < 1 (CRN)
    R["exp4_closed"], R["exp4_forced"] = {}, {}
    R["exp4_dro_closed"], R["exp4_dro_forced"] = {}, {}
    for k, (w, br, sc) in EXP4_CELLS.items():
        if R["exp4_crn"][k]["pi"] < 1:
            R["exp4_closed"][k] = replicate(w, br, sc, seed_cfg=sq,
                                            participation="closed")
            R["exp4_forced"][k] = replicate(w, br, sc, seed_cfg=sq,
                                            participation=False)
        if R["exp4_dro_crn"][k]["pi"] < 1:
            R["exp4_dro_closed"][k] = replicate(
                w, br, sc, seed_cfg=sq, participation="closed",
                desk_reject_only=True)
            R["exp4_dro_forced"][k] = replicate(
                w, br, sc, seed_cfg=sq, participation=False,
                desk_reject_only=True)
    # the rationed cell A+E with the flat reward of Experiment II in place of
    # the stake (CRN): what the reviewer-side reform contributes is full
    # participation, by whatever transfer
    R["exp4_flat"] = dict(
        w_r=wr_star,
        alignment_editor=replicate(0.10, 0.00, ALIGNED, seed_cfg=sq,
                                   p=dict(P, w_r=wr_star)))
    R["exp4_flat"]["share_of_joint_gain"] = float(
        (np.mean(R["exp4_flat"]["alignment_editor"]["GER_runs"])
         - np.mean(R["exp4_crn"]["status_quo"]["GER_runs"]))
        / R["exp4_contrasts"]["vs_status_quo"]["optimal"]["crn"]["delta"])
    # level versus gap of the standards (CRN): the alignment lever of the
    # factorial changes both; these cells separate them
    R["exp4_levelgap"] = {
        name: {cell: replicate(w, br, sc, seed_cfg=sq)
               for cell, (w, br) in (("none", (0.70, 0.00)),
                                     ("E", (0.10, 0.00)),
                                     ("R", (0.70, 0.40)),
                                     ("E+R", (0.10, 0.40)))}
        for name, sc in (("low_matched", ALIGNED_LOW),
                         ("high_mismatched", MISALIGNED_HIGH))}

    # Experiment V ----------------------------------------------------------
    print("Experiment V ...")
    K5 = (1.5, 2.0, 3.0, 4.0, 5.0)
    R["exp5"] = {str(k0): rep("exp5", k0, 0.1, 0.40, ALIGNED, k0=k0)
                 for k0 in K5}
    R["exp5_closed"] = {str(k0): replicate(0.1, 0.40, ALIGNED, k0=k0,
                                           participation="closed")
                        for k0 in K5}
    # forced-participation counterfactual
    R["exp5_forced"] = {str(k0): rep("exp5_forced", k0, 0.1, 0.40, ALIGNED,
                                     k0=k0, participation=False)
                        for k0 in K5}

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

    # Sensitivity S2: minimal stake sustaining full participation -----------
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
                          for wr in (0.0, 0.15, 0.3, 0.45, 0.5, 0.55, 0.6)}

    # Sensitivity S3: crowding out of intrinsic motivation ------------------
    print("Sensitivity S3 (crowding out) ...")

    def beta_star_crowding(kappa, kappa0=0.0, br_max=6.0):
        """Smallest positive contract on the 0.05 grid with forced utility
        >= 0 when the stake crowds out the fixed social reward:
        w_r(beta_r) = max{0, w_r - kappa0 * 1{beta_r > 0} - kappa * beta_r}
        (kappa0: loss at the introduction of any stake; kappa: loss per unit
        of stake; the stake can destroy the fixed reward but cannot turn it
        negative).  Also returns the reviewer's effort and the expected
        published quality E[Y*] at the last node scanned (the threshold, or
        br_max if no contract in the range sustains full participation)."""
        o, br = None, None
        start = 0.0 if kappa0 == 0.0 else 0.05
        for br in np.round(np.arange(start, br_max + 1e-9, 0.05), 2):
            p2 = dict(P)
            p2["w_r"] = max(0.0, P["w_r"] - kappa0 * float(br > 0)
                            - kappa * float(br))
            o = run_config(0.1, float(br), **ALIGNED, p=p2, N=4000,
                           participation=False)
            if o["Pi_r"] >= 0:
                return dict(beta_star=float(br), EY=o["EY"], Q=o["Q"],
                            w_r_eff=p2["w_r"])
        return dict(beta_star=None, EY=o["EY"], Q=o["Q"],
                    scanned_up_to=float(br))

    R["sens_bstar_kappa"] = {
        str(kp): beta_star_crowding(kp)
        for kp in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0)}
    R["sens_bstar_kappa0"] = {
        str(k0f): {str(kp): beta_star_crowding(kp, kappa0=k0f)
                   for kp in (0.0, 0.2, 0.4)}
        for k0f in (0.05, 0.10, 0.20)}

    # Appendix D: structural robustness ---------------------------------
    # For each structural variant we first re-derive the variant's own
    # full-participation threshold beta_r* on the 0.05 grid (as Experiment
    # II does for the baseline technology), and then re-run the headline
    # diagnostics at that contract.  Seeds follow the same derivation
    # rule as everywhere else, keyed by (one_minus_phi, beta_r, k0).
    print("Appendix D (structural robustness) ...")
    VARIANTS = {
        "baseline": None,
        "cubic_reviewer": dict(eta_r=2),      # reviewer cost k0 Q^3 / 3
        "concave_link": dict(q_link="sqrt"),  # p(e) = pi0 + pi1 sqrt(e)
        "convex_author": dict(m_a=2),         # author cost A e^3 / 3
    }
    # V4: the author's return is tied to publication.  The survival
    # probability omega in the author's best response is replaced by the
    # journal's discriminating power Pr(pub | theta=1) - Pr(pub | theta=0),
    # scaled so that the healthy baseline (1 - phi = 0.1, beta_r = 0.40)
    # is unchanged.
    disc_scale = R["exp2"][0.40]["omega"] / R["exp2"][0.40]["disc"]
    VARIANTS["pub_return"] = dict(author_return="disc",
                                  disc_scale=float(disc_scale))
    R["structD_disc_scale"] = float(disc_scale)
    SD = {}
    for name, struct in VARIANTS.items():
        kw = {} if struct is None else dict(struct=struct)
        d = {}
        # (ii) rationing of review and its removal (Exp. II): smallest
        # contract on the 0.05 grid sustaining full participation
        bstar = None
        for br in np.round(np.arange(0.0, 1.501, 0.05), 2):
            o = run_config(0.1, float(br), **ALIGNED, N=4000,
                           participation=False, **kw)
            if o["Pi_r"] >= 0:
                bstar = float(br)
                break
        d["beta_star_scan"] = bstar
        # confirm on the ten replications (N = 10,000): the threshold is the
        # smallest grid contract at which the mean utility of the reviewer
        # under full participation is non-negative
        while replicate(0.1, bstar, ALIGNED, participation=False,
                        **kw)["Pi_r"] < 0:
            bstar = float(round(bstar + 0.05, 2))
        d["beta_star"] = bstar
        d["expII"] = {str(br): replicate(0.1, br, ALIGNED, **kw)
                      for br in (0.0, bstar)}
        d["expII_closed"] = replicate(0.1, 0.0, ALIGNED,
                                      participation="closed", **kw)
        # (i) signaling sweep (Exp. I) at the variant's own threshold:
        # low / peak / corner signaling weights
        d["expI"] = {str(w): replicate(w, bstar, ALIGNED, **kw)
                     for w in (0.1, 0.65, 1.0)}
        # (iii) fatigue (Exp. V): smallest friction on the grid at which
        # full participation fails at the variant's contract, then the
        # equilibrium, closed-channel and forced outcomes at that friction
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
            d["expV_closed"] = replicate(0.1, bstar, ALIGNED, k0=k0_cliff,
                                         participation="closed", **kw)
            d["expV_forced"] = replicate(0.1, bstar, ALIGNED, k0=k0_cliff,
                                         participation=False, **kw)
        SD[name] = d
    R["structD"] = SD

    # Participation profile: the reviewer's maximized utility as a function
    # of an imposed acceptance rate pi, for the rationed configurations.  The
    # bisection of run_config relies on this profile being decreasing in pi
    # (a higher acceptance rate produces a healthier pool, which is costlier
    # to review), so that the indifference point is unique.
    print("Participation profile ...")
    PGRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

    def profile(w, br, k0=None, **kw):
        rows = np.array([[run_config(w, br, **ALIGNED, k0=k0, pi=g,
                                     seed=config_seed(i, w, br, k0),
                                     **kw)["Pi_loop"] for g in PGRID]
                         for i in range(NSEED)])
        return dict(
            pi_grid=list(PGRID),
            Pi_r_mean=[float(v) for v in rows.mean(0)],
            Pi_r_se=[float(v) for v in rows.std(0, ddof=1) / np.sqrt(NSEED)],
            decreasing_in_every_run=bool(np.all(np.diff(rows, axis=1) < 0)),
            sign_changes_max=int(max(
                int(np.sum(np.diff(np.sign(r)) != 0)) for r in rows)))

    R["participation_profile"] = {
        "exp2_beta_0.0": profile(0.1, 0.0),
        "exp2_beta_0.2": profile(0.1, 0.20),
        "exp5_k0_3.0": profile(0.1, 0.40, k0=3.0),
        "exp5_k0_5.0": profile(0.1, 0.40, k0=5.0),
        "dro_alignment_only": profile(0.70, 0.0, desk_reject_only=True),
        "dro_alignment_editor": profile(0.10, 0.0, desk_reject_only=True),
    }

    # Appendix E: solver robustness -------------------------------------
    # Every main configuration (Experiments I-V and the desk-reject-only
    # variants) is re-solved on the same seeds with a tolerance three orders
    # of magnitude tighter (1e-7, cap 3000): (a) on the same map ("tight")
    # and (b) on the smooth expected-frequency map ("smooth").  The
    # participation stage is re-solved under each solver.
    print("Appendix E (solver robustness: %d configurations) ..." % len(REG))
    by_cfg, summ = {}, {}
    for family, label, (w, br, sc), kw, base in REG:
        tight = replicate(w, br, sc, **kw, **TIGHT)
        smooth = replicate(w, br, sc, **kw, **SMOOTH)
        by_cfg.setdefault(family, {})[label] = dict(
            baseline=_headline(base), tight=tight, smooth=smooth)
        s = summ.setdefault(family, dict(
            n_configs=0, n_runs=0, base_converged_runs=0.0, base_n_iter_max=0,
            base_n_iter_sum=0.0,
            base_resid_max=0.0, tight_converged_runs=0.0, tight_resid_max=0.0,
            smooth_converged_runs=0.0, smooth_n_iter_max=0,
            smooth_resid_max=0.0, participation_flips=0,
            max_abs_diff_tight={k: 0.0 for k in HEAD_STATS},
            max_abs_diff_smooth={k: 0.0 for k in HEAD_STATS},
            max_GER_diff_in_se_tight=0.0, max_GER_diff_in_se_smooth=0.0))
        n = base["n_runs"]
        s["n_configs"] += 1
        s["n_runs"] += n
        s["base_converged_runs"] += base["converged_share"] * n
        s["base_n_iter_max"] = max(s["base_n_iter_max"], base["n_iter_max"])
        s["base_n_iter_sum"] += base["n_iter"] * n
        s["base_resid_max"] = max(s["base_resid_max"], base["resid_max"])
        s["tight_converged_runs"] += tight["converged_share"] * n
        s["tight_resid_max"] = max(s["tight_resid_max"], tight["resid_max"])
        s["smooth_converged_runs"] += smooth["converged_share"] * n
        s["smooth_n_iter_max"] = max(s["smooth_n_iter_max"],
                                     smooth["n_iter_max"])
        s["smooth_resid_max"] = max(s["smooth_resid_max"],
                                    smooth["resid_max"])
        for alt, tag in ((tight, "tight"), (smooth, "smooth")):
            # regime change of the participation stage (pi* = 1, interior,
            # or 0) between solvers
            if _regime(alt["pi_runs"]) != _regime(base["pi_runs"]):
                s["participation_flips"] += 1
            for k in HEAD_STATS:
                dk = abs(alt[k] - base[k])
                s["max_abs_diff_" + tag][k] = max(
                    s["max_abs_diff_" + tag][k], dk)
            if base["GER_se"] > 1e-4:     # skip degenerate (shut-down) cells
                z = abs(alt["GER"] - base["GER"]) / base["GER_se"]
                s["max_GER_diff_in_se_" + tag] = max(
                    s["max_GER_diff_in_se_" + tag], z)
    tot = dict(n_configs=0, n_runs=0, base_converged_runs=0.0,
               base_n_iter_sum=0.0, tight_converged_runs=0.0,
               smooth_converged_runs=0.0, participation_flips=0)
    for s in summ.values():
        for k in tot:
            tot[k] += s[k]
    for tag in ("tight", "smooth"):
        tot["max_abs_diff_" + tag] = {
            k: max(s["max_abs_diff_" + tag][k] for s in summ.values())
            for k in HEAD_STATS}
        tot["max_GER_diff_in_se_" + tag] = max(
            s["max_GER_diff_in_se_" + tag] for s in summ.values())
    for k in ("base_n_iter_max", "base_resid_max", "tight_resid_max",
              "smooth_n_iter_max", "smooth_resid_max"):
        tot[k] = max(s[k] for s in summ.values())
    summ["all"] = tot
    # the Experiment IV contrasts re-computed under each solver (CRN seeds)
    alt_contrasts, alt_contrasts_dro = {}, {}
    for tag, skw in (("tight", TIGHT), ("smooth", SMOOTH)):
        crn_alt = {k: replicate(w, br, sc, seed_cfg=sq, **skw)
                   for k, (w, br, sc) in EXP4_CELLS.items()}
        alt_contrasts[tag] = exp4_contrasts(crn_alt, crn_alt)
        crn_alt = {k: replicate(w, br, sc, seed_cfg=sq,
                                desk_reject_only=True, **skw)
                   for k, (w, br, sc) in EXP4_CELLS.items()}
        alt_contrasts_dro[tag] = exp4_contrasts(crn_alt, crn_alt)
    R["convergence"] = dict(
        settings=dict(baseline=dict(tol=1e-4, max_iter=400, damp=0.4),
                      tight=dict(TIGHT, damp=0.4),
                      smooth=dict(tol=SMOOTH["tol"],
                                  max_iter=SMOOTH["max_iter"], damp=0.4)),
        summary=summ, by_config=by_cfg, exp4_contrasts=alt_contrasts,
        exp4_dro_contrasts=alt_contrasts_dro)

    with open("results.json", "w") as f:
        json.dump(R, f, indent=1)
    print("results.json written")
    make_figures(R)


# ======================================================================
#  Figure generation (reads R; run "python final_run_v3_6.py --figures-only"
#  to rebuild all PNG figures from the archived results.json; Figure 3
#  re-solves one configuration at its first replication seed)
# ======================================================================

def _healthy_pool_q(p=P):
    """Reconstruct the equilibrium quality pool of the healthy baseline
    (Experiment II, beta_r = 0.40, first replication seed) for Figure 3."""
    seed = config_seed(0, 0.1, 0.40)
    out = run_config(0.1, 0.40, **ALIGNED, seed=seed)
    rng = np.random.default_rng(seed)
    sA = rng.uniform(ALIGNED["sA_lo"], ALIGNED["sA_hi"], p["N"])
    eps = rng.uniform(-p["sigma"], p["sigma"], p["N"])
    bA = sA
    bE = ALIGNED["sE"]
    h = np.maximum(0.0, 1 - p["lam"] * np.abs(bE - bA))
    e = np.minimum(1.0, out["omega"] * p["gA"] * p["gE"] * bA * bE * h
                   / p["A"])
    return np.clip(e + eps, 0.0, 1.0 + p["sigma"])


def make_figures(R, p=P):
    ks = sorted(R["exp1"], key=float)
    xs = [float(k) for k in ks]

    # ---- Figure 2 (figphase1.png): Experiment I sweep -----------------
    fig, ax = plt.subplots(figsize=(8.5, 5.3), dpi=300)
    ax.plot(xs, [R["exp1"][k]["desk_rate_h"] * 100 for k in ks],
            "o-", color="tab:blue", label="Desk-decision rate, $h$-type (%)")
    ax.plot(xs, [R["exp1"][k]["desk_rate_l"] * 100 for k in ks],
            "o:", color="tab:blue", mfc="white", ms=5,
            label="Desk-decision rate, $l$-type (%)")
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
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
              fontsize=9, frameon=False)   # below the axes: hides no curve
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

    fig, ax = plt.subplots(figsize=(8.5, 5.3), dpi=300)
    for br, col, ls in ((0.0, "tab:red", "--"), (0.40, "tab:blue", "-")):
        U = np.array([util(Q, br) for Q in Qg])
        Qopt = float(minimize_scalar(lambda Q: -util(Q, br),
                                     bounds=(0, 1), method="bounded").x)
        Uopt = util(Qopt, br)
        note = ("below reservation utility" if Uopt < 0 else "")
        lab = ("$\\beta_r=%.2f$  ($Q^*=%.2f$, $\\Pi_r^*=%+.2f$%s)"
               % (br, Qopt, Uopt, (": " + note) if note else ""))
        ax.plot(Qg, U, color=col, lw=2, ls=ls, label=lab)
        ax.plot(Qopt, Uopt, "o", color=col, ms=7)
    ax.axhline(0, color="gray", lw=1.2)
    ax.text(0.98, 0.01, "reservation utility $\\underline{u}_r = 0$", color="0.3",
            ha="right", va="bottom", transform=ax.get_yaxis_transform())
    ax.set_xlabel("Reviewer evaluative effort $Q$")
    ax.set_ylabel("Expected reviewer utility $E[\\Pi_r]$")
    ax.grid(ls="--", alpha=0.4)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig("figphase2.png")
    plt.close(fig)

    # ---- Figure 4 (figexp2.png): contract, acceptance rate, effort ----
    W = R["exp2_WJ"]
    brs = sorted(W, key=float)
    xb = [float(k) for k in brs]
    pis = [W[k]["pi"] for k in brs]
    Qs = [W[k]["Q"] for k in brs]
    gs = [W[k]["GER"] for k in brs]
    # smallest contract on the grid with full participation (beta_r* = 0.40)
    cross = next(x for x, v in zip(xb, pis) if v >= 1.0)
    fig, b = plt.subplots(figsize=(8.5, 5.3), dpi=300)
    b.axvspan(min(xb) - 0.02, cross, color="red", alpha=0.07)
    b.text(min(xb) + 0.005, 1.02, "review rationed ($\\pi^*<1$)",
           color="darkred", ha="left", va="bottom", fontsize=9)
    b.text(cross + 0.01, 1.02, "full participation", color="0.25",
           ha="left", va="bottom", fontsize=9)
    b.plot(xb, pis, "s-", color="tab:green",
           label="Acceptance rate of review invitations $\\pi^*$")
    b.plot(xb, gs, "^-", color="black", label="GER")
    b.plot(xb, Qs, "o-", color="tab:blue",
           label="Reviewer effort $Q^*$")
    b.set_xlim(min(xb) - 0.02, max(xb) + 0.02)
    b.set_ylim(0, 1.1)
    b.set_xlabel("Quality reward $\\beta_r$")
    b.set_ylabel("Rate / effort / GER")
    b.grid(ls="--", alpha=0.4)
    b.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig("figexp2.png")
    plt.close(fig)

    # ---- Figure 5 (figsimulation.png): alignment catalyst -------------
    cv = R["exp3_curves"]
    fig, ax = plt.subplots(figsize=(8.5, 5.3), dpi=300)
    for bA, col, ls in (("0.6", "tab:red", "-."), ("0.75", "tab:orange", "--"),
                        ("0.9", "tab:blue", "-")):
        ax.plot(cv["bE"], cv["curves"][bA], color=col, lw=2, ls=ls,
                label="$b_A = %s$" % bA)
    ax.set_xlabel("Editorial strictness $b_E$")
    ax.set_ylabel("Optimal author effort $e^*$")
    ax.grid(ls="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig("figsimulation.png")
    plt.close(fig)

    # ---- Figure 6 (figphase4_integration.png): radar + factorial bars -
    E4c = R["exp4_crn"]     # Panel A: common random numbers (the Status Quo
    #                         and the reviewer-only cell then coincide exactly)
    E4 = R["exp4"]          # Panel B: own seeds, as the GER column of Table 8
    names = ["status_quo", "partial", "optimal"]
    disp = ["Status Quo", "Reviewer-only reform", "Joint reform"]
    cols = ["tab:red", "tab:orange", "tab:green"]
    axes_lbl = ["$e^*$", "$Q^*$", "$\\omega$", "Pub. prob.", "GER"]
    keys = ["e_mean", "Q", "omega", "pub_prob", "GER"]
    # Status Quo: solid + circles; reviewer-only: dashed + squares, unfilled
    # (so that the overlap with the Status Quo stays visible); joint reform:
    # solid + triangles.
    styles = [dict(lw=2.8, marker="o", ms=6, fill=0.12),
              dict(lw=2.2, marker="s", ms=5.5, fill=0.0, dashes=(5, 3)),
              dict(lw=2.0, marker="^", ms=5.5, fill=0.15)]
    ang = np.linspace(0, 2 * np.pi, len(keys), endpoint=False) + np.pi / 2
    fig = plt.figure(figsize=(13.6, 5.8), dpi=300)
    ar = fig.add_subplot(121, polar=True)
    for n, d, c, st in zip(names, disp, cols, styles):
        vals = [E4c[n][k] for k in keys]
        line, = ar.plot(np.r_[ang, ang[0]], np.r_[vals, vals[0]], color=c,
                        lw=st["lw"], marker=st["marker"], ms=st["ms"],
                        label=d)
        if "dashes" in st:
            line.set_dashes(st["dashes"])
        if st["fill"] > 0:
            ar.fill(np.r_[ang, ang[0]], np.r_[vals, vals[0]], color=c,
                    alpha=st["fill"])
    ar.set_thetagrids(np.degrees(ang) % 360, axes_lbl)
    ar.set_rlim(0, 1)
    ar.set_rlabel_position(22.5)
    ar.set_title("(A) Performance profile of three corner states", pad=22)
    ar.legend(loc="lower left", bbox_to_anchor=(-0.16, -0.14), fontsize=9)

    # Panel B: the eight cells of the 2 x 2 x 2 factorial
    ab = fig.add_subplot(122)
    groups = ["None", "Reviewer\n(R)", "Editor\n(E)",
              "Editor + reviewer\n(E + R)"]
    mis = ["status_quo", "partial", "editor_only", "editor_reviewer"]
    ali = ["alignment_only", "alignment_reviewer", "alignment_editor",
           "optimal"]
    x = np.arange(len(groups))
    wbar = 0.37
    series = ((mis, -wbar / 2 - 0.01, "#a6761d", "//",
               "Misaligned standards"),
              (ali, +wbar / 2 + 0.01, "#1f77b4", "",
               "Aligned standards (A)"))
    for cells, off, col, hatch, lab in series:
        g = [E4[k]["GER"] for k in cells]
        se = [E4[k]["GER_se"] for k in cells]
        bars = ab.bar(x + off, g, wbar, color=col, edgecolor="white",
                      linewidth=1.0, hatch=hatch, yerr=se, capsize=3,
                      error_kw=dict(lw=1.0, ecolor="0.25"), label=lab)
        for r_, v in zip(bars, g):
            ab.text(r_.get_x() + r_.get_width() / 2, v + 0.018,
                    "%.3f" % v, ha="center", va="bottom", fontsize=9,
                    color="0.15")
    ab.annotate("Status Quo", xy=(x[0] - 0.03, 0.335),
                ha="right", va="bottom", fontsize=8.5, color="0.35")
    ab.annotate("Joint reform", xy=(x[3] + wbar / 2 + 0.01, 0.815),
                ha="center", va="bottom", fontsize=8.5, color="0.35")
    ab.set_xticks(x)
    ab.set_xticklabels(groups)
    ab.set_xlabel("Editor- and reviewer-side nodes reformed")
    ab.set_ylabel("Global Efficiency Ratio (GER)")
    ab.set_ylim(0, 0.92)
    ab.set_axisbelow(True)
    ab.grid(axis="y", ls="--", alpha=0.4)
    for sp in ("top", "right"):
        ab.spines[sp].set_visible(False)
    ab.legend(loc="upper left", fontsize=9, frameon=False)
    ab.set_title("(B) GER in the eight cells of the $2\\times2\\times2$ "
                 "factorial (own seeds; mean $\\pm$ s.e.)")
    fig.subplots_adjust(left=0.05, right=0.98, top=0.86, bottom=0.13,
                        wspace=0.24)
    fig.savefig("figphase4_integration.png")
    plt.close(fig)

    # ---- Figure B.1 (figsens.png): participation threshold ------------
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.0, 4.6), dpi=300)
    k0s = sorted(R["sens_bstar_k0"], key=float)
    a.plot([float(k) for k in k0s],
           [R["sens_bstar_k0"][k] for k in k0s], "o-", color="tab:blue")
    a.axhline(0.40, color="gray", ls="--", lw=1.2)
    a.text(3.0, 0.42, "contract of Exp. II ($\\beta_r = 0.40$)",
           color="gray", fontsize=9)
    a.set_title("(A) $\\beta_r^*$ rises with fatigue")
    a.set_xlabel("Reviewer evaluation friction $k_0$")
    a.set_ylabel("Minimal reward for full participation $\\beta_r^*$")
    a.grid(ls="--", alpha=0.4)
    wrs = sorted(R["sens_bstar_wr"], key=float)
    b.plot([float(k) for k in wrs],
           [R["sens_bstar_wr"][k] for k in wrs], "s-", color="tab:green")
    b.set_title("(B) $\\beta_r^*$ falls with the fixed reward")
    b.set_xlabel("Fixed social reward $w_r$")
    b.set_ylabel("Minimal reward for full participation $\\beta_r^*$")
    b.grid(ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig("figsens.png")
    plt.close(fig)
    print("figures written")


if __name__ == "__main__":
    main()
