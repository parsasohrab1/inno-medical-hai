"""Hospital-acquired infection cluster detection and antibiogram.

Lab prototype on SYNTHETIC data only (TRL 4). Alerts are for the infection-control committee to investigate.
"""
import math
import numpy as np

WARDS = ["ICU", "SURG", "MED", "ORTHO", "PED", "NICU"]
ORGS = ["E.coli", "K.pneumoniae", "S.aureus", "P.aeruginosa"]
ABX = ["ceftriaxone", "ciprofloxacin", "meropenem"]
DAYS = 365
LAMBDA = 0.2  # isolates per ward-organism-day (baseline)
P_THRESHOLD = 1e-4

CRITERIA = [
    ("cluster_recall", ">=", 0.80, "injected clusters detected"),
    ("false_alerts_per_year", "<=", 3.0, "alerts not overlapping an injected cluster, per simulated year"),
    ("antibiogram_mae", "<=", 0.08, "mean abs. error of ward resistance rates vs truth"),
]


def poisson_sf(k, mu):
    """P(X >= k) for X~Poisson(mu)."""
    if k <= 0:
        return 1.0
    term = math.exp(-mu); cdf = term
    for i in range(1, k):
        term *= mu / i; cdf += term
    return max(1.0 - cdf, 1e-300) if cdf < 1 else 1e-300


def simulate(seed=0):
    rng = np.random.default_rng(seed)
    counts = np.zeros((len(WARDS), len(ORGS), DAYS), int)
    clusters = []
    for _ in range(5):
        w, o = int(rng.integers(0, len(WARDS))), int(rng.integers(0, len(ORGS)))
        start = int(rng.integers(120, DAYS - 20)); dur = int(rng.integers(10, 15))
        if any(c["ward"] == w and c["org"] == o for c in clusters):
            continue
        clusters.append({"ward": w, "org": o, "start": start, "end": start + dur - 1, "mult": float(rng.uniform(5, 8))})
    lam = np.full(counts.shape, LAMBDA)
    for c in clusters:
        lam[c["ward"], c["org"], c["start"]:c["end"] + 1] *= c["mult"]
    counts = rng.poisson(lam)
    true_res = rng.uniform(0.1, 0.6, (len(WARDS), len(ORGS), len(ABX)))
    isolates = []
    for w in range(len(WARDS)):
        for o in range(len(ORGS)):
            for d in range(DAYS):
                for _ in range(counts[w, o, d]):
                    isolates.append({"ward": w, "org": o, "day": d, "res": [bool(rng.random() < true_res[w, o, a]) for a in range(len(ABX))]})
    return counts, clusters, isolates, true_res


def scan(counts_wo, windows=(5, 7, 14), baseline_days=90, start=100, p_thr=P_THRESHOLD):
    """Sliding-window Poisson scan for one ward-organism series. Returns list of (start,end,p)."""
    hits = []
    for t in range(start, len(counts_wo)):
        for w in windows:
            if t - w + 1 < baseline_days // 3:
                continue
            lo = max(0, t - w + 1 - baseline_days); hi = t - w + 1
            if hi - lo < 30:
                continue
            mu = counts_wo[lo:hi].mean() * w
            mu = max(mu, 0.05 * w)
            k = int(counts_wo[t - w + 1:t + 1].sum())
            p = poisson_sf(k, mu)
            if p < p_thr:
                hits.append((t - w + 1, t, p))
    return hits


def merge(hits, gap=3):
    out = []
    for s, e, p in sorted(hits):
        if out and s <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], e), min(out[-1][2], p))
        else:
            out.append((s, e, p))
    return out


def detect_all(counts):
    alerts = []
    for w in range(counts.shape[0]):
        for o in range(counts.shape[1]):
            for s, e, p in merge(scan(counts[w, o])):
                alerts.append({"ward": WARDS[w], "org": ORGS[o], "start": s, "end": e, "p": p})
    return alerts


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0, c - h), min(1, c + h))


def antibiogram(isolates, min_n=10):
    out = {}
    for w in range(len(WARDS)):
        for o in range(len(ORGS)):
            sub = [i for i in isolates if i["ward"] == w and i["org"] == o]
            for a in range(len(ABX)):
                n = len(sub); k = sum(i["res"][a] for i in sub)
                out[(w, o, a)] = {"n": n, "rate": k / n if n else None, "ci": wilson(k, n), "reportable": n >= min_n}
    return out


def committee_report(alerts):
    if not alerts:
        return "No clusters detected this period."
    return "\n".join("Cluster: %s / %s days %d-%d (p=%.1e)" % (a["ward"], a["org"], a["start"], a["end"], a["p"]) for a in alerts)


def evaluate(seed=0):
    counts, clusters, isolates, true_res = simulate(seed)
    alerts = detect_all(counts)
    names = {(WARDS[c["ward"]], ORGS[c["org"]]): c for c in clusters}

    def overlaps(a):
        c = names.get((a["ward"], a["org"]))
        return bool(c and a["start"] <= c["end"] + 5 and a["end"] >= c["start"] - 5)
    rec = sum(any(a["ward"] == WARDS[c["ward"]] and a["org"] == ORGS[c["org"]] and a["start"] <= c["end"] + 5 and a["end"] >= c["start"] for a in alerts) for c in clusters)
    ab = antibiogram(isolates)
    err = [abs(v["rate"] - true_res[k]) for k, v in ab.items() if v["rate"] is not None and v["n"] >= 30]
    return {"cluster_recall": rec / len(clusters), "false_alerts_per_year": float(sum(not overlaps(a) for a in alerts)),
            "antibiogram_mae": float(np.mean(err)), "n_clusters": len(clusters)}


def generate(seed=0):
    counts, clusters, isolates, _ = simulate(seed)
    return {"isolates": [dict(ward=WARDS[i["ward"]], organism=ORGS[i["org"]], day=i["day"],
                              **{"R_" + ABX[a]: int(i["res"][a]) for a in range(len(ABX))}) for i in isolates[:6000]],
            "injected_clusters": [dict(ward=WARDS[c["ward"]], organism=ORGS[c["org"]], start=c["start"], end=c["end"], multiplier=round(c["mult"], 2)) for c in clusters]}


def demo(seed=0):
    counts, *_ = simulate(seed)
    return committee_report(detect_all(counts))
