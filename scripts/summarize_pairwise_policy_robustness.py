"""Summarize pairwise policy margins from already computed headline records."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = {
    "visdrone": ("visdrone_sahi640", "visdrone_baseline640"),
    "uavdt": ("uavdt_tiling", "uavdt_baseline640"),
}


def summarize(metric, frame, value):
    rows = []
    for source, (winner, runner_up) in CANDIDATES.items():
        subset = frame.loc[frame["source"].eq(source)].copy()
        pivot = subset.pivot_table(
            index=["mode", "threshold", "small_object_policy"],
            columns="candidate", values=value, aggfunc="first",
        ).dropna(subset=[winner, runner_up])
        margins = pivot[winner] - pivot[runner_up]
        reference = margins[[idx[2] == "include_all" for idx in margins.index]].iloc[0]
        rows.append({
            "source": source, "metric": metric, "winner": winner,
            "runner_up": runner_up, "reference_margin": reference,
            "differential_radius": (margins - reference).abs().max(),
            "minimum_margin": margins.min(), "stable": bool((margins > 0).all()),
            "settings": len(margins),
        })
    return rows


def main():
    ap = pd.read_csv(ROOT / "outputs/ap_headline/ap_headline_surface.csv").rename(columns={"AP50": "value"})
    f1_records = []
    for mode, path, threshold in [
        ("absolute", ROOT / "outputs/coverage_corrected_grid/absolute/metrics_long.parquet", 24.0),
        ("normalized", ROOT / "outputs/coverage_corrected_grid/normalized/metrics_long.parquet", 0.015),
    ]:
        table = pd.read_parquet(path)
        table = table.loc[(table.confidence == 0.25) & (table.iou == 0.25) & (table.matching == "greedy_iou")].copy()
        table["mode"], table["threshold"] = mode, threshold
        f1_records.append(table[["source", "candidate", "mode", "threshold", "small_object_policy", "f1"]].rename(columns={"f1": "value"}))
    f1 = pd.concat(f1_records, ignore_index=True)
    result = pd.DataFrame(summarize("AP50", ap, "value") + summarize("F1", f1, "value"))
    out = ROOT / "outputs/statistics/pairwise_policy_robustness.csv"
    result.to_csv(out, index=False)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
