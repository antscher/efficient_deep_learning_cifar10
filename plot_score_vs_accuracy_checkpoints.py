import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


RESNET_BASE_STATS = {
    "resnet10": {"w": 4_903_242.0, "f": 253_432_832.0},
    "resnet12": {"w": 4_977_226.0, "f": 328_930_304.0},
    "resnet14": {"w": 5_272_650.0, "f": 404_427_776.0},
    "resnet18": {"w": 11_173_962.0, "f": 555_422_720.0},
    "resnet34": {"w": 21_282_122.0, "f": 1_159_402_496.0},
    "resnet50": {"w": 23_520_842.0, "f": 1_297_829_888.0},
    "resnet101": {"w": 42_512_970.0, "f": 2_509_983_744.0},
    "resnet152": {"w": 58_156_618.0, "f": 3_722_137_600.0},
}
DEFAULT_BASE_RESNET = "resnet18"


def compute_score(w, f, p_s=0.0, p_u=0.0, q_w=32.0, q_a=32.0):
    """
    score = ([1 - (p_s + p_u)] * (q_w/32) * w) / 5.6e6
          + ((1 - p_s) * (max(q_w, q_a)/32) * f) / 2.8e8
    """
    term1 = (1.0 - (p_s + p_u)) * (q_w / 32.0) * float(w) / 5.6e6
    term2 = (1.0 - p_s) * (max(q_w, q_a) / 32.0) * float(f) / 2.8e8
    return term1 + term2


def infer_pruning_ratios(filename):
    """Infer p_s and p_u from checkpoint filename."""
    name = filename.lower()

    m = re.search(r"struct(\d+)_unstruct(\d+)", name)
    if m:
        p_s = int(m.group(1)) / 100.0
        # Unstructured pruning is applied on the remaining weights after structured pruning.
        # Example: struct70_unstruct70 => p_u = 0.70 * (1 - 0.70) = 0.21
        p_u = (int(m.group(2)) / 100.0) * (1.0 - p_s)
        return p_s, p_u

    m = re.search(r"structured_(\d+)", name)
    if m:
        p_s = int(m.group(1)) / 100.0
        return p_s, 0.0

    m = re.search(r"pruned_(\d+)", name)
    if m:
        p_u = int(m.group(1)) / 100.0
        return 0.0, p_u

    return 0.0, 0.0


def infer_base_resnet(filename):
    name = filename.lower()
    m = re.search(r"(resnet(?:10|12|14|18|34|50|101|152))", name)
    if not m:
        return DEFAULT_BASE_RESNET
    arch = m.group(1)
    return arch if arch in RESNET_BASE_STATS else DEFAULT_BASE_RESNET


def resolve_w_f(filename, w_override=None, f_override=None):
    base_arch = infer_base_resnet(filename)
    base_stats = RESNET_BASE_STATS[base_arch]
    w = float(w_override) if w_override is not None else float(base_stats["w"])
    f = float(f_override) if f_override is not None else float(base_stats["f"])
    return w, f


def try_read_accuracy_from_checkpoint(path):
    """
    Try extracting accuracy from checkpoint metadata.
    Returns float or None.
    """
    try:
        import torch
    except Exception:
        return None

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Backward compatibility if torch version does not support weights_only.
        ckpt = torch.load(path, map_location="cpu")
    except Exception:
        return None

    if not isinstance(ckpt, dict):
        return None

    candidate_keys = [
        "acc",
        "accuracy",
        "test_acc",
        "best_acc",
        "val_acc",
    ]
    for key in candidate_keys:
        if key in ckpt:
            value = ckpt[key]
            if isinstance(value, (int, float)):
                return float(value)
    return None


def infer_family(filename):
    name = filename.lower()
    if "struct" in name and "unstruct" in name:
        return "structured+unstructured"
    if "structured_" in name:
        return "structured"
    if "pruned_" in name:
        return "unstructured"
    if "mixup" in name:
        return "mixup"
    if "transform" in name:
        return "transform"
    if "cos" in name:
        return "cosine"
    return "baseline"


def compact_label(filename):
    name = filename.replace(".pth", "")
    low = name.lower()
    q_match = re.search(r"_q(\d+)$", low)
    q_suffix = f" q{q_match.group(1)}" if q_match else ""

    m = re.search(r"struct(\d+)_unstruct(\d+)(_full_model)?", low)
    if m:
        suffix = " full" if m.group(3) else ""
        return f"s{m.group(1)}+u{m.group(2)}{suffix}{q_suffix}"

    m = re.search(r"structured_(\d+)", low)
    if m:
        return f"structured_{m.group(1)}"

    m = re.search(r"pruned_(\d+)_pipeline", low)
    if m:
        return f"pruned_{m.group(1)}_pipeline"

    m = re.search(r"pruned_(\d+)", low)
    if m:
        return f"pruned_{m.group(1)}"

    return name.replace("ResNet18_", "").replace("ResNet18", "resnet18").replace("_", " ")


def plot_by_family(rows_with_acc, out_path, no_annotations=False, title_suffix=""):
    family_colors = {
        "baseline": "tab:blue",
        "mixup": "tab:green",
        "transform": "tab:olive",
        "cosine": "tab:cyan",
        "structured": "tab:orange",
        "unstructured": "tab:red",
        "structured+unstructured": "tab:purple",
    }

    plt.figure(figsize=(14, 8))
    families = sorted({r["family"] for r in rows_with_acc})
    texts = []
    fallback_offsets = [(4, 4), (6, -6), (-10, 6), (-12, -8), (10, 10)]
    for fam in families:
        fam_rows = [r for r in rows_with_acc if r["family"] == fam]
        x = [r["score"] for r in fam_rows]
        y = [r["accuracy"] for r in fam_rows]
        labels = [compact_label(r["checkpoint"]) for r in fam_rows]
        color = family_colors.get(fam, "gray")
        plt.scatter(x, y, s=80, alpha=0.9, color=color, edgecolor="white", linewidth=0.8, label=fam)
        if not no_annotations:
            for i, (xi, yi, label) in enumerate(zip(x, y, labels)):
                dx, dy = fallback_offsets[i % len(fallback_offsets)]
                txt = plt.annotate(
                    label,
                    (xi, yi),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                )
                txt.set_path_effects([pe.withStroke(linewidth=2.0, foreground="white")])
                texts.append(txt)

    plt.xlabel("Score")
    plt.ylabel("Accuracy (%)")
    title = "Score vs Accuracy for checkpoints"
    if title_suffix:
        title += f" ({title_suffix})"
    plt.title(title)
    plt.grid(True, alpha=0.25)
    if texts:
        try:
            from adjustText import adjust_text
            adjust_text(
                texts,
                only_move={"text": "xy"},
                arrowprops=dict(arrowstyle="-", color="0.45", lw=0.6),
            )
        except Exception:
            pass

    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)


def main():
    parser = argparse.ArgumentParser(description="Score vs accuracy for all checkpoints")
    parser.add_argument("--checkpoints-dir", default="checkpoints", help="Directory with .pth files")
    parser.add_argument("--output-plot", default="results/score_vs_accuracy_checkpoints.png", help="Output plot path")
    parser.add_argument("--output-csv", default="results/score_vs_accuracy_checkpoints.csv", help="Output CSV path")
    parser.add_argument("--q-w", type=float, default=32.0, help="Weight quantization bits")
    parser.add_argument("--q-a", type=float, default=32.0, help="Activation quantization bits")
    parser.add_argument("--w", type=float, default=None, help="Reference number of weights (override auto from base ResNet)")
    parser.add_argument("--f", type=float, default=None, help="Reference number of MAC ops (override auto from base ResNet)")
    parser.add_argument("--zoom-threshold", type=float, default=0.85, help="Second plot uses accuracy > threshold")
    parser.add_argument("--no-annotations", action="store_true", help="Disable labels on points")
    args = parser.parse_args()

    checkpoints_dir = Path(args.checkpoints_dir)
    output_plot = Path(args.output_plot)
    output_csv = Path(args.output_csv)
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Manual accuracies from:
    # wandb_export_2026-03-02T14_25_21.278+01_00.csv
    # - Column A: Name
    # - Column AE: test_acc
    # - Excluding rows where State (column B) is "killed" or "running"
    accuracy_manual = {
        "ResNet18_pruned_50_pipeline.pth": 94.38,
        "ResNet18_pruned_60_pipeline.pth": 94.62,
        "ResNet18_pruned_70_pipeline.pth": 94.47,
        "ResNet18_pruned_80_pipeline.pth": 94.42,
        "ResNet18_struct20_unstruct80.pth": 93.91,
        "ResNet18_struct30_unstruct70.pth": 93.81,
        "ResNet18_struct40_unstruct60.pth": 94.09,
        "ResNet18_struct50_unstruct50.pth": 93.84,
        "ResNet18_struct60_unstruct40.pth": 93.54,
        "ResNet18_struct70_unstruct30.pth": 93.05,
        "ResNet18_struct80_unstruct20.pth": 92.62,
        "ResNet18_structured_30.pth": 93.98,
        "ResNet18_structured_50.pth": 93.51,
        "ResNet18_structured_60.pth": 93.36,
        "ResNet18_structured_70.pth": 93.31,
        "ResNet18_structured_80.pth": 92.51,
        "ResNet18_structured_90.pth": 90.54,
        "ResNet18_pruned_60.pth": 94.6,
        "ResNet18_pruned_70.pth": 94.41,
        "ResNet18_pruned_80.pth": 94.27,
        "ResNet18_pruned_90.pth": 93.5,
        "ResNet18_pruned_95.pth": 92.0,
        "ResNet18_pruned_99.pth": 67.24,
        "ResNet18_mixup_cos.pth": 94.64,
        "ResNet18_transform_cos.pth": 93.78,
        "ResNet18_cos.pth": 88.32,
        "ResNet18_mixup.pth": 93.47,
        "ResNet18_transform.pth": 92.52,
        "ResNet18.pth": 88.08,
    }
    # Extra points to compare 32b (before quantization) vs 16b (after quantization).
    # They are injected even if corresponding .pth files are absent in checkpoints_dir.
    extra_quant_points = [
        {
            "base_checkpoint": "ResNet18_struct60_unstruct60.pth",
            "acc_q32": 92.07,
            "acc_q16": 92.07,
        },
        {
            "base_checkpoint": "ResNet18_struct70_unstruct70.pth",
            "acc_q32": 91.33,
            "acc_q16": 91.31,
        },
    ]

    rows = []
    for ckpt_path in sorted(checkpoints_dir.glob("*.pth")):
        p_s, p_u = infer_pruning_ratios(ckpt_path.name)
        w_ref, f_ref = resolve_w_f(ckpt_path.name, w_override=args.w, f_override=args.f)
        score = compute_score(w=w_ref, f=f_ref, p_s=p_s, p_u=p_u, q_w=args.q_w, q_a=args.q_a)

        accuracy = try_read_accuracy_from_checkpoint(ckpt_path)
        if accuracy is None:
            accuracy = accuracy_manual.get(ckpt_path.name)

        rows.append(
            {
                "checkpoint": ckpt_path.name,
                "family": infer_family(ckpt_path.name),
                "p_s": p_s,
                "p_u": p_u,
                "q_w": args.q_w,
                "q_a": args.q_a,
                "w": w_ref,
                "f": f_ref,
                "score": score,
                "accuracy": accuracy,
            }
        )

    for item in extra_quant_points:
        base_name = item["base_checkpoint"]
        p_s, p_u = infer_pruning_ratios(base_name)
        w_ref, f_ref = resolve_w_f(base_name, w_override=args.w, f_override=args.f)
        for bits, acc in ((32.0, item["acc_q32"]), (16.0, item["acc_q16"])):
            rows.append(
                {
                    "checkpoint": base_name.replace(".pth", f"_q{int(bits)}.pth"),
                    "family": infer_family(base_name),
                    "p_s": p_s,
                    "p_u": p_u,
                    "q_w": bits,
                    "q_a": bits,
                    "w": w_ref,
                    "f": f_ref,
                    "score": compute_score(w=w_ref, f=f_ref, p_s=p_s, p_u=p_u, q_w=bits, q_a=bits),
                    "accuracy": acc,
                }
            )

    print(rows)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["checkpoint", "family", "p_s", "p_u", "q_w", "q_a", "w", "f", "score", "accuracy"],
        )
        writer.writeheader()
        writer.writerows(rows)

    rows_with_acc = [r for r in rows if r["accuracy"] is not None and not math.isnan(r["accuracy"])]
    threshold_tag = str(args.zoom_threshold).replace(".", "_")
    output_plot_zoom = output_plot.with_name(f"{output_plot.stem}_gt_{threshold_tag}{output_plot.suffix}")

    if rows_with_acc:
        plot_by_family(rows_with_acc, output_plot, no_annotations=args.no_annotations)
        rows_zoom = [r for r in rows_with_acc if r["accuracy"] > args.zoom_threshold]
        if rows_zoom:
            plot_by_family(
                rows_zoom,
                output_plot_zoom,
                no_annotations=args.no_annotations,
                title_suffix=f"accuracy > {args.zoom_threshold}",
            )
        else:
            print(f"No points with accuracy > {args.zoom_threshold}, zoom plot not generated.")
    else:
        print("No accuracy found automatically. Fill 'accuracy_manual' in the script, then rerun.")

    print(f"Saved CSV: {output_csv}")
    print(f"Saved plot: {output_plot}")
    if rows_with_acc and any(r["accuracy"] > args.zoom_threshold for r in rows_with_acc):
        print(f"Saved zoom plot: {output_plot_zoom}")
    print(f"Total checkpoints found: {len(rows)}")
    print(f"Checkpoints with accuracy: {len(rows_with_acc)}")


if __name__ == "__main__":
    main()
