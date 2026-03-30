"""
scripts/generate_thesis_figures.py — Publication-quality figures for thesis.

Generates vector (PDF) and raster (PNG 300 dpi) versions of all key figures.
Uses a consistent style suitable for IEEE / academic publication.

Usage:
    python scripts/generate_thesis_figures.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

# ── Publication style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
})

DPDK_COLOR = "#2171b5"
USR_COLOR  = "#cb181d"
OUT_DIR = Path("reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}")
    print(f"  Saved {name}.pdf / .png")
    plt.close(fig)


# ── Load data ─────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/processed/features.csv")
merged = pd.read_csv("data/interim/merged.csv")

dpdk = df[df["is_dpdk"] == 1]
usr  = df[df["is_dpdk"] == 0]

targets = ["power_watts", "net_power_watts", "sec_total", "sec_net"]
features_only = [c for c in df.columns if c not in targets]


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Power vs Throughput scatter (the core relationship)
# ═══════════════════════════════════════════════════════════════════════════
print("\nFigure 1: Power vs Throughput...")
fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), sharey=False)

for ax, (label, sub, color) in zip(axes,
        [("SD-Core DPDK", dpdk, DPDK_COLOR), ("USR-UPF", usr, USR_COLOR)]):
    ax.scatter(sub["throughput_gbps"], sub["power_watts"],
               alpha=0.12, s=6, color=color, rasterized=True)
    ax.set_xlabel("Throughput (Gbps)")
    ax.set_ylabel("Power consumption (W)")
    ax.set_title(label, fontweight="bold")

axes[0].set_ylim(0.80, 0.87)
axes[1].set_ylim(-0.1, 5.0)
fig.tight_layout()
save(fig, "fig1_power_vs_throughput")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Variant profile comparison (4-panel bar chart)
# ═══════════════════════════════════════════════════════════════════════════
print("Figure 2: Variant profiles...")
tp_bins   = [0, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.1]
tp_labels = ["<0.01", "0.01–0.1", "0.1–0.5", "0.5–1.0", "1.0–2.0", "2.0–3.0", "3.0–5.0"]

records = []
for label, sub, color in [("SD-Core DPDK", dpdk, DPDK_COLOR), ("USR-UPF", usr, USR_COLOR)]:
    sub = sub.copy()
    sub["tp_bin"] = pd.cut(sub["throughput_gbps"], bins=tp_bins, labels=tp_labels)
    for b in tp_labels:
        bsub = sub[sub["tp_bin"] == b]
        if len(bsub) == 0:
            continue
        records.append({
            "Variant": label, "bin": b,
            "n": len(bsub),
            "Power (W)": bsub["power_watts"].mean(),
            "Power std": bsub["power_watts"].std(),
            "Loss (%)": 100 * (bsub["gtpu_packets_dn__packets_lost_delta"] > 0).mean(),
            "DL delay (ms)": bsub["downlink_one_way_delay_distribution__weighted_mean_delay_us"].mean() / 1000,
            "CPU (%)": bsub["cpu_pct"].mean(),
        })
binned = pd.DataFrame(records)

# Determine full ordered list of bins that have any data (preserves order)
bins_with_data = [b for b in tp_labels if (binned["bin"] == b).any()]
n_bins = len(bins_with_data)
bin_x = {b: i for i, b in enumerate(bins_with_data)}

fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
metrics = [("Power (W)", "Power (W)", "Power std"),
           ("CPU (%)", "CPU (%)", None),
           ("Loss (%)", "Loss (%)", None),
           ("DL delay (ms)", "DL delay (ms)", None)]

for ax, (title, col, err_col) in zip(axes.flat, metrics):
    for i, (vlabel, color) in enumerate([("SD-Core DPDK", DPDK_COLOR), ("USR-UPF", USR_COLOR)]):
        d = binned[binned["Variant"] == vlabel]
        x_pos = np.array([bin_x[b] for b in d["bin"]])
        kwargs = {}
        if err_col:
            kwargs["yerr"] = d[err_col].values
            kwargs["capsize"] = 2
        ax.bar(x_pos + (-0.18 + i * 0.36), d[col].values, 0.32, label=vlabel,
               color=color, alpha=0.85, **kwargs)
    ax.set_xticks(np.arange(n_bins))
    ax.set_xticklabels(bins_with_data, rotation=45, ha="right", fontsize=7.5)
    ax.set_xlabel("Throughput (Gbps)")
    ax.set_ylabel(title)
    ax.legend(fontsize=7)

axes[0, 0].set_title("(a) Power consumption", fontweight="bold")
axes[0, 1].set_title("(b) CPU utilisation", fontweight="bold")
axes[1, 0].set_title("(c) Packet loss", fontweight="bold")
axes[1, 1].set_title("(d) Downlink delay", fontweight="bold")

fig.tight_layout()
save(fig, "fig2_variant_profiles")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: USR saturation transition (4-panel line plot)
# ═══════════════════════════════════════════════════════════════════════════
print("Figure 3: USR saturation...")
usr_s = usr.sort_values("throughput_gbps").copy()
fine_bins = np.arange(0, 2.6, 0.05)
usr_s["tp_fine"] = pd.cut(usr_s["throughput_gbps"], bins=fine_bins)

sat = usr_s.groupby("tp_fine", observed=True).agg(
    n=("power_watts", "count"),
    loss_frac=("gtpu_packets_dn__packets_lost_delta", lambda x: (x > 0).mean()),
    loss_mean=("gtpu_packets_dn__packets_lost_delta", lambda x: x[x > 0].mean() if (x > 0).any() else 0),
    dl_delay=("downlink_one_way_delay_distribution__weighted_mean_delay_us", "mean"),
    power=("power_watts", "mean"),
    cpu=("cpu_pct", "mean"),
).reset_index()
sat["tp_mid"] = sat["tp_fine"].apply(lambda x: x.mid)
sat = sat[sat["n"] >= 5]

onset_row = sat[sat["loss_frac"] > 0.05]
onset_tp = onset_row["tp_mid"].iloc[0] if len(onset_row) else None

fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))

ax = axes[0, 0]
ax.plot(sat["tp_mid"], sat["loss_frac"] * 100, "o-", color=USR_COLOR, markersize=3)
if onset_tp:
    ax.axvline(onset_tp, ls="--", color="grey", alpha=0.6, lw=0.8,
               label=f"Onset $\\approx${onset_tp:.2f} Gbps")
ax.set_ylabel("Samples with loss (%)")
ax.set_title("(a) Packet loss incidence", fontweight="bold")
ax.set_ylim(-5, 105)
ax.legend(fontsize=8)

ax = axes[0, 1]
ax.plot(sat["tp_mid"], sat["loss_mean"], "s-", color=USR_COLOR, markersize=3)
if onset_tp:
    ax.axvline(onset_tp, ls="--", color="grey", alpha=0.6, lw=0.8)
ax.set_ylabel("Packets lost / interval")
ax.set_title("(b) Loss magnitude", fontweight="bold")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e3:.0f}k" if x >= 1000 else f"{x:.0f}"))

ax = axes[1, 0]
ax.plot(sat["tp_mid"], sat["dl_delay"] / 1000, "^-", color=USR_COLOR, markersize=3)
if onset_tp:
    ax.axvline(onset_tp, ls="--", color="grey", alpha=0.6, lw=0.8)
ax.set_ylabel("Downlink delay (ms)")
ax.set_title("(c) Delay degradation", fontweight="bold")

ax = axes[1, 1]
ax2 = ax.twinx()
ln1 = ax.plot(sat["tp_mid"], sat["power"], "o-", color=USR_COLOR, markersize=3, label="Power (W)")
ln2 = ax2.plot(sat["tp_mid"], sat["cpu"], "s-", color="#6a51a3", markersize=3, label="CPU (%)")
if onset_tp:
    ax.axvline(onset_tp, ls="--", color="grey", alpha=0.6, lw=0.8)
ax.set_ylabel("Power (W)", color=USR_COLOR)
ax2.set_ylabel("CPU (%)", color="#6a51a3")
ax.set_title("(d) Power and CPU", fontweight="bold")
lines = ln1 + ln2
ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="center right")

for a in axes.flat:
    a.set_xlabel("Throughput (Gbps)")
fig.suptitle("USR-UPF Saturation Transition", fontsize=12, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "fig3_oai_saturation")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Temporal power traces (sample runs)
# ═══════════════════════════════════════════════════════════════════════════
print("Figure 4: Temporal traces...")

dpdk_merged = merged[merged["dataplane"] == "dpdk"]
usr_merged  = merged[merged["dataplane"] == "userspaceapplication"]

def pick_high_tp_run(subset):
    best, best_tp = None, 0
    for r in subset["run_dir"].unique()[:30]:
        tp = subset[subset["run_dir"] == r]["gtpu_kbitss_dn__kbits_rx_s"].max()
        if tp > best_tp:
            best, best_tp = r, tp
    return best

fig, axes = plt.subplots(2, 1, figsize=(7, 4.5), sharex=False)

for ax, (label, subset, color) in zip(axes,
        [("SD-Core DPDK", dpdk_merged, DPDK_COLOR),
         ("USR-UPF", usr_merged, USR_COLOR)]):
    run_id = pick_high_tp_run(subset)
    if run_id is None:
        continue
    rdf = subset[subset["run_dir"] == run_id].copy()
    rdf["t_sec"] = (rdf["Timestamp epoch ms"] - rdf["Timestamp epoch ms"].iloc[0]) / 1000

    ax_pw = ax.twinx()
    ax.plot(rdf["t_sec"], rdf["gtpu_kbitss_dn__kbits_rx_s"] / 1e6,
            color=DPDK_COLOR, lw=1.2, label="Throughput (Gbps)")
    ax_pw.plot(rdf["t_sec"], rdf["power_microwatts"] / 1e6,
               color=USR_COLOR, lw=1.2, label="Power (W)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Throughput (Gbps)", color=DPDK_COLOR)
    ax_pw.set_ylabel("Power (W)", color=USR_COLOR)
    ax.set_title(f"{label}", fontweight="bold")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_pw.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

fig.tight_layout()
save(fig, "fig4_temporal_traces")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: SEC paradox — total vs net at high throughput
# ═══════════════════════════════════════════════════════════════════════════
print("Figure 5: SEC paradox...")
fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

for ax, (sec_col, title) in zip(axes,
        [("sec_total", "SEC$_{\\mathrm{total}}$ (W/Gbps)"),
         ("sec_net",   "SEC$_{\\mathrm{net}}$ (W/Gbps)")]):
    for label, sub, color in [("SD-Core DPDK", dpdk, DPDK_COLOR), ("USR-UPF", usr, USR_COLOR)]:
        valid = sub[sub["throughput_gbps"] > 0.01].dropna(subset=[sec_col])
        # Clip extreme outliers for visibility
        clip_val = valid[sec_col].quantile(0.98)
        valid = valid[valid[sec_col] <= clip_val]
        ax.scatter(valid["throughput_gbps"], valid[sec_col],
                   alpha=0.12, s=6, color=color, label=label, rasterized=True)
    ax.set_xlabel("Throughput (Gbps)")
    ax.set_ylabel(title)
    ax.legend(fontsize=8)

axes[0].set_title("(a) Total SEC", fontweight="bold")
axes[1].set_title("(b) Net SEC", fontweight="bold")
fig.tight_layout()
save(fig, "fig5_sec_paradox")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6: Correlation heatmap (compact, per-variant)
# ═══════════════════════════════════════════════════════════════════════════
print("Figure 6: Correlation heatmaps...")
key_cols = [
    "throughput_gbps", "cpu_pct", "packet_rate_pps",
    "gtpu_kbitss_dn__kbits_rx_s", "gtpu_kbitss_ngran__gtpu_kbits_rx_s",
    "one_way_delay_average_dn__average", "delay_variation_jitter_average_dn__avg_us",
    "downlink_one_way_delay_distribution__high_delay_frac",
    "l2l3_overhead_ratio", "avg_packet_size_bytes",
    "power_watts", "net_power_watts",
]
short_labels = [
    "Throughput", "CPU", "Pkt rate", "DN Rx kbps", "NGRAN Rx kbps",
    "DL delay", "DN jitter", "High delay frac",
    "L2/L3 ratio", "Pkt size", "Power", "Net power",
]
key_cols = [c for c in key_cols if c in df.columns]
short_labels = short_labels[:len(key_cols)]

# Use manual axes layout to place colorbar cleanly outside both heatmaps
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
for ax, (label, sub) in zip(axes, [("SD-Core DPDK", dpdk), ("USR-UPF", usr)]):
    corr = sub[key_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    im = ax.imshow(np.ma.array(corr.values, mask=mask), cmap="RdBu_r",
                   vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(short_labels)))
    ax.set_yticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticklabels(short_labels, fontsize=6.5)
    ax.set_title(label, fontweight="bold")

# Reserve right margin for colorbar, then add it in a dedicated axes
fig.subplots_adjust(left=0.12, right=0.86, bottom=0.25, top=0.92, wspace=0.35)
cbar_ax = fig.add_axes([0.88, 0.25, 0.018, 0.67])
fig.colorbar(im, cax=cbar_ax, label="Pearson r")
save(fig, "fig6_correlation_heatmaps")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7: USR per-run saturation scatter
# ═══════════════════════════════════════════════════════════════════════════
print("Figure 7: USR per-run saturation...")
run_stats = usr_merged.groupby("run_dir").agg(
    max_tp=("gtpu_kbitss_dn__kbits_rx_s", lambda x: x.max() / 1e6),
    mean_pw=("power_microwatts", lambda x: x.mean() / 1e6),
    total_lost=("gtpu_packets_dn__packets_lost_delta", "sum"),
    total_rx=("gtpu_packets_dn__packets_rx_delta", "sum"),
    mean_delay=("downlink_one_way_delay_distribution__weighted_mean_delay_us", "mean"),
).reset_index()
run_stats["loss_pct"] = 100 * run_stats["total_lost"] / (
    run_stats["total_rx"] + run_stats["total_lost"]).replace(0, np.nan)

fig, axes = plt.subplots(1, 2, figsize=(7, 3))

ax = axes[0]
sc = ax.scatter(run_stats["max_tp"], run_stats["loss_pct"],
                c=run_stats["mean_pw"], cmap="YlOrRd", s=25,
                edgecolors="k", linewidths=0.3)
plt.colorbar(sc, ax=ax, label="Mean power (W)")
ax.set_xlabel("Max offered throughput (Gbps)")
ax.set_ylabel("Packet loss (%)")
ax.set_title("(a) Loss vs offered load", fontweight="bold")

ax = axes[1]
sc2 = ax.scatter(run_stats["max_tp"], run_stats["mean_pw"],
                 c=run_stats["loss_pct"], cmap="RdYlGn_r", s=25,
                 edgecolors="k", linewidths=0.3)
plt.colorbar(sc2, ax=ax, label="Loss (%)")
ax.set_xlabel("Max offered throughput (Gbps)")
ax.set_ylabel("Mean power (W)")
ax.set_title("(b) Power vs offered load", fontweight="bold")

fig.suptitle("USR-UPF: Per-Run Saturation", fontsize=11, fontweight="bold")
fig.tight_layout()
save(fig, "fig7_oai_per_run_saturation")


print(f"\nAll figures saved to {OUT_DIR}/")
print("Files:")
for f in sorted(OUT_DIR.glob("*")):
    print(f"  {f.name}  ({f.stat().st_size / 1024:.0f} KB)")
