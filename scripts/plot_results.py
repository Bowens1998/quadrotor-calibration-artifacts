"""Render Figures 2–5 and Table I from the supplied numerical summaries.

Only display formatting is performed: no simulation, fitting, model selection,
metric aggregation, or confidence-interval estimation. Inputs are under
``data/results/`` by default; outputs are PNG/PDF/SVG figures and a LaTeX table.

Example: python scripts/plot_results.py --output figures
"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REGIMES = ("mass_1p4", "lag_3")
SELECTORS = ("R", "D", "C")
COLORS = {"R": "#286B94", "D": "#327A6D", "C": "#A16831"}


def save_figure(fig, destination):
    for extension in ("pdf", "svg", "png"):
        target = destination.with_suffix("." + extension)
        fig.savefig(target, dpi=220, facecolor="white")
        if extension == "svg":
            target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n")
    plt.close(fig)


def evidence_style():
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})


def render_figure2(summary, destination):
    evidence_style()
    s = summary['primary']; assert len(s) == 8
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.85),gridspec_kw={'width_ratios':[1,1]},layout='constrained')
    labels=[['Lag: F − scalar','Lag: F − random adapt.','Lag: F − pretrained adapt.','Lag: pretrained − random','Mass: F − phys. features'],['Lag: F − scalar accuracy','Lag: F − scalar contact','Mass: F − phys. accuracy']]
    for ax,indices,lab,scale in zip(axes,[range(5),range(5,8)],labels,[1,100]):
        for y,i in enumerate(indices):
            r=s[i];v=r['difference']*scale;lo,hi=np.array(r['adjusted_99_375_interval'])*scale
            ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',color='#176AA6' if r['regime']=='lag_3' else '#AD5939',capsize=3,ms=4)
        ax.axvline(0,color='#707070',lw=.8,ls='--');ax.set_yticks(range(len(lab)),lab,fontsize=7.2);ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    axes[0].set_xlabel('Trajectory RMSE difference (m)');axes[1].set_xlabel('Accuracy / contact difference (pp)')
    axes[0].set_title('A  Prediction',loc='left');axes[1].set_title('B  Action outcomes',loc='left')
    save_figure(fig, destination)


def render_figure4(summary, destination):
    evidence_style()
    s = summary
    methods=['observer','scalar_old','scalar','physics_features','raw','supervised','feedback'];labels=['Observer feedback','Old-data scalar MPC','New-data scalar MPC','Physical-feature MPC','Raw-history MPC','Frozen-context MPC','Nominal feedback']
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.75),layout='constrained')
    for ax,reg,title in zip(axes,['mass_1p4','lag_3'],['A  Mass × 1.4','B  Actuator lag × 3']):
        vals=[next(r['tracking_rmse'] for r in s['means'] if r['regime']==reg and r['method']==m and r['tier']=='pooled') for m in methods]
        for i,(m,v) in enumerate(zip(methods,vals)):
            ax.hlines(i,0,v,color='#D5DCE0',lw=2);ax.scatter(v,i,color='#176AA6' if m=='supervised' else '#3D6651' if m=='observer' else '#71818B',s=32,zorder=3);ax.text(v+.018,i,f'{v:.3f}',va='center',fontsize=8)
        ax.set_yticks(range(7),labels,fontsize=8);ax.invert_yaxis();ax.set_xlim(0,1.35 if reg=='mass_1p4' else .72);ax.set_xlabel('Valid-flight tracking RMSE (m)');ax.set_title(title,loc='left',fontsize=9);ax.grid(axis='x',alpha=.15)
    save_figure(fig, destination)

def render_figure3(data, destination):
    plt.rcdefaults()
    methods = ['scalar', 'physics_features', 'raw', 'raw_depth', 'adapt_random', 'adapt_pretrained', 'frozen']
    labels = ['Scalar physics', 'Physical features', 'Raw history', 'Raw + depth', 'Adapt-random', 'Adapt-pretrained', 'Frozen context']
    regimes = [('mass_1p4', r'Mass $\times$ 1.4', 98), ('lag_3', r'Actuator lag $\times$ 3', 106)]
    metrics = [('shared_rmse', 'Prediction RMSE (m)', .45, '.3f'), ('regret', 'Selected-action regret', .049, '.4f')]
    colors = ['#64767f'] * 6 + ['#176daa']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 9,
                         'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.linewidth': .6, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 3.65))
    fig.subplots_adjust(left=.17, right=.985, top=.88, bottom=.105, wspace=.23, hspace=.67)
    for row, (regime, name, n) in enumerate(regimes):
        for col, (metric, xlabel, xmax, fmt) in enumerate(metrics):
            ax = axes[row, col]
            rows = [next(v for v in data['method_means'] if v['regime'] == regime and v['method'] == method and v['tier'] == 'pooled') for method in methods]
            values = [v[metric] for v in rows]
            y = np.arange(len(methods))
            ax.hlines(y, 0, values, color=colors, linewidth=1.1, alpha=.6, zorder=2)
            for j, v in enumerate(values):
                ax.scatter(v, j, s=22 if j in [0, 6] else 16, marker='D' if j == 6 else 'o', color=colors[j], zorder=3)
                ax.annotate(format(v, fmt), (v, j), xytext=(5, 0), textcoords='offset points', va='center', fontsize=7.2, color=colors[j])
            ax.set_yticks(y, labels if col == 0 else [''] * len(methods))
            ax.set_ylim(6.65, -.7)
            ax.set_xlim(0, xmax)
            ax.set_xlabel(xlabel + '  (lower is better)', labelpad=3)
            if col == 0:
                ax.set_title(('A  ' if row == 0 else 'B  ') + name + f'   |   {n} shared states', loc='left', pad=8, fontweight='bold')
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.tick_params(axis='y', length=0, pad=6)
            ax.grid(axis='x', color='#e5e9ec', linewidth=.5, zorder=0)
            if col == 0:
                ax.set_xticks([0, .1, .2, .3, .4])
            else:
                ax.set_xticks([0, .01, .02, .03, .04])
    fig.suptitle('Same states and candidate commands | Five calibration campaigns | All seven predictors', y=.992, fontsize=8.4)
    target = destination
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(target.with_suffix('.' + ext), dpi=220, metadata={'Creator': 'Accepted data renderer'} if ext == 'pdf' else None)
    plt.close(fig)


def render_figure5(summary, destination):
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
                         "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.hashsalt": "icra-selection-20260912",
                         "mathtext.fontset": "dejavusans", "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    fig.subplots_adjust(left=.08, right=.985, bottom=.27, top=.82, wspace=.58)
    values = [r["capped_loss_4"] for r in summary["panel_results"]]
    ymax = max(.05, max(values) * 1.17)
    markers = ("o", "s", "D")
    for ri, regime in enumerate(REGIMES):
        ax = axes[ri]
        for si, selector in enumerate(SELECTORS):
            rows = sorted([r for r in summary["panel_results"] if (r["regime"], r["selector"]) == (regime, selector)], key=lambda r: r["panel"])
            for offset, row in zip((-.10, 0, .10), rows):
                ax.scatter(si+offset, row["capped_loss_4"], s=21, marker=markers[row["panel"]],
                           facecolors=COLORS[selector] if row["supported"] else "none", edgecolors=COLORS[selector], linewidths=.9, zorder=3)
            mean = float(np.mean([r["capped_loss_4"] for r in rows]))
            ax.plot([si-.21, si+.21], [mean, mean], color="#223C4A", linewidth=1.6, zorder=4)
        ax.set_xlim(-.45, 2.45); ax.set_ylim(0, ymax)
        ax.set_xticks(range(3), ["R\nPrediction", "D\nAction", "C\nFlight"])
        ax.set_title(("(a) Mass mismatch", "(b) Actuator lag")[ri], loc="left", pad=9)
        ax.set_axisbelow(True); ax.grid(axis="y", color="#E0E5E8", linewidth=.6)
        ax.tick_params(axis="x", length=0)
        if ri == 0:
            ax.set_ylabel(r"Capped tracking loss (m$^2$)")
    ax = axes[2]
    ax.axvline(0, color="#9BAAB2", linestyle=(0, (3, 2)), linewidth=.9)
    edges = [0.0]
    for row, regime in enumerate(REGIMES):
        r = next(r for r in summary["primary"] if r["regime"] == regime)
        lo, hi = r["interval"]; y = 1-row
        assert np.isfinite([lo, hi, r["difference"]]).all() and lo <= hi
        ax.plot([lo, hi], [y, y], color="#223C4A", linewidth=1.6)
        ax.plot([lo, lo], [y-.05, y+.05], color="#223C4A", linewidth=1)
        ax.plot([hi, hi], [y-.05, y+.05], color="#223C4A", linewidth=1)
        ax.scatter(r["difference"], y, color="#223C4A", s=25, zorder=4)
        edges.extend([lo, hi, r["difference"]])
    span = max(max(edges)-min(edges), .002)
    ax.set_xlim(min(edges)-.15*span, max(edges)+.15*span); ax.set_ylim(-.5, 1.5)
    ax.set_yticks([1, 0], ["Mass", "Lag"]); ax.tick_params(axis="y", length=0)
    ax.set_title("(c) Paired test difference", loc="left", pad=9)
    ax.set_xlabel(r"D $-$ R loss (m$^2$)" + "\n97.5% adjusted intervals")
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    handles = [Line2D([], [], linestyle="none", marker=markers[p], color="#697D88", markersize=4, label=f"Panel {p+1}") for p in range(3)]
    handles.append(Line2D([], [], color="#223C4A", linewidth=1.6, label="Fixed-panel mean"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.44, 1.005), ncol=4, frameon=False, fontsize=7.5, columnspacing=1.4, handlelength=1.4)
    for suffix in ("pdf", "png", "svg"):
        metadata = {"CreationDate": None} if suffix == "pdf" else None
        fig.savefig(destination.with_suffix("."+suffix), dpi=220, metadata=metadata)
    plt.close(fig)

def render_table1(summary, output_path):
    """Write the original Table I TeX to a new caller-selected output path."""
    families = ("scalar", "affine", "additive", "joint")
    pooled = [row for row in summary["means"] if row["tier"] == "pooled"]
    rows = {row["family"]: row for row in pooled}
    if len(pooled) != 4 or set(rows) != set(families):
        raise ValueError("Expected exactly one pooled row for each of four readout families")
    if summary.get("common_valid") != 106:
        raise ValueError("This Table I caption requires 106 shared development states")
    metrics = ("forecast_rmse", "held_regret", "nominal_regret", "observer_regret")
    for family in families:
        if not all(np.isfinite(rows[family][metric]) for metric in metrics):
            raise ValueError(f"Nonfinite pooled table value: {family}")
    title = {"scalar": "Scalar physics", "affine": "Affine",
             "additive": "Additive MLP", "joint": "Joint MLP"}
    lines = [r'\begin{table}[t]\centering\footnotesize',
             r'\caption{Lag readout sensitivity on the same 106 development states. RMSE is in metres; regret uses held (H), nominal (N), or observer (O) continuation. Means include all campaigns and seeds.}\label{tab:nonlinear-readout}',
             r'\begin{tabular}{lrrrr}\toprule',
             r'Readout & RMSE & Regret H & Regret N & Regret O\\\midrule']
    for family in families:
        row = rows[family]
        lines.append(title[family] + ' & ' +
                     ' & '.join(f'{row[metric]:.4f}' for metric in metrics) + r'\\')
    lines += [r'\bottomrule\end{tabular}', r'\end{table}', '']
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines))
    return target

RESULT_FILES = {
    "figure2": "figure2.json",
    "figure3": "figure3.json",
    "figure4": "figure4.json",
    "figure5": "figure5.json",
    "table1": "table1.json",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/results",
                        help="Directory containing the five supplied result JSON files")
    parser.add_argument("--output", type=Path, default=ROOT / "figures",
                        help="New directory for the four figures and LaTeX table")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Output already exists; choose a new --output directory")
    data = {}
    for key, filename in RESULT_FILES.items():
        path = args.data_dir / filename
        if not path.is_file():
            parser.error(f"Missing result data: {path}")
        data[key] = json.loads(path.read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    render_figure2(data["figure2"], output / "confirmed_calibration")
    render_figure3(data["figure3"], output / "campaign_decisions_20260911")
    render_figure4(data["figure4"], output / "closed_loop_boundary")
    render_figure5(data["figure5"], output / "selection_validation")
    render_table1(data["table1"], output / "nonlinear_readout_table.tex")
    print(f"Saved four figures (PNG, PDF, SVG) and Table I to {output}")
    print("Displayed the supplied summary values; no new statistical analysis or simulation.")


if __name__ == "__main__":
    main()
