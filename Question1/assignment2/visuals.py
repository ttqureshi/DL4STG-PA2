"""Figures that expose observations, forecasts, and communication paths."""
from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from matplotlib.colors import ListedColormap
from .models import STACKS, receptive_field
from .evaluation import score_summary

COLORS = {"T": "#4774b5", "S-first": "#ce791c", "T-first": "#8d5aad", "ST": "#228b73", "A": "#c44963"}


def show(figure):
    """Display once and release the figure, including in static notebook previews."""
    from IPython.display import display
    display(figure)
    plt.close(figure)


def heatmap(ax, values, title="Observed history", source=None):
    image = ax.imshow(np.asarray(values), origin="upper", aspect="auto", cmap="coolwarm", vmin=-max(1., float(np.abs(values).max())), vmax=max(1., float(np.abs(values).max())))
    ax.set(xlabel="Sensor", ylabel="Observed time", title=title,
           yticks=[0, len(values)-1], yticklabels=[f"−{len(values)-1}", "0"],
           xticks=np.arange(0, np.asarray(values).shape[1], 2))
    if source is not None:
        ax.axvline(source, color="#242424", linewidth=1.5, linestyle=":")
        ax.text(source, -.8, "source", ha="center", fontsize=9)
    return image


def world_view(batch):
    fig, ax = plt.subplots(figsize=(8, 3))
    source = int(batch.meta["source"][0]) if batch.meta["world"] == "relay" else None
    im = heatmap(ax, batch.x[0, :, :, 0], source=source)
    fig.colorbar(im, ax=ax, label="Reading", fraction=.025, pad=.02)
    fig.tight_layout()
    return fig


def architecture_diagrams(names=None):
    stacks = {name: STACKS[name] for name in names} if names else STACKS
    fig, axes = plt.subplots(len(stacks), 1, figsize=(9, .82*len(stacks)), squeeze=False)
    fill = {"T": "#d8e6f8", "S": "#fbe2bb", "ST": "#cfeee5", "A": "#f9dbe3"}
    for ax, (name, kinds) in zip(axes[:, 0], stacks.items()):
        ax.set(xlim=(-2, 10), ylim=(0, 1))
        ax.axis("off")
        ax.text(-2, .5, name, va="center", fontsize=11)
        for i, kind in enumerate(kinds):
            x = i * 1.4
            ax.add_patch(FancyBboxPatch((x, .18), .95, .64, boxstyle="round,pad=.04", fc=fill[kind], ec="#666666"))
            ax.text(x+.475, .5, kind, ha="center", va="center")
            if i < len(kinds)-1:
                ax.annotate("", xy=(x+1.35,.5), xytext=(x+1.02,.5), arrowprops={"arrowstyle": "->", "color": "#555555"})
    fig.tight_layout(h_pad=.35)
    return fig


def paired_forecast(first, second, architectures, sensor, labels=None):
    """Matched input intervention and within-run forecast differences."""
    from .evaluation import load_model
    import torch
    world = first.meta["world"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    a = first.x[0, :, :, 0].numpy()
    b = second.x[0, :, :, 0].numpy()
    changed = np.argwhere(a != b)
    limit = max(1., abs(a).max(), abs(b).max())
    for ax, batch, title in zip(axes[0], (first, second), ("Before intervention", "After intervention")):
        im = ax.imshow(batch.x[0, :, :, 0], cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
        for t, n in changed:
            ax.add_patch(Rectangle((n-.5, t-.5), 1, 1, fill=False, ec="#111111", lw=1.8))
        ax.add_patch(Rectangle((sensor-.5, -.5), 1, 8, fill=False, ec="#07966e", lw=2.5))
        ax.set(title=title, xlabel="Sensor", ylabel="Observed time", xticks=np.arange(0,16,2),
               yticks=[0,7], yticklabels=["−7", "0"])
    fig.colorbar(im, ax=axes[0].tolist(), label="Reading", shrink=.8)
    times = np.arange(-7, 1)
    axes[1,0].plot(times, a[:,sensor], 'o-', color="#4774b5", label="Before")
    axes[1,0].plot(times, b[:,sensor], 'x--', color="#ce791c", label="After")
    axes[1,0].set(title=f"Forecast sensor {sensor}: history held fixed", xlabel="Observed time", ylabel="Reading", xticks=times)
    axes[1,0].legend(frameon=False)
    truth = float(second.y[0,sensor]-first.y[0,sensor])
    names = ["Required shift"] + [(labels or {}).get(k,k) for k in architectures]
    axes[1,1].axvline(0, color="#aaaaaa", lw=1)
    axes[1,1].axvline(truth, color="#222222", ls=":", lw=1)
    axes[1,1].plot(truth, 0, 'D', color="#222222", markersize=8)
    for i, architecture in enumerate(architectures, 1):
        shifts = []
        for seed in (0,1,2):
            model = load_model(world, architecture, seed=seed)
            with torch.inference_mode():
                shifts.append(float(model(second.x)[0,sensor]-model(first.x)[0,sensor]))
        axes[1,1].errorbar(np.mean(shifts), i, xerr=np.std(shifts), fmt='o', capsize=5,
                          color=COLORS[architecture], markersize=7)
    axes[1,1].set(yticks=range(len(names)), yticklabels=names, xlabel="Next-value shift (after − before)",
                  title="Forecast response: mean ± SD", ylim=(len(names)-.5,-.5))
    axes[1,1].margins(x=.2)
    before, after = float(first.y[0,sensor]), float(second.y[0,sensor])
    fig.suptitle(f"Outlined cells change; green column is held fixed\nRequired next value: {before:.3f} → {after:.3f}", fontsize=12)
    return fig

def field_view(architecture="ST", depth=3, sensor=8):
    kinds = (["S", "T"] if architecture == "S-first" else ["T", "S"]) * depth if architecture in ("S-first", "T-first") else [architecture] * depth
    mask = receptive_field(kinds, 7, sensor)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.imshow(mask, cmap=ListedColormap(["#eeeeee", "#4774b5"]), vmin=0, vmax=1, aspect="auto")
    ax.set(xticks=np.arange(16), yticks=np.arange(9),
           yticklabels=[str(t) for t in range(-7, 1)] + ["+1 forecast"],
           xlabel="Sensor index (one spatial block reaches one sensor farther)", ylabel="Time relative to now",
           ylim=(8.65, -.5))
    ax.set_xticks(np.arange(-.5, 16, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.8)
    ax.tick_params(which="minor", length=0)
    ax.axhline(7.5, color="#333333", linewidth=1.2)
    ax.plot(sensor, 8, "*", color="#c44963", markersize=16)
    ax.annotate("", xy=(sensor, 7.8), xytext=(sensor, 7),
                arrowprops={"arrowstyle": "->", "color": "#c44963"})
    right = sensor < 11
    ax.text(sensor+(.5 if right else -.5), 8, "forecast from latest hidden state",
            va="center", ha="left" if right else "right", fontsize=9)
    reached = np.flatnonzero(mask.any(axis=0))
    unit = "pairs" if architecture in ("S-first", "T-first") else "blocks"
    name = {"T": "T only", "S-first": "S → T", "T-first": "T → S", "ST": "Joint ST", "A": "Attention"}[architecture]
    ax.set_title(f"{name}: {depth} {unit} → forecast at sensor {sensor}\n"
                 f"Reachable sensors: {reached[0]}–{reached[-1]} · colored cells are observed inputs")
    ax.legend(handles=[Patch(facecolor="#4774b5", label="Can influence forecast"),
                       Patch(facecolor="#eeeeee", label="No path to forecast")],
              loc="upper center", bbox_to_anchor=(.5, -.19), ncol=2, frameon=False)
    fig.tight_layout()
    return fig


def identical_histories(first, second, sensor=8):
    """Compare overlapping local histories and their distinct next-step targets."""
    if not 0 < sensor < first.x.shape[2]:
        raise ValueError("The forecast sensor must have a left neighbor.")
    if not np.array_equal(first.x[0, :, sensor], second.x[0, :, sensor]):
        raise ValueError("The two examples must have identical local histories.")
    targets = [float(batch.y[0, sensor]) for batch in (first, second)]
    sources = [float(batch.x[0, -1, sensor-1, 0]) for batch in (first, second)]
    if targets[0] == targets[1] or targets != sources:
        raise ValueError("Targets must differ and equal the latest left-neighbor readings.")

    fig, ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
    times = np.arange(1-first.x.shape[1], 1)
    for i, (batch, target, color, marker) in enumerate(zip(
            (first, second), targets, ("#4774b5", "#ce791c"), ("o", "x")), 1):
        ax.plot(times, batch.x[0, :, sensor-1, 0], color=color,
                ls=":", alpha=.28, lw=1.2)
        ax.plot(times, batch.x[0, :, sensor, 0], color=color, marker=marker,
                ls="-" if i == 1 else "--", lw=1.7, label=f"Example {i}: sensor {sensor}")
        ax.plot([0, 1], [float(batch.x[0, -1, sensor, 0]), target],
                color=color, ls="--", lw=1.3)
        ax.scatter([1], [target], color=color, marker="D", s=60, zorder=5)
        ax.scatter([0], [target], facecolors="white", edgecolors=color, s=45, zorder=5)
        ax.annotate(f"{target:.3f}", (1, target), xytext=(9, 0),
                    textcoords="offset points", va="center", color=color)
    ax.axvline(.5, color="#888888", lw=.8)
    ax.axvspan(.5, 1.7, color="#eeeeee", alpha=.45)
    ax.text(-.8, 1.05, f"Sensor {sensor-1} histories (faint)",
            fontsize=9, color="#777777", ha="right")
    ax.text(1, 1.05, "Next targets", ha="center", fontsize=10)
    ax.set(title=f"Sensor {sensor}: same observed history, different next values",
           xlabel="Time", ylabel="Reading", xticks=np.append(times, 1),
           xticklabels=[str(t) for t in times] + ["+1"],
           ylim=(-1.1, 1.18), xlim=(times[0]-.3, 1.7))
    ax.legend(loc="lower left", frameon=False)
    return fig

def moving_pair_predictions(first, second, sensor=8):
    """Compare forecasts on the same two examples across three training runs."""
    return _paired_history_predictions(
        first, second, sensor,
        (("T", "T only"), ("S-first", "S → T, repeated 3 times")),
        f"Sensor {sensor}: forecasting the two examples from Part 1")


def relay_pair_predictions(first, second, sensor):
    """Show the response to a source change while local history stays fixed."""
    source = int(first.meta["source"][0])
    distance = abs(sensor-source)
    return _paired_history_predictions(
        first, second, sensor, (("ST", "ST: 3 blocks"), ("A", "Attention: 3 blocks")),
        f"Distant-source forecasts: sensor {source} → sensor {sensor} (distance {distance})",
        context=source)


def _paired_history_predictions(first, second, sensor, architectures, title, context=None):
    from .evaluation import load_model
    import torch

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True,
                             layout="constrained")
    times = np.arange(1-first.x.shape[1], 1)
    for example, (ax, batch) in enumerate(zip(axes, (first, second)), 1):
        history = batch.x[0, :, sensor, 0].numpy()
        if context is not None:
            # Faint source history; its latest reading is the only input that differs.
            ax.plot(times, batch.x[0, :, context, 0].numpy(), ":", color="#555555",
                    alpha=.45, lw=1.2, label=f"Source sensor {context} history")
            latest = float(batch.x[0, -1, context, 0])
            ax.plot(0, latest, "o", ms=9, mfc="white", mec="#222222", mew=1.4, zorder=5,
                    label=f"Source's latest reading")
        ax.plot(times, history, "o-", color="#777777", ms=4, lw=1.5,
                label="Observed history" if context is None else f"Forecast sensor {sensor} history")
        target = float(batch.y[0, sensor])
        for (architecture, label), marker in zip(architectures, ("o", "s")):
            with torch.inference_mode():
                predictions = np.array([
                    float(load_model(batch.meta["world"], architecture, seed=seed)(batch.x)[0, sensor])
                    for seed in (0, 1, 2)])
            ax.plot([0, 1], [history[-1], predictions.mean()], "--",
                    color=COLORS[architecture], lw=1.3)
            ax.errorbar(1, predictions.mean(), yerr=predictions.std(),
                        fmt=marker, color=COLORS[architecture], ms=9,
                        mfc="none", mew=1.5, capsize=4, label=label, zorder=4)
        ax.plot(1, target, "D", color="#222222", ms=5, zorder=5,
                label="True next value")
        # With a source shown, its latest reading sits at t=0; label the target on the right.
        ax.annotate(f"{target:.3f}", (1, target), xytext=(-10, 0) if context is None else (10, 0),
                    textcoords="offset points", ha="right" if context is None else "left",
                    va="center", fontsize=10)
        ax.axvline(.5, color="#aaaaaa", lw=.8)
        ax.axvspan(.5, 1.7, color="#eeeeee", alpha=.45, zorder=0)
        ax.text(1.1, 1.05, "Forecast", ha="center", fontsize=9, color="#555555")
        ax.set(title=f"Example {example}", xlabel="Time",
               ylim=(-1.1, 1.18), xlim=(times[0]-.3, 1.7),
               xticks=np.append(times, 1),
               xticklabels=[str(t) for t in times] + ["+1"])
        ax.grid(axis="y", alpha=.15)
    axes[0].set_ylabel("Reading")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4 if context is None else 3,
               frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=12)
    return fig


@lru_cache(maxsize=3)
def source_attention(seed=0, episodes=256):
    """Mean weight from the forecast sensor's latest token to the source's latest token, per block and head."""
    import torch
    from .evaluation import load_model, results
    from .worlds import generate
    batch = generate("relay", episodes, results()["evaluation"]["test_seed"])
    source = torch.as_tensor(batch.meta["source"])
    query = torch.as_tensor(batch.meta["query"])
    keep = source != query
    h, n = batch.x.shape[1:3]
    with torch.inference_mode():
        _, _, weights = load_model("relay", "A", seed=seed)(batch.x, attention=True)
    rows = torch.arange(episodes)
    return np.array([w[rows, :, (h-1)*n+query, (h-1)*n+source][keep].mean(0).numpy() for w in weights])


def attention_view(batch, layer=0, head=0, seed=0):
    import torch
    from .evaluation import load_model
    query = int(batch.meta["query"][0])
    source = int(batch.meta["source"][0])
    with torch.inference_mode():
        _, _, weights = load_model("relay", "A", seed=seed)(batch.x, attention=True)
    h, n = batch.x.shape[1:3]
    row = weights[layer][0, head, (h-1)*n+query].numpy().reshape(h, n)
    grid = source_attention(seed)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), gridspec_kw={"width_ratios": [1, 1, .8]},
                             layout="constrained")
    observed = heatmap(axes[0], batch.x[0, :, :, 0], title="Observed readings (one episode)", source=source)
    axes[0].set_title("Observed readings (one episode)", pad=16)
    fig.colorbar(observed, ax=axes[0], label="Reading", fraction=.05)
    axes[0].plot(query, h-1, "*", color="#242424", markersize=12)
    im = axes[1].imshow(row, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axes[1].plot(query, h-1, "*", color="#eeeeee", markersize=12)
    axes[1].plot(source, h-1, "o", markerfacecolor="none", markeredgecolor="#eeeeee", markersize=10)
    axes[1].set(xlabel="Sensor", ylabel="Observed time", title=f"Block {layer+1}, head {head+1}: sensor {query} attends here",
                yticks=[0, h-1], yticklabels=[f"−{h-1}", "0"])
    axes[2].imshow(grid, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for (b, k), value in np.ndenumerate(grid):
        axes[2].text(k, b, f"{value:.2f}", ha="center", va="center", fontsize=9,
                     color="#111111" if value > .5 else "#eeeeee")
    axes[2].add_patch(Rectangle((head-.45, layer-.45), .9, .9, fill=False, ec="#c44963", lw=2.5))
    axes[2].set(xticks=range(grid.shape[1]), xticklabels=[f"Head {k+1}" for k in range(grid.shape[1])],
                yticks=range(grid.shape[0]), yticklabels=[f"Block {b+1}" for b in range(grid.shape[0])],
                title="Mean weight on source's latest token\n256 held-out episodes (uniform = 1/128)")
    fig.colorbar(im, ax=axes[1:], label="Attention weight", fraction=.04)
    fig.legend(handles=[plt.Line2D([], [], marker="*", ls="", color="#555555", ms=11, label=f"Forecast sensor {query}, latest token"),
                        plt.Line2D([], [], marker="o", ls="", mfc="none", mec="#555555", ms=9, label=f"Source sensor {source}, latest token"),
                        Patch(facecolor="none", edgecolor="#c44963", lw=2, label="Selected head")],
               loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"Attention, training run {seed+1}", fontsize=12)
    return fig


def relay_scores():
    fig, ax = plt.subplots(figsize=(7, 3.1))
    names = ["ST", "A"]
    for i, case in enumerate(("near", "far")):
        summaries = [score_summary("relay", name, case=case) for name in names]
        ax.bar(np.arange(len(names))+(i-.5)*.36, [s[0] for s in summaries], .36,
               yerr=[s[1] for s in summaries], capsize=3, label="Distance 1–3" if case=="near" else "Distance 4–15",
               color="#4774b5" if case=="near" else "#ce791c")
    ax.set(xticks=np.arange(len(names)), xticklabels=["ST: 3 blocks", "Attention: 3 blocks"],
           ylabel="Held-out MAE", ylim=(0, None), title="Local reach versus attention · mean ± SD")
    ax.legend()
    fig.tight_layout()
    return fig


def moving_capacity():
    """Compare alternating and joint processing as hidden width increases."""
    from .evaluation import results
    widths = (4, 8, 32)
    records = {w: results(None if w == 8 else f"moving-width{w}")["results"]["moving"]
               for w in widths}
    fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    for architecture, label in (("S-first", "S → T"), ("T-first", "T → S"), ("ST", "Joint ST")):
        name = architecture+"-p3" if architecture != "ST" else architecture
        scores = np.array([[e["test_mae"] for e in records[w][name]] for w in widths])
        ax.errorbar(range(3), scores.mean(1), yerr=scores.std(1), fmt="o-", capsize=4,
                    label=label, color=COLORS[architecture])
    ax.set(xticks=range(3), xticklabels=widths,
           xlabel="Hidden width (features per sensor–time position)",
           ylabel="Held-out MAE", ylim=(0, .5),
           title="Alternating and joint processing · mean ± SD")
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False)
    show(fig)


def filtering_evidence():
    """Concrete fault episodes and axis-specific, non-learned recovery rules."""
    from .worlds import generate
    fig, axes = plt.subplots(2,2,figsize=(11,6.5), layout="constrained", gridspec_kw={"height_ratios":[2,1.2]})
    for col, case in enumerate(("agreement", "persistence")):
        batch = generate("filtering", 1, 17, mode=case)
        x = batch.x[0,:,:,0].numpy()
        ax = axes[0,col]
        im = ax.imshow(x, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        for t,n in np.argwhere(batch.meta["corrupt"][0]):
            ax.add_patch(Rectangle((n-.5,t-.5),1,1, fill=False, ec="#222222", lw=.8))
        ax.set(title=f"{case.capitalize()}: outlined measurements are faulty", xlabel="Sensor", ylabel="Observed time",
               xticks=range(0,16,2), yticks=[0,7], yticklabels=["−7","0"])
        spatial = np.median(np.lib.stride_tricks.sliding_window_view(x[-1],3),axis=-1)[2:12]
        temporal = np.median(x[1:],axis=0)[3:13]
        truth = batch.y[0,3:13].numpy()
        values = np.stack([truth,x[-1,3:13],spatial,temporal])
        axes[1,col].imshow(values,cmap="coolwarm",vmin=-1,vmax=1,aspect="auto")
        axes[1,col].set(yticks=range(4),yticklabels=["Clean target","Latest reading","Spatial median","Temporal median"],
                        xticks=range(10),xticklabels=range(3,13),xlabel="Forecast sensor")
        for row in (1,2,3):
            for n in np.flatnonzero(abs(values[row]-truth)>1e-6):
                axes[1,col].add_patch(Rectangle((n-.5,row-.5),1,1,fill=False,ec="#222222",lw=1.5))
    fig.colorbar(im,ax=axes.ravel().tolist(),label="Reading",shrink=.8)
    fig.suptitle("Which evidence rejects the fault?\nRecovery rows: outlines mark disagreement with the clean target",fontsize=12)
    return fig


