"""Small notebook interfaces. No training is reachable from these controls."""
import ipywidgets as w
from IPython.display import display
import matplotlib.pyplot as plt
from . import visuals
from .worlds import generate, relay_pair


def _panel(controls, draw):
    output = w.Output()
    def render():
        with output:
            output.clear_output(wait=True)
            return draw(**{name: control.value for name, control in controls.items()})
    panel = w.VBox([w.HBox(list(controls.values())), output])
    display(panel)
    result = render()
    # Keep the figure outside Output: it remains visible in static previews.
    handle = display(result, display_id=True)
    if hasattr(result, "savefig"):
        plt.close(result)
    def update(_=None):
        result = render()
        if result is not None:
            handle.update(result)
            if hasattr(result, "savefig"):
                plt.close(result)
    for control in controls.values():
        control.observe(update, names="value")
    return panel


def receptive_fields():
    stacks = [("T only", "T"), ("S → T", "S-first"), ("T → S", "T-first"), ("Joint ST", "ST"), ("Attention", "A")]
    return _panel({"architecture": w.Dropdown(options=stacks, value="ST", description="Stack"),
                   "depth": w.IntSlider(value=3, min=1, max=15, description="Depth", continuous_update=False),
                   "sensor": w.IntSlider(value=8, min=0, max=15, description="Forecast at", continuous_update=False)},
                  lambda architecture, depth, sensor: visuals.field_view(architecture, depth, sensor))


def distant_pair():
    return _panel({"distance": w.IntSlider(value=8, min=1, max=15, description="Distance", continuous_update=False)},
                  lambda distance: _relay_pair_figure(distance))


def _relay_pair_figure(distance):
    first, second = relay_pair(distance=distance)
    return visuals.relay_pair_predictions(first, second, int(first.meta["query"][0]))


def attention_lab():
    return _panel({"seed": w.Dropdown(options=[("Run 1", 0), ("Run 2", 1), ("Run 3", 2)], value=0, description="Run"),
                   "layer": w.Dropdown(options=[("Block 1", 0), ("Block 2", 1), ("Block 3", 2)], value=0, description="Block"),
                   "head": w.Dropdown(options=[("Head 1", 0), ("Head 2", 1), ("Head 3", 2), ("Head 4", 3)], value=0, description="Head"),
                   "distance": w.IntSlider(value=8, min=1, max=15, description="Distance", continuous_update=False)},
                  lambda seed, layer, head, distance: visuals.attention_view(generate("relay", 1, 23, distance=distance), layer, head, seed))


