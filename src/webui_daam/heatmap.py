from .log import debug, warning
from typing import List

from daam.trace import DiffusionHeatMapHooker
from daam.heatmap import GlobalHeatMap


def calc_global_heatmap(
    trace: DiffusionHeatMapHooker,
    prompts: List[str],
    num_input_blocks: int,
    num_output_blocks: int,
    trace_each_layer=False,
) -> List[GlobalHeatMap]:
    try:
        debug("Global heatmap using prompt:")
        debug(f"prompt: {prompts}")
        if trace_each_layer:
            global_heatmaps = [
                trace.compute_global_heat_map(prompts, layer_idx=layer_idx)
                for layer_idx in range(
                    num_input_blocks + 1 + num_output_blocks
                )
            ]
        else:
            global_heatmaps = [trace.compute_global_heat_map(prompts)]
    except RuntimeError as err:
        warning(
            err,
            "DAAM: Failed to get computed global heatmap for " + f" {prompts}",
        )
        return []

    return global_heatmaps
