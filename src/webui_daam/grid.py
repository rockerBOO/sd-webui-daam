from dataclasses import dataclass
from typing import List, Union

from PIL import Image

GRID_LAYOUT_AUTO = "Auto"
GRID_LAYOUT_PREVENT_EMPTY = "Prevent Empty Spot"
GRID_LAYOUT_BATCH_LENGTH_AS_ROW = "Batch Length As Row"


@dataclass
class GridOpts:
    """Grid options."""
    layout: str
    batch_size: int
    n_iter: int
    num_attention_words: int
    num_heatmap_images: int
    layers_as_row: bool


def make_grid(img_list: List[Image.Image], opts: GridOpts):
    if opts.layout == GRID_LAYOUT_AUTO:
        if opts.batch_size * opts.n_iter == 1:
            opts.layout = GRID_LAYOUT_PREVENT_EMPTY
        else:
            opts.layout = GRID_LAYOUT_BATCH_LENGTH_AS_ROW

    if opts.layout == GRID_LAYOUT_PREVENT_EMPTY:
        return img_list, opts.batch_size, None
    elif opts.layout == GRID_LAYOUT_BATCH_LENGTH_AS_ROW:
        if opts.layers_as_row:
            batch_size = opts.num_attention_words
            rows = opts.num_heatmap_images
        else:
            batch_size = opts.batch_size
            rows = opts.batch_size * opts.n_iter
        return img_list, batch_size, rows
    else:
        raise RuntimeError(f"Invalid grid layout: {opts.layout}")
