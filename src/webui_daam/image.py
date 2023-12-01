import numpy as np
import math
import torch
from PIL import Image
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
from .log import warning, debug

matplotlib.use("Agg")


def plot_overlay_heat_map(
    im: Image,
    heat_map: torch.Tensor,
    word: Union[None, str] = None,
    out_file=None,
    crop=None,
    color_normalize: bool = True,
    ax=None,
    alpha=1.0,
    opts=None,
):
    # type: (PIL.Image.Image | np.ndarray, torch.Tensor, str, Path, int, bool, plt.Axes) -> None
    dpi = 100
    header_size = 40
    scale = 1.1

    width = math.ceil((im.size[0] / dpi) * scale)
    height = math.ceil(((im.size[1] + header_size) / dpi) * scale)

    if ax is None:
        plt.clf()
        plt_ = create_plot_for_img(im, opts)
    else:
        plt_ = ax

    im = np.array(im)

    heat_map = heat_map.permute(1, 0)  # swap width/height to match numpy array
    # shape height, width

    if crop is not None:
        heat_map = heat_map[crop:-crop, crop:-crop]
        im = im[crop:-crop, crop:-crop]

    if color_normalize:
        plt_.imshow(heat_map.cpu().numpy(), cmap="jet")
    else:
        heat_map = heat_map.clamp_(min=0, max=1)
        plt_.imshow(heat_map.cpu().numpy(), cmap="jet", vmin=0.0, vmax=1.0)

    im = torch.from_numpy(im).float() / 255
    im = torch.cat((im, (1 - (heat_map.unsqueeze(-1) * alpha))), dim=-1)

    plt_.imshow(im)

    if word is not None:
        if ax is None:
            plt_.title(word)
        else:
            ax.set_title(word)

    plt_.gcf().set(
        facecolor=opts.grid_background_color
        if opts is not None
        else "#fff",
        figwidth=width,
        figheight=height,
    )

    img = fig2img(fig=plt_.gcf())
    return img


def create_heatmap_image_overlay(
    heatmap,
    attention_word,
    image,
    show_word=True,
    alpha=1.0,
    batch_idx=0,
    opts=None,
):
    try:
        debug("Heatmap for batch_idx: ", batch_idx)
        word_heatmap = heatmap.compute_word_heat_map(
            word=attention_word, batch_idx=batch_idx
        )
    except ValueError as e:
        warning(e, f"Could not conpute the word heat map for {attention_word}")
        return

    img = plot_overlay_heat_map(
        image,
        word_heatmap.expand_as(image),
        word=attention_word if show_word else None,
        alpha=alpha,
        opts=opts,
    )

    return img


def create_plot_for_img(img, opts):
    plt.clf()
    dpi = 100
    header_size = 40
    scale = 1.1

    width = math.ceil((img.size[0] / dpi) * scale)
    height = math.ceil(((img.size[1] + header_size) / dpi) * scale)

    plt.tight_layout()
    plt.rcParams.update(
        {
            "font.size": 24,
            "figure.figsize": (width, height),
            "figure.dpi": dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0,
            "figure.frameon": False,
            "axes.spines.left": False,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.spines.bottom": False,
            "ytick.major.left": False,
            "ytick.major.right": False,
            "ytick.minor.left": False,
            "xtick.major.top": False,
            "xtick.major.bottom": False,
            "xtick.minor.top": False,
            "xtick.minor.bottom": False,
        }
    )

    if opts is not None:
        plt.rcParams.update(
            {
                "text.color": opts.grid_text_active_color,
                "axes.labelcolor": opts.grid_background_color,
                "figure.facecolor": opts.grid_background_color,
            }
        )

    return plt


# Get the PIL image from a plot figure or the current plot
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

