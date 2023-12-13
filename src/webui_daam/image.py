import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from daam.heatmap import GlobalHeatMap
from PIL import Image

from webui_daam.grid import GRID_LAYOUT_AUTO, GridOpts, make_grid

from .log import debug, warning

matplotlib.use("Agg")


@dataclass
class Opts:
    grid_background_color: str = "white"
    grid_text_active_color: str = "black"


def plot_overlay_heat_map(
    im: Union[Image.Image, torch.Tensor],
    heat_map: torch.Tensor,
    word: Union[None, str] = None,
    out_file=None,
    crop: Optional[bool] = None,
    color_normalize: bool = True,
    ax: Optional[plt.Axes] = None,
    alpha: Optional[float] = 1.0,
    opts: Optional[Opts] = None,
):
    # type: (PIL.Image.Image | np.ndarray, torch.Tensor, str, Path, int, bool, )lt.Axes) -> None
    dpi = 100
    header_size = 40
    scale = 1.1

    if isinstance(im, Image.Image):
        w = im.size[0]
        h = im.size[1]
    elif isinstance(im, torch.Tensor):
        w = im.size(2)
        h = im.size(1)
    else:
        raise RuntimeError("Invalid image")

    width = math.ceil((w / dpi) * scale)
    height = math.ceil(((h + header_size) / dpi) * scale)

    if ax is None:
        plt.clf()
        plt_ = create_plot_for_img(im, opts)
    else:
        plt_ = ax

    heat_map = heat_map.permute(1, 0)  # swap width/height to match numpy array
    # shape height, width

    if crop is not None:
        heat_map = heat_map[crop:-crop, crop:-crop]
        im = im[crop:-crop, crop:-crop]

    if color_normalize:
        plt_.imshow(heat_map.cpu().numpy(), cmap="jet")
    else:
        heat_map = heat_map.clamp_(min=-1, max=1)
        plt_.imshow(heat_map.cpu().numpy(), cmap="jet", vmin=0.0, vmax=1.0)

    if isinstance(im, Image.Image):
        im = np.array(im)
        im = torch.from_numpy(im).float() / 255
    elif isinstance(im, torch.Tensor):
        # Tensor comes in channel, width, height to width, height, channel
        im = im.permute(1, 2, 0)

    im = torch.cat((im, (1 - (heat_map.unsqueeze(-1) * alpha))), dim=-1)

    plt_.imshow(im)

    if word is not None:
        if ax is None:
            plt_.title(word)
        else:
            ax.set_title(word)

    if ax is None:
        plt_.gcf().set(
            facecolor=get_opt(opts, "grid_background_color", "#FFF")
            if opts is not None
            else "#fff",
            figwidth=width,
            figheight=height,
        )

        img = fig2img(fig=plt_.gcf())
    else:
        img = fig2img(fig=plt)

    if out_file is not None:
        img.save(out_file)

    return img


def create_heatmap_image_overlay(
    heatmap: GlobalHeatMap,
    attention_word: str,
    image: Union[Image.Image, torch.Tensor],
    show_word=True,
    alpha=1.0,
    batch_idx=0,
    opts=None,
):
    try:
        word_heatmap = heatmap.compute_word_heat_map(
            word=attention_word, batch_idx=batch_idx
        )
    except ValueError as e:
        warning(e, f"Could not compute the word heat map for {attention_word}")
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

    if isinstance(img, Image.Image):
        w = img.size[0]
        h = img.size[1]
    elif isinstance(img, torch.Tensor):
        w = img.size(0)
        h = img.size(1)
    else:
        raise RuntimeError("Invalid image")

    width = math.ceil((w / dpi) * scale)
    height = math.ceil(((h + header_size) / dpi) * scale)

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
                "text.color": get_opt(opts, "grid_text_active_color", "#000"),
                "axes.labelcolor": get_opt(
                    opts, "grid_background_color", "#FFF"
                ),
                "figure.facecolor": get_opt(
                    opts, "grid_background_color", "#FFF"
                ),
            }
        )

    return plt


def get_opt(opts, opt, default):
    if hasattr(opts, opt):
        return opts[opt]

    return default


# Get the PIL image from a plot figure or the current plot
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def compile_processed_image(
    image: Image.Image,
    heatmap_images: List[Image.Image],
    infotext: str,
    offset: int,
    grid_opts: GridOpts,
    use_grid=False,
    grid_per_image=False,
    show_images=False,
) -> Tuple[
    List[Image.Image], List[str], int, List[Tuple[List[Image.Image], int, int]]
]:
    grid_images_list = []
    images = []
    infotexts = []
    offset = 0

    # HEATMAP IMAGES

    # if heatmap_images and use_grid:
    #     grid_images_list.append(make_grid(heatmap_images, grid_opts))

    if show_images:
        images.extend(heatmap_images)
        infotexts.extend([infotext] * len(heatmap_images))
        offset += len(heatmap_images)

    # ORIGINAL IMAGES

    if use_grid:
        img_heatmap_grid_img = make_grid(
            heatmap_images + [image], opts=grid_opts
        )

        grid_images_list.append(img_heatmap_grid_img)

        # if show_images:
        # images.append(img_heatmap_grid_img[0])
        # infotexts.append(infotext)
        # offset += 1
        # if  and grid_per_image:
        #     images, infotexts, offset = add_to_start(
        #         #       getting the list of grid images
        #         images, img_heatmap_grid_img[0], infotexts, infotext, offset
        #     )

    return images, infotexts, offset, grid_images_list


def add_to_start(
    images: List[Image.Image],
    imgs: Union[List[Image.Image], Image.Image],
    infotexts: List[str],
    infotext: str,
    offset: int,
) -> Tuple[List[Image.Image], List[str], int]:
    if isinstance(imgs, list):
        images[:0] = imgs
    else:
        images.insert(0, imgs)

    assert isinstance(infotext, list) is False

    infotexts.insert(0, infotext)

    offset += len(images) if isinstance(imgs, list) else 1
    return images, infotexts, offset
