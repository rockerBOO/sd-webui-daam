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
    # dpi = 100
    # header_size = 40
    # scale = 1.1
    #
    # width = math.ceil((im.size[0] / dpi) * scale)
    # height = math.ceil(((im.size[1] + header_size) / dpi) * scale)

    # type: (PIL.Image.Image | np.ndarray, torch.Tensor, str, Path, int, bool, plt.Axes) -> None
    dpi = 100
    header_size = 40
    scale = 1.1

    width = math.ceil((im.size[0] / dpi) * scale)
    height = math.ceil(((im.size[1] + header_size) / dpi) * scale)

    if ax is None:
        plt.clf()
        print("CLEAR PLOT")
        plt_ = create_plot_for_img(im, opts)
    else:
        plt_ = ax

    im = np.array(im)

    # print("plot", heat_map.size())
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
            word=attention_word if show_word else None, batch_idx=batch_idx
        )
    except ValueError as e:
        warning(e, f"Could not conpute the word heat map for {attention_word}")
        return

    img = plot_overlay_heat_map(
        image,
        word_heatmap.expand_as(image),
        word=attention_word,
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


# @torch.no_grad()
# def image_overlay_heat_map(
#     img: Union[Image.Image, np.nparray],
#     heatmap: torch.Tensor,
#     word=None,
#     out_file=None,
#     crop=None,
#     alpha=0.5,
#     caption=None,
#     image_scale=1.0,
# ):
#     # type: (Image.Image | np.ndarray, torch.Tensor, str, Path, int, float, str, float) -> Image.Image
#     assert img is not None
#
#     with devices.without_autocast():
#         if heatmap is not None:
#             shape: torch.Size = heatmap.shape
#             # heatmap = heatmap.permute(1, 0)  # flip width / height
#             heatmap = _convert_heat_map_colors(heatmap)
#
#             # heatmap = heatmap.float().numpy().copy().astype(np.uint8)
#             # heatmap_img = Image.fromarray(heatmap)
#             # print("heatmap", heatmap.size(), "img",  img.size)
#             # print('permute', heatmap.unsqueeze(-1).size())
#             # heatmap_img = to_pil_image(heatmap.float().numpy(), do_rescale=True)
#             print("heatmap", heatmap.size(), "img", img.size)
#             heatmap_img = to_pil_image(
#                 heatmap.permute(1, 0, 2).clone(), do_rescale=True
#             )
#             heatmap_img.save("yo.png")
#             print("heatmap_img", heatmap_img, "img", img)
#             img = Image.blend(img, heatmap_img, alpha)
#         else:
#             img = img.copy()
#
#         if caption:
#             img = _write_on_image(img, caption)
#
#         if image_scale != 1.0:
#             x, y = img.size
#             size = (int(x * image_scale), int(y * image_scale))
#             # img = img.resize(size, Image.BICUBIC)
#             resize_image(resize_mode=0, im=img, width=size[0], height=size[1])
#
#     return img
#
#
# def _convert_heat_map_colors(heat_map: torch.Tensor):
#     def get_color(value):
#         return np.array(cm.turbo(value / 255)[0:3])
#
#     color_map = np.array([get_color(i) * 255 for i in range(256)])
#     color_map = torch.tensor(
#         color_map, device=heat_map.device, dtype=devices.dtype
#     )
#
#     heat_map = (heat_map * 255).long()
#
#     return color_map[heat_map]
#
#
# def _write_on_image(img, caption, fontsize=32):
#     ix, iy = img.size
#     draw = ImageDraw.Draw(img)
#     margin = 2
#     draw = ImageDraw.Draw(img)
#     # font = ImageFont.truetype(Roboto, fontsize)
#
#     font = get_font(fontsize)
#     text_height = iy - 60
#     tx = draw.textbbox((0, 0), caption, font)
#     draw.text(
#         (int((ix - tx[2]) / 2), text_height + margin),
#         caption,
#         (0, 0, 0),
#         font=font,
#     )
#     draw.text(
#         (int((ix - tx[2]) / 2), text_height - margin),
#         caption,
#         (0, 0, 0),
#         font=font,
#     )
#     draw.text(
#         (int((ix - tx[2]) / 2 + margin), text_height),
#         caption,
#         (0, 0, 0),
#         font=font,
#     )
#     draw.text(
#         (int((ix - tx[2]) / 2 - margin), text_height),
#         caption,
#         (0, 0, 0),
#         font=font,
#     )
#     draw.text(
#         (int((ix - tx[2]) / 2), text_height),
#         caption,
#         (255, 255, 255),
#         font=font,
#     )
#
#     return img
