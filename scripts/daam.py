from __future__ import annotations
import os
import re
from itertools import chain
from typing import Union
from pathlib import Path
import math

import gradio as gr
import numpy as np
import modules.scripts as scripts
from modules import devices
from modules.images import image_grid, save_image, resize_image, get_font
import torch
from ldm.modules.encoders.modules import FrozenOpenCLIPEmbedder
import open_clip.tokenizer
from modules import script_callbacks, sd_hijack_clip, sd_hijack_open_clip
from modules.processing import (
    StableDiffusionProcessing,
    fix_seed,
)
from modules.shared import opts
import modules.shared as shared
from PIL import Image, ImageDraw
from modules.sd_hijack_clip import (
    FrozenCLIPEmbedderWithCustomWordsBase,
)
from modules.sd_hijack_open_clip import FrozenOpenCLIPEmbedderWithCustomWords
from transformers.image_transforms import to_pil_image
import matplotlib

matplotlib.use("Agg")
from matplotlib import cm
import matplotlib.pyplot as plt
from daam import trace

global before_image_saved_handler
before_image_saved_handler = None


class Script(scripts.Script):
    GRID_LAYOUT_AUTO = "Auto"
    GRID_LAYOUT_PREVENT_EMPTY = "Prevent Empty Spot"
    GRID_LAYOUT_BATCH_LENGTH_AS_ROW = "Batch Length As Row"

    DEBUG = True

    def title(self):
        return "DAAM script"

    def describe(self):
        return """
        Description of the DAAM script
        """

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def run(
        self,
        p: StableDiffusionProcessing,
        attention_texts: str,
        enabled: bool,
        show_images: bool,
        save_images: bool,
        show_caption: bool,
        use_grid: bool,
        grid_layout: str,
        alpha: float,
        heatmap_image_scale: float,
        trace_each_layers: bool,
        layers_as_row: bool,
    ):
        print("RUN!")

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("Attention Heatmap", open=False):
                with gr.Row():
                    attention_texts = gr.Textbox(
                        placeholder="Attention texts (required)",
                        value="",
                        info="Comma separated. Must be in prompt.",
                        show_label=False,
                        scale=4,
                    )
                    enabled = gr.Checkbox(
                        label="Enabled",
                        value=True,
                        info="Enable tracing the images",
                    )

                with gr.Row():
                    show_images = gr.Checkbox(
                        label="Show heatmap images",
                        value=True,
                        info="Show images in the output area",
                        show_label=False,
                    )

                    save_images = gr.Checkbox(
                        label="Save heatmap images",
                        value=True,
                        info="Save images to the output directory",
                        show_label=False,
                    )

                    show_caption = gr.Checkbox(
                        label="Show caption",
                        value=True,
                        info="Show captions on top of the images",
                        show_label=False,
                    )

                with gr.Row(elem_classes="row-spacer"):
                    use_grid = gr.Checkbox(
                        label="Use grid", value=False, info="Output to grid dir"
                    )

                    grid_layout = gr.Dropdown(
                        [
                            Script.GRID_LAYOUT_AUTO,
                            Script.GRID_LAYOUT_PREVENT_EMPTY,
                            Script.GRID_LAYOUT_BATCH_LENGTH_AS_ROW,
                        ],
                        label="Grid layout",
                        value=Script.GRID_LAYOUT_AUTO,
                    )

                with gr.Row(elem_classes="row-spacer"):
                    alpha = gr.Slider(
                        label="Heatmap blend alpha",
                        value=0.5,
                        minimum=0,
                        maximum=1,
                        step=0.01,
                    )

                    heatmap_image_scale = gr.Slider(
                        label="Heatmap image scale",
                        value=1.0,
                        minimum=0.1,
                        maximum=1,
                        step=0.025,
                    )

                with gr.Row():
                    trace_each_layers = gr.Checkbox(
                        label="Trace IN MID OUT blocks",
                        value=False,
                    )

                    layers_as_row = gr.Checkbox(
                        label="Use layers as row instead of Batch Length",
                        value=False,
                    )

        return [
            attention_texts,
            enabled,
            show_images,
            save_images,
            show_caption,
            use_grid,
            grid_layout,
            alpha,
            heatmap_image_scale,
            trace_each_layers,
            layers_as_row,
        ]

    @torch.no_grad()
    def process(
        self,
        p: StableDiffusionProcessing,
        attention_texts: str,
        enabled: bool,
        show_images: bool,
        save_images: bool,
        show_caption: bool,
        use_grid: bool,
        grid_layout: str,
        alpha: float,
        heatmap_image_scale: float,
        trace_each_layers: bool,
        layers_as_row: bool,
    ):
        self.enabled = False  # in case the assert fails

        def handle_before_image_saved(params):
            self.trace_each_layers = trace_each_layers
            self.before_image_saved(params)

        global before_image_saved_handler
        before_image_saved_handler = handle_before_image_saved

        self.debug("DAAM Process...")

        self.debug(f"attention text {attention_texts}")
        assert (
            opts.samples_save
        ), "Cannot run Daam script. Enable 'Always save all generated images' setting."

        self.images = []
        self.show_images = show_images
        self.save_images = save_images
        self.show_caption = show_caption
        self.alpha = alpha
        self.use_grid = use_grid
        self.grid_layout = grid_layout
        self.heatmap_image_scale = heatmap_image_scale
        self.heatmap_images = dict()

        self.attentions = [
            s.strip()
            for s in attention_texts.split(",")
            if s.strip() and len(s.strip()) > 0
        ]
        self.enabled = len(self.attentions) > 0 and enabled
        self.trace = None

        fix_seed(p)

    def get_tokenizer(self, p):
        if isinstance(p.sd_model.cond_stage_model.wrapped, FrozenOpenCLIPEmbedder):
            return Tokenizer(open_clip.tokenizer._tokenizer.encode)

        return Tokenizer(p.sd_model.cond_stage_model.wrapped.tokenizer.tokenize)

    def tokenize(self, p, prompt):
        tokenizer = self.get_tokenizer(p)

        return tokenizer.tokenize(prompt)

    def get_context_size(self, p: StableDiffusionProcessing, prompt: str):
        embedder = None
        if isinstance(
            p.sd_model.cond_stage_model,
            sd_hijack_clip.FrozenCLIPEmbedderWithCustomWords,
        ) or isinstance(
            p.sd_model.cond_stage_model,
            sd_hijack_open_clip.FrozenOpenCLIPEmbedderWithCustomWords,
        ):
            embedder = p.sd_model.cond_stage_model
        else:
            assert (
                False
            ), f"Embedder '{type(p.sd_model.cond_stage_model)}' is not supported."

        tokens = self.tokenize(p, escape_prompt(prompt))
        self.debug(f"DAAM tokens: {tokens}")
        context_size = calc_context_size(len(tokens))

        prompt_analyzer = PromptAnalyzer(embedder, prompt)
        self.prompt_analyzer = prompt_analyzer
        context_size = prompt_analyzer.context_size

        self.debug(
            f"daam run with context_size={prompt_analyzer.context_size}, token_count={prompt_analyzer.token_count}"
        )
        self.debug(
            f"remade_tokens={prompt_analyzer.tokens}, multipliers={prompt_analyzer.multipliers}"
        )
        self.debug(
            f"hijack_comments={prompt_analyzer.hijack_comments}, used_custom_terms={prompt_analyzer.used_custom_terms}"
        )
        self.debug(f"fixes={prompt_analyzer.fixes}")

        return context_size

    @torch.no_grad()
    def process_batch(
        self,
        p: StableDiffusionProcessing,
        attention_texts: str,
        enabled: bool,
        show_images: bool,
        save_images: bool,
        show_caption: bool,
        use_grid: bool,
        grid_layout: str,
        alpha: float,
        heatmap_image_scale: float,
        trace_each_layers: bool,
        layers_as_row: bool,
        prompts,
        **kwargs,
    ):
        self.debug("Process batch")
        if not self.is_enabled(attention_texts, enabled):
            self.debug("not enabled")
            return

        self.debug("Processing batch...")

        styled_prompt = prompts[0]

        context_size = self.get_context_size(p, styled_prompt)

        if any(
            item[0] in self.attentions
            for item in self.prompt_analyzer.used_custom_terms
        ):
            print("Embedding heatmap cannot be shown.")

        tokenizer = self.get_tokenizer(p)

        # if trace_each_layers:
        #     num_input = len(p.sd_model.model.diffusion_model.input_blocks)
        #     num_output = len(p.sd_model.model.diffusion_model.output_blocks)
        #     self.attn_captions = (
        #         [f"IN{i:02d}" for i in range(num_input)]
        #         + ["MID"]
        #         + [f"OUT{i:02d}" for i in range(num_output)]
        #     )
        # else:
        #     self.attn_captions = [""]
        #
        #

        self.trace = trace(
            unet=p.sd_model.model.diffusion_model,
            vae=p.sd_model.first_stage_model,
            vae_scale_factor=8,
            tokenizer=tokenizer,
            width=p.width,
            height=p.height,
            context_size=context_size,
            sample_size=64,  # Update to proper sample size (using 1.5 here)
            image_processor=to_pil_image,
        )

        self.heatmap_blend_alpha = alpha

        self.trace.hook()

    @torch.no_grad()
    def postprocess(
        self,
        p,
        processed,
        attention_texts: str,
        enabled: bool,
        show_images: bool,
        save_images: bool,
        show_caption: bool,
        use_grid: bool,
        grid_layout: str,
        alpha: float,
        heatmap_image_scale: float,
        trace_each_layers: bool,
        layers_as_row: bool,
        **kwargs,
    ):
        self.debug("DAAM: postprocess...")
        if self.is_enabled(attention_texts, enabled) == False:
            self.debug("DAAM: disabled...")
            return

        initial_info = None

        if initial_info is None:
            initial_info = processed.info

        self.images += processed.images

        if layers_as_row:
            images_list = []
            for i in range(p.batch_size * p.n_iter):
                imgs = []
                for k in sorted(self.heatmap_images.keys()):
                    imgs += [
                        self.heatmap_images[k][len(self.attentions) * i + j]
                        for j in range(len(self.attentions))
                    ]
                images_list.append(imgs)
        else:
            images_list = [
                self.heatmap_images[k] for k in sorted(self.heatmap_images.keys())
            ]

        self.debug(f"Heatmap images: {len(images_list)}")
        self.debug(f"Images: {len(self.images)}")

        for img_list, img in zip(images_list, self.images):
            if img_list and self.use_grid:
                grid_img = self.save_grid(p, img_list, layers_as_row)

                if self.show_images:
                    processed.images.insert(0, grid_img)
                    processed.index_of_first_image += 1
                    processed.infotexts.insert(0, processed.infotexts[0])

            if self.show_images:
                processed.images[:0] = img_list
                processed.index_of_first_image += len(img_list)
                processed.infotexts[:0] = [processed.infotexts[0]] * len(img_list)

            # if trace_each_layers:
            #     save_image_resized = resize_image(
            #         resize_mode=0,
            #         im=img,
            #         width=img_list[0].size[0],
            #         height=img_list[0].size[1],
            #     )
            #
            #     img_heatmap_grid_img = self.save_grid(
            #         p,
            #         [img_list[0]] + [save_image_resized],
            #     )
            # else:
            #     save_image_resized = resize_image(
            #         resize_mode=0,
            #         im=img,
            #         width=img_list[0].size[0],
            #         height=img_list[0].size[1],
            #     )
            #
            #     img_heatmap_grid_img = self.save_grid(
            #         p,
            #         img_list + [save_image_resized],
            #     )
            #
            # processed.images.insert(0, img_heatmap_grid_img)
            # processed.index_of_first_image += 1
            # processed.infotexts.insert(0, processed.infotexts[0])

        return processed

    def is_enabled(self, attention_texts, enabled):
        if self.enabled is False:
            return False

        if enabled is False:
            return False

        if attention_texts == "":
            return False

        return True

    def save_grid(self, p, img_list, layers_as_row=False):
        grid_layout = self.grid_layout
        if grid_layout == Script.GRID_LAYOUT_AUTO:
            if p.batch_size * p.n_iter == 1:
                grid_layout = Script.GRID_LAYOUT_PREVENT_EMPTY
            else:
                grid_layout = Script.GRID_LAYOUT_BATCH_LENGTH_AS_ROW

        if grid_layout == Script.GRID_LAYOUT_PREVENT_EMPTY:
            grid_img = image_grid(img_list)
        elif grid_layout == Script.GRID_LAYOUT_BATCH_LENGTH_AS_ROW:
            if layers_as_row:
                batch_size = len(self.attentions)
                rows = len(self.heatmap_images)
            else:
                batch_size = p.batch_size
                rows = p.batch_size * p.n_iter
            grid_img = image_grid(img_list, batch_size=batch_size, rows=rows)

        if self.save_images:
            save_image(grid_img, p.outpath_grids, "grid_daam", grid=True, p=p)

        return grid_img

    def before_image_saved(self, params: script_callbacks.ImageSaveParams):
        debug(f"Before image saved...")
        print(
            "batch size",
            params.p.batch_size,
            "iteration",
            params.p.iteration,
        )

        if self.trace is None:
            info("No trace")

        if len(self.attentions) == 0:
            info("No attentions to heatmap")

        if self.trace is None or len(self.attentions) == 0:
            return

        styled_prompt = shared.prompt_styles.apply_styles_to_prompt(
            params.p.prompt, params.p.styles
        )

        try:
            if self.trace_each_layers:
                batched_heatmaps = [
                    self.trace.compute_global_heat_map(
                        styled_prompt, layer_idx=layer_idx
                    )
                    for layer_idx in range(num_input + 1 + num_output)
                ]
            else:
                batched_heatmaps = [self.trace.compute_global_heat_map(styled_prompt)]
        except RuntimeError as err:
            self.warning(
                err,
                f"DAAM: Failed to get computed global heatmap for "
                + f" {styled_prompt}",
            )
            return

        debug(
            f"Batched Heatmaps: {len(batched_heatmaps)} heatmaps: {sum([len(hm.heat_maps) for hm in batched_heatmaps])}"
        )
        # debug(f"Attn captions: {len(self.attn_captions)} {[cap for cap in self.attn_captions]}")

        for i, global_heat_map in enumerate(batched_heatmaps):
            if i not in self.heatmap_images:
                self.heatmap_images[i] = []

            heatmap_images = []
            for attention in self.attentions:
                img_size = params.image.size
                caption = attention

                img = create_heatmap_image_overlay(
                    global_heat_map,
                    attention,
                    params=params,
                    alpha=self.heatmap_blend_alpha,
                )

                filename = Path(params.filename)
                attention_caption_filename = filename.with_name(
                    f"{filename.stem}_TEST_{attention}{filename.suffix}"
                )

                if self.use_grid:
                    heatmap_images.append(img)
                else:
                    heatmap_images.append(img)
                    if self.save_images:
                        filename = Path(params.filename)
                        attention_caption_filename = filename.with_name(
                            f"{filename.stem}_{attention}{filename.suffix}"
                        )

                        img.save(attention_caption_filename)

            self.heatmap_images[i] += heatmap_images

        if len(self.heatmap_images) == 0:
            info("DAAM: Did not create any heatmap images.")

        self.heatmap_images = {
            j: self.heatmap_images[j]
            for j in self.heatmap_images.keys()
            if self.heatmap_images[j]
        }

        # if it is last batch pos, clear heatmaps
        # if batch_pos == params.p.batch_size - 1:
        #     for tracer in self.traces:
        #         tracer.reset()

        try:
            self.trace.unhook()
        except RuntimeError as e:
            if e == "Module is not hooked":
                debug(e)
                pass

        return

    def debug(self, message):
        if Script.DEBUG:
            print(f"DAAM Debug: {message}")

    def log(self, message):
        print(f"DAAM: {message}")

    def error(self, err, message):
        print(err)
        self.log(message)

        import traceback

        traceback.print_stack()

    def warning(self, err, message):
        self.log(f"{err} {message}")

    def __getattr__(self, attr):
        import traceback

        print("unknown call", attr)
        # traceback.print_stack()
        # if attr not in self.__dict__:
        #     return getattr(self.obj, attr)
        # return super().__getattr__(attr)


def create_heatmap_image_overlay(heatmap, attention_word, alpha, params):
    try:
        word_heatmap = heatmap.compute_word_heat_map(attention_word)
    except ValueError as e:
        warning(e, "")
        return

    img = plot_overlay_heat_map(
        params.image,
        word_heatmap.expand_as(params.image),
        word=attention_word,
        alpha=alpha,
    )

    return img


def debug(message):
    if Script.DEBUG:
        print(f"DAAM Debug: {message}")


def info(message):
    print(f"DAAM: {message}")


def error(err, message):
    print(err)
    log(message)

    import traceback

    traceback.print_stack()


def warning(err, message):
    log(f"{err} {message}")


@torch.no_grad()
def on_before_image_saved(params):
    global before_image_saved_handler
    return before_image_saved_handler(params)


script_callbacks.on_before_image_saved(on_before_image_saved)


# Emulating hugging face tokenizer
class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, prompt):
        return self.tokenizer(prompt)


# Get the PIL image from a plot figure or the current plot
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


@torch.no_grad()
def plot_overlay_heat_map(
    im,
    heat_map,
    word=None,
    out_file=None,
    crop=None,
    color_normalize=True,
    ax=None,
    alpha=1.0,
):
    dpi = 100
    header_size = 40
    scale = 1.1

    width = math.ceil((im.size[0] / dpi) * scale)
    height = math.ceil(((im.size[1] + header_size) / dpi) * scale)

    # type: (PIL.Image.Image | np.ndarray, torch.Tensor, str, Path, int, bool, plt.Axes) -> None
    if ax is None:
        plt.clf()
        plt_ = plt

        plt_.tight_layout()
        plt_.rcParams.update(
            {
                "font.size": 24,
                "text.color": opts.grid_text_active_color,
                "axes.labelcolor": opts.grid_background_color,
                "figure.facecolor": opts.grid_background_color,
                "figure.figsize": (
                    math.ceil((im.size[0] / dpi) * scale),
                    math.ceil(((im.size[1] + header_size) / dpi) * scale),
                ),
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
    else:
        plt_ = ax

    with devices.without_autocast():
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

        # print(
        #     f"im {im.size()} heat_map {heat_map.size()} {heat_map.unsqueeze(-1).size()}"
        # )

        plt_.imshow(im)

        if word is not None:
            if ax is None:
                plt_.title(word)
            else:
                ax.set_title(word)

        plt_.gcf().set(
            facecolor=opts.grid_background_color, figwidth=width, figheight=height
        )

        # disable_fig_axis(plt_=plt_)
        img = fig2img(fig=plt_.gcf())

        # if word:
        #     img = _write_on_image(img, word)

        return img


@torch.no_grad()
def image_overlay_heat_map(
    img: Union[Image.Image, np.nparray],
    heatmap: torch.Tensor,
    word=None,
    out_file=None,
    crop=None,
    alpha=0.5,
    caption=None,
    image_scale=1.0,
):
    # type: (Image.Image | np.ndarray, torch.Tensor, str, Path, int, float, str, float) -> Image.Image
    assert img is not None

    with devices.without_autocast():
        if heatmap is not None:
            shape: torch.Size = heatmap.shape
            # heatmap = heatmap.permute(1, 0)  # flip width / height
            heatmap = _convert_heat_map_colors(heatmap)

            # heatmap = heatmap.float().numpy().copy().astype(np.uint8)
            # heatmap_img = Image.fromarray(heatmap)
            # print("heatmap", heatmap.size(), "img",  img.size)
            # print('permute', heatmap.unsqueeze(-1).size())
            # heatmap_img = to_pil_image(heatmap.float().numpy(), do_rescale=True)
            print("heatmap", heatmap.size(), "img", img.size)
            heatmap_img = to_pil_image(
                heatmap.permute(1, 0, 2).clone(), do_rescale=True
            )
            heatmap_img.save("yo.png")
            print("heatmap_img", heatmap_img, "img", img)
            img = Image.blend(img, heatmap_img, alpha)
        else:
            img = img.copy()

        if caption:
            img = _write_on_image(img, caption)

        if image_scale != 1.0:
            x, y = img.size
            size = (int(x * image_scale), int(y * image_scale))
            # img = img.resize(size, Image.BICUBIC)
            resize_image(resize_mode=0, im=img, width=size[0], height=size[1])

    return img


def _convert_heat_map_colors(heat_map: torch.Tensor):
    def get_color(value):
        return np.array(cm.turbo(value / 255)[0:3])

    color_map = np.array([get_color(i) * 255 for i in range(256)])
    color_map = torch.tensor(color_map, device=heat_map.device, dtype=devices.dtype)

    heat_map = (heat_map * 255).long()

    return color_map[heat_map]


def _write_on_image(img, caption, fontsize=32):
    ix, iy = img.size
    draw = ImageDraw.Draw(img)
    margin = 2
    draw = ImageDraw.Draw(img)
    # font = ImageFont.truetype(Roboto, fontsize)

    font = get_font(fontsize)
    text_height = iy - 60
    tx = draw.textbbox((0, 0), caption, font)
    draw.text(
        (int((ix - tx[2]) / 2), text_height + margin), caption, (0, 0, 0), font=font
    )
    draw.text(
        (int((ix - tx[2]) / 2), text_height - margin), caption, (0, 0, 0), font=font
    )
    draw.text(
        (int((ix - tx[2]) / 2 + margin), text_height), caption, (0, 0, 0), font=font
    )
    draw.text(
        (int((ix - tx[2]) / 2 - margin), text_height), caption, (0, 0, 0), font=font
    )
    draw.text((int((ix - tx[2]) / 2), text_height), caption, (255, 255, 255), font=font)

    return img


def calc_context_size(token_length: int):
    len_check = 0 if (token_length - 1) < 0 else token_length - 1
    return ((int)(len_check // 75) + 1) * 77


def escape_prompt(prompt):
    if type(prompt) is str:
        prompt = prompt.lower()
        prompt = re.sub(r"[\(\)\[\]]", "", prompt)
        prompt = re.sub(r":\d+\.*\d*", "", prompt)
        return prompt
    elif type(prompt) is list:
        prompt_new = []
        for i in range(len(prompt)):
            prompt_new.append(escape_prompt(prompt[i]))
        return prompt_new


class PromptAnalyzer:
    def __init__(self, clip: FrozenCLIPEmbedderWithCustomWordsBase, text: str):
        use_old = opts.use_old_emphasis_implementation
        assert not use_old, "use_old_emphasis_implementation is not supported"

        self.clip = clip
        self.id_start = clip.id_start
        self.id_end = clip.id_end
        self.is_open_clip = (
            True if type(clip) == FrozenOpenCLIPEmbedderWithCustomWords else False
        )
        self.used_custom_terms = []
        self.hijack_comments = []

        chunks, token_count = self.tokenize_line(text)

        self.token_count = token_count
        self.fixes = list(chain.from_iterable(chunk.fixes for chunk in chunks))
        self.context_size = calc_context_size(token_count)

        tokens = list(chain.from_iterable(chunk.tokens for chunk in chunks))
        multipliers = list(chain.from_iterable(chunk.multipliers for chunk in chunks))

        self.tokens = []
        self.multipliers = []
        for i in range(self.context_size // 77):
            self.tokens.extend(
                [self.id_start] + tokens[i * 75 : i * 75 + 75] + [self.id_end]
            )
            self.multipliers.extend([1.0] + multipliers[i * 75 : i * 75 + 75] + [1.0])

    def create(self, text: str):
        return PromptAnalyzer(self.clip, text)

    def tokenize_line(self, line):
        chunks, token_count = self.clip.tokenize_line(line)
        return chunks, token_count

    def process_text(self, texts):
        (
            batch_multipliers,
            remade_batch_tokens,
            used_custom_terms,
            hijack_comments,
            hijack_fixes,
            token_count,
        ) = self.clip.process_text(texts)
        return (
            batch_multipliers,
            remade_batch_tokens,
            used_custom_terms,
            hijack_comments,
            hijack_fixes,
            token_count,
        )

    def encode(self, text: str):
        return self.clip.tokenize([text])[0]

    def calc_word_indecies(self, word: str, limit: int = -1, start_pos=0):
        word = word.lower()
        merge_idxs = []

        tokens = self.tokens
        needles = self.encode(word)

        limit_count = 0
        current_pos = 0
        for i, token in enumerate(tokens):
            current_pos = i
            if i < start_pos:
                continue

            if needles[0] == token and len(needles) > 1:
                next = i + 1
                success = True
                for needle in needles[1:]:
                    if next >= len(tokens) or needle != tokens[next]:
                        success = False
                        break
                    next += 1

                # append consecutive indexes if all pass
                if success:
                    merge_idxs.extend(list(range(i, next)))
                    if limit > 0:
                        limit_count += 1
                        if limit_count >= limit:
                            break

            elif needles[0] == token:
                merge_idxs.append(i)
                if limit > 0:
                    limit_count += 1
                    if limit_count >= limit:
                        break

        return merge_idxs, current_pos
