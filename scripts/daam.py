from __future__ import annotations
import os
import re
from itertools import chain
from typing import Union

import gradio as gr
import numpy as np
import modules.scripts as scripts
from modules import devices
from modules.images import get_font
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
from matplotlib import cm
from daam import trace

before_image_saved_handler = None


class Script(scripts.Script):
    GRID_LAYOUT_AUTO = "Auto"
    GRID_LAYOUT_PREVENT_EMPTY = "Prevent Empty Spot"
    GRID_LAYOUT_BATCH_LENGTH_AS_ROW = "Batch Length As Row"

    DEBUG = False

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
        hide_images: bool,
        dont_save_images: bool,
        hide_caption: bool,
        use_grid: bool,
        grid_layouyt: str,
        alpha: float,
        heatmap_image_scale: float,
        trace_each_layers: bool,
        layers_as_row: bool,
    ):
        global before_image_saved_handler
        before_image_saved_handler = lambda params: self.before_image_saved(params)

    def ui(self, is_img2img):
        with gr.Group():
            with gr.Accordion("Attention Heatmap", open=False):
                attention_texts = gr.Text(
                    label="Attention texts for visualization. (comma separated)",
                    value="",
                )

                with gr.Row():
                    hide_images = gr.Checkbox(label="Hide heatmap images", value=False)

                    dont_save_images = gr.Checkbox(
                        label="Do not save heatmap images", value=False
                    )

                    hide_caption = gr.Checkbox(label="Hide caption", value=False)

                with gr.Row():
                    use_grid = gr.Checkbox(
                        label="Use grid (output to grid dir)", value=False
                    )

                    grid_layouyt = gr.Dropdown(
                        [
                            Script.GRID_LAYOUT_AUTO,
                            Script.GRID_LAYOUT_PREVENT_EMPTY,
                            Script.GRID_LAYOUT_BATCH_LENGTH_AS_ROW,
                        ],
                        label="Grid layout",
                        value=Script.GRID_LAYOUT_AUTO,
                    )

                with gr.Row():
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
                        label="Trace each layers", value=False
                    )

                    layers_as_row = gr.Checkbox(
                        label="Use layers as row instead of Batch Length", value=False
                    )

        return [
            attention_texts,
            hide_images,
            dont_save_images,
            hide_caption,
            use_grid,
            grid_layouyt,
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
        hide_images: bool,
        dont_save_images: bool,
        hide_caption: bool,
        use_grid: bool,
        grid_layouyt: str,
        alpha: float,
        heatmap_image_scale: float,
        trace_each_layers: bool,
        layers_as_row: bool,
    ):
        self.enabled = False  # in case the assert fails

        self.debug("DAAM Process...")

        self.debug(f"attention text {attention_texts}")
        assert (
            opts.samples_save
        ), "Cannot run Daam script. Enable 'Always save all generated images' setting."

        self.images = []
        self.hide_images = hide_images
        self.dont_save_images = dont_save_images
        self.hide_caption = hide_caption
        self.alpha = alpha
        self.use_grid = use_grid
        self.grid_layouyt = grid_layouyt
        self.heatmap_image_scale = heatmap_image_scale
        self.heatmap_images = dict()

        self.attentions = [s.strip() for s in attention_texts.split(",") if s.strip()]
        self.enabled = len(self.attentions) > 0

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

        print(
            f"daam run with context_size={prompt_analyzer.context_size}, token_count={prompt_analyzer.token_count}"
        )
        print(
            f"remade_tokens={prompt_analyzer.tokens}, multipliers={prompt_analyzer.multipliers}"
        )
        print(
            f"hijack_comments={prompt_analyzer.hijack_comments}, used_custom_terms={prompt_analyzer.used_custom_terms}"
        )
        print(f"fixes={prompt_analyzer.fixes}")

        return context_size

    @torch.no_grad()
    def process_batch(
        self,
        p: StableDiffusionProcessing,
        attention_texts: str,
        hide_images: bool,
        dont_save_images: bool,
        hide_caption: bool,
        use_grid: bool,
        grid_layouyt: str,
        alpha: float,
        heatmap_image_scale: float,
        trace_each_layers: bool,
        layers_as_row: bool,
        prompts,
        **kwargs,
    ):
        self.debug("Process batch")
        if not self.enabled:
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

        global before_image_saved_handler
        before_image_saved_handler = lambda params: self.before_image_saved(params)

        self.debug("Set before_image_saved_handler to self.before_image_saved")

        tokenizer = self.get_tokenizer(p)

        # cannot trace the same block from two tracers
        if trace_each_layers:
            num_input = len(p.sd_model.model.diffusion_model.input_blocks)
            num_output = len(p.sd_model.model.diffusion_model.output_blocks)
            self.tracers = [
                trace(
                    unet=p.sd_model.model.diffusion_model,
                    vae=p.sd_model.first_stage_model,
                    vae_scale_factor=8,
                    tokenizer=tokenizer,
                    width=p.width,
                    height=p.height,
                    context_size=context_size,
                    layer_idx={i},
                    sample_size=64,  # Update to proper sample size (using 1.5 here)
                    image_processor=to_pil_image,
                )
                for i in range(num_input + num_output + 1)
            ]
            self.attn_captions = (
                [f"IN{i:02d}" for i in range(num_input)]
                + ["MID"]
                + [f"OUT{i:02d}" for i in range(num_output)]
            )
        else:
            self.tracers = [
                trace(
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
            ]
            self.attn_captions = [""]

        for tracer in self.tracers:
            tracer.hook()

    @torch.no_grad()
    def postprocess(
        self,
        p,
        processed,
        attention_texts: str,
        hide_images: bool,
        dont_save_images: bool,
        hide_caption: bool,
        use_grid: bool,
        grid_layouyt: str,
        alpha: float,
        heatmap_image_scale: float,
        trace_each_layers: bool,
        layers_as_row: bool,
        **kwargs,
    ):
        self.debug("DAAM: postprocess...")
        if self.enabled == False:
            self.debug("DAAM: disabled...")
            return

        for trace in self.tracers:
            try:
                trace.unhook()
            except RuntimeError as e:
                self.error(e, "Could not unhook")

        self.tracers = []

        initial_info = None

        if initial_info is None:
            initial_info = processed.info

        self.images += processed.images

        global before_image_saved_handler
        before_image_saved_handler = None

        if Script.DEBUG:
            print("DAAM: Disabled before_image_saved_handler")

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

        for img_list in images_list:
            if img_list and self.use_grid:
                grid_layout = self.grid_layouyt
                if grid_layout == Script.GRID_LAYOUT_AUTO:
                    if p.batch_size * p.n_iter == 1:
                        grid_layout = Script.GRID_LAYOUT_PREVENT_EMPTY
                    else:
                        grid_layout = Script.GRID_LAYOUT_BATCH_LENGTH_AS_ROW

                if grid_layout == Script.GRID_LAYOUT_PREVENT_EMPTY:
                    grid_img = images.image_grid(img_list)
                elif grid_layout == Script.GRID_LAYOUT_BATCH_LENGTH_AS_ROW:
                    if layers_as_row:
                        batch_size = len(self.attentions)
                        rows = len(self.heatmap_images)
                    else:
                        batch_size = p.batch_size
                        rows = p.batch_size * p.n_iter
                    grid_img = images.image_grid(
                        img_list, batch_size=batch_size, rows=rows
                    )
                else:
                    pass

                if not self.dont_save_images:
                    images.save_image(
                        grid_img, p.outpath_grids, "grid_daam", grid=True, p=p
                    )

                if not self.hide_images:
                    processed.images.insert(0, grid_img)
                    processed.index_of_first_image += 1
                    processed.infotexts.insert(0, processed.infotexts[0])

            else:
                if not self.hide_images:
                    processed.images[:0] = img_list
                    processed.index_of_first_image += len(img_list)
                    processed.infotexts[:0] = [processed.infotexts[0]] * len(img_list)

        return processed

    @torch.no_grad()
    def before_image_saved(self, params: script_callbacks.ImageSaveParams):
        batch_pos = -1
        if params.p.batch_size > 1:
            match = re.search(r"Batch pos: (\d+)", params.pnginfo["parameters"])
            if match:
                batch_pos = int(match.group(1))
        else:
            batch_pos = 0

        if batch_pos < 0:
            print(f"DAAM: Invalid batch size")
            return

        if len(self.tracers) == 0:
            print("DAAM: No tracers to heatmap")

        if len(self.attentions) == 0:
            print("DAAM: No attentions to heamap")

        if len(self.tracers) == 0 or len(self.attentions) < 1:
            return

        for i, tracer in enumerate(self.tracers):
            styled_prompt = shared.prompt_styles.apply_styles_to_prompt(
                params.p.prompt, params.p.styles
            )
            # try:
            try:
                global_heat_map = tracer.compute_global_heat_map(styled_prompt)
            except RuntimeError as err:
                self.error(
                    err,
                    f"DAAM: Failed to get computed global heatmap at"
                    + f" {batch_pos} for {styled_prompt}",
                )
                continue

            if i not in self.heatmap_images:
                self.heatmap_images[i] = []

            if global_heat_map is None:
                continue

            heatmap_images = []
            for attention in self.attentions:
                img_size = params.image.size
                caption = (
                    attention
                    + (" " + self.attn_captions[i] if self.attn_captions[i] else "")
                    if not self.hide_caption
                    else None
                )

                word_heatmap = global_heat_map.compute_word_heat_map(attention)
                if word_heatmap is None:
                    print(f"No heatmaps for '{attention}'")

                word_heatmap.plot_overlay(params.image, f"test-{caption}.png")

                heatmap_img = (
                    word_heatmap.expand_as(params.image)
                    if word_heatmap is not None
                    else None
                )

                img: Image.Image = image_overlay_heat_map(
                    params.image,
                    heatmap_img,
                    alpha=self.alpha,
                    caption=caption,
                    image_scale=self.heatmap_image_scale,
                )
                # img = overlay_heat_map(params.image, heatmap_img, word=caption)

                fullfn_without_extension, extension = os.path.splitext(params.filename)
                full_filename = (
                    fullfn_without_extension
                    + "_"
                    + attention
                    + ("_" + self.attn_captions[i] if self.attn_captions[i] else "")
                    + extension
                )

                if self.use_grid:
                    heatmap_images.append(img)
                else:
                    heatmap_images.append(img)
                    if not self.dont_save_images:
                        img.save(full_filename)

            self.heatmap_images[i] += heatmap_images

        if len(self.heatmap_images) == 0:
            self.log("DAAM: Did not create any heatmap images.")

        self.heatmap_images = {
            j: self.heatmap_images[j]
            for j in self.heatmap_images.keys()
            if self.heatmap_images[j]
        }

        # if it is last batch pos, clear heatmaps
        # if batch_pos == params.p.batch_size - 1:
        #     for tracer in self.tracers:
        #         tracer.reset()

        global before_image_saved_handler
        before_image_saved_handler = None

        self.debug("Disabled inside before_image_saved_handler")

        return

    def debug(self, message):
        print(f"DAAM Debug: {message}")

    def log(self, message):
        print(f"DAAM: {message}")

    def error(self, err, message):
        print(f"DAAM: {message}")

    def __getattr__(self, attr):
        print("unknown call", attr)
        # if attr not in self.__dict__:
        #     return getattr(self.obj, attr)
        # return super().__getattr__(attr)


def handle_before_image_saved(params: script_callbacks.ImageSaveParams):
    if before_image_saved_handler is not None and callable(before_image_saved_handler):
        print("Caling handler image saved callback")

        before_image_saved_handler(params)

    return


script_callbacks.on_before_image_saved(handle_before_image_saved)


# Emulating hugging face tokenizer
class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, prompt):
        return self.tokenizer(prompt)


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
            heatmap = heatmap.permute(1, 0)  # flip width / height
            heatmap = _convert_heat_map_colors()
            heatmap = heatmap.float().detach().numpy().copy().astype(np.uint8)
            heatmap_img = to_pil_image(heatmap)
            img = Image.blend(img, heatmap_img, alpha)
        else:
            img = img.copy()

        if caption:
            img = _write_on_image(img, caption)

        if image_scale != 1.0:
            x, y = img.size
            size = (int(x * image_scale), int(y * image_scale))
            img = img.resize(size, Image.BICUBIC)

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
