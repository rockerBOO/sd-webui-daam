from __future__ import annotations

import gradio as gr
import matplotlib
import modules.scripts as scripts
import modules.shared as shared
import open_clip.tokenizer
import torch
from daam import trace
from ldm.modules.encoders.modules import FrozenOpenCLIPEmbedder
from modules import (
    script_callbacks,
    sd_hijack_clip,
    sd_hijack_open_clip,
)
from modules.images import image_grid, save_image
from modules.processing import (
    StableDiffusionProcessing,
    fix_seed,
)
from modules.shared import opts
from transformers.image_transforms import to_pil_image

from webui_daam.log import debug, info, warning, error, log
from webui_daam.image import (
    create_heatmap_image_overlay,
    compile_processed_image,
)
from webui_daam.tokenizer import Tokenizer
from webui_daam.prompt import PromptAnalyzer, calc_context_size, escape_prompt
from webui_daam.grid import GridOpts, GRID_LAYOUT_AUTO
from webui_daam.heatmap import calc_global_heatmap

matplotlib.use("Agg")

addnet_paste_params = {"txt2img": [], "img2img": []}


class Script(scripts.Script):
    GRID_LAYOUT_AUTO = "Auto"
    GRID_LAYOUT_PREVENT_EMPTY = "Prevent Empty Spot"
    GRID_LAYOUT_BATCH_LENGTH_AS_ROW = "Batch Length As Row"

    def __init__(self):
        self.trace = None

    def title(self):
        return "DAAM script"

    def describe(self):
        return """
        Description of the DAAM script
        """

    def show(self, is_img2img):
        return scripts.AlwaysVisible

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
                        label="Use grid",
                        value=False,
                        info="Output to grid dir",
                    )

                    grid_per_image = gr.Checkbox(
                        label="Grid per image",
                        value=True,
                        info="Attention heatmap grid per image",
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

                with gr.Row(visible=False):
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
            grid_per_image,
            grid_layout,
            alpha,
            heatmap_image_scale,
            trace_each_layers,
            layers_as_row,
            # False,  # disabling trace for now
            # False,
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
        grid_per_image: bool,
        grid_layout: str,
        alpha: float,
        heatmap_image_scale: float,
        trace_each_layers: bool,
        layers_as_row: bool,
    ):
        self.enabled = False  # in case the assert fails

        # Panicing level of unhooking...
        self.try_unhook()

        self.debug("DAAM Process...")

        self.debug(f"attention text {attention_texts}")
        assert opts.samples_save, (
            "Cannot run Daam script. Enable "
            + "Always save all generated images' setting."
        )

        self.images = []
        self.show_images = show_images
        self.save_images = save_images
        self.show_caption = show_caption
        self.alpha = alpha
        self.use_grid = use_grid
        self.grid_layout = grid_layout
        self.heatmap_image_scale = heatmap_image_scale
        self.heatmap_images = dict()
        self.global_heatmaps = []

        self.attentions = [
            s.strip()
            for s in attention_texts.split(",")
            if s.strip() and len(s.strip()) > 0
        ]
        self.enabled = len(self.attentions) > 0 and enabled
        self.try_unhook()

        fix_seed(p)

    def get_tokenizer(self, p):
        if isinstance(
            p.sd_model.cond_stage_model.wrapped, FrozenOpenCLIPEmbedder
        ):
            return Tokenizer(open_clip.tokenizer._tokenizer.encode)

        return Tokenizer(
            p.sd_model.cond_stage_model.wrapped.tokenizer.tokenize
        )

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
            assert False, (
                f"Embedder '{type(p.sd_model.cond_stage_model)}' "
                + "is not supported."
            )

        tokens = self.tokenize(p, escape_prompt(prompt))
        self.debug(f"DAAM tokens: {tokens}")
        context_size = calc_context_size(len(tokens))

        prompt_analyzer = PromptAnalyzer(embedder, prompt)
        self.prompt_analyzer = prompt_analyzer
        context_size = prompt_analyzer.context_size

        self.debug(
            f"daam run with context_size={prompt_analyzer.context_size}, "
            + f"token_count={prompt_analyzer.token_count}"
        )
        self.debug(
            f"remade_tokens={prompt_analyzer.tokens}, "
            + f"multipliers={prompt_analyzer.multipliers}"
        )
        self.debug(
            f"hijack_comments={prompt_analyzer.hijack_comments}, "
            + f"used_custom_terms={prompt_analyzer.used_custom_terms}"
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
        grid_per_image: bool,
        grid_layout: str,
        alpha: float,
        heatmap_image_scale: float,
        trace_each_layers: bool,
        layers_as_row: bool,
        prompts,
        **kwargs,
    ):
        self.try_unhook()
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
            info("Embedding heatmap cannot be shown.")

        tokenizer = self.get_tokenizer(p)

        self.trace = trace(
            unet=p.sd_model.model.diffusion_model,
            vae=p.sd_model.first_stage_model,
            vae_scale_factor=8,
            tokenizer=tokenizer,
            width=p.width,
            height=p.height,
            context_size=context_size,
            sample_size=64,  # TODO: Update to proper sample size
            image_processor=to_pil_image,
            batch_size=p.batch_size,
        )

        info("Trace attention heatmaps for prompt: ")
        info(f"\t{styled_prompt}")
        info("Attention words: ")
        for attn in self.attentions:
            info(f"\t{attn}")

        self.heatmap_blend_alpha = alpha

        self.trace.hook()

        # self.set_infotext_fields(p, self.latest_params)

    def postprocess_batch(
        self,
        p: StableDiffusionProcessing,
        attention_texts: str,
        enabled: bool,
        show_images: bool,
        save_images: bool,
        show_caption: bool,
        use_grid: bool,
        grid_per_image: bool,
        grid_layout: str,
        alpha: float,
        heatmap_image_scale: float,
        trace_each_layers: bool,
        layers_as_row: bool,
        *args,
        **kwargs,
    ):
        debug("POSTPROCESS BATCH")
        # pprint(kwargs)

    def postprocess_batch_list(
        self,
        p,
        pp: scripts.PostprocessBatchListArgs,
        attention_texts: str,
        enabled: bool,
        show_images: bool,
        save_images: bool,
        show_caption: bool,
        use_grid: bool,
        grid_per_image: bool,
        grid_layout: str,
        alpha: float,
        heatmap_image_scale: float,
        trace_each_layers: bool,
        layers_as_row: bool,
        *args,
        **kwargs,
    ):
        debug("POSTPROCESS BATCH LIST")

        images = pp.images
        batch_number = kwargs["batch_number"]

        batch_size = p.batch_size

        # Batch count number
        # n_iter = p.n_iter

        # Iteration number of the batch count (of n_iter)
        # iteration = p.iteration

        # seed of the image we are currently processing
        # seed = p.seeds[p.batch_index]

        # Num input/output blocks for tracing the layers
        num_input_blocks = len(p.sd_model.model.diffusion_model.input_blocks)
        num_output_blocks = len(p.sd_model.model.diffusion_model.output_blocks)

        global_heat_maps = calc_global_heatmap(
            self.trace,
            p.prompts,  # TODO: we should be getting the right prompt here
            trace_each_layer=self.trace_each_layers,
            num_input_blocks=num_input_blocks,
            num_output_blocks=num_output_blocks,
        )

        for global_heat_map in global_heat_maps:
            debug(
                f"Global heatmap ({len(global_heat_map.heat_maps)}) "
                + f"for {global_heat_map.prompts} "
            )
            heatmap_images = []

            for image_idx, (image, seed) in enumerate(zip(images, p.seeds)):
                for attention in self.attentions:
                    debug(f"batch_idx {image_idx} attention: {attention}")

                    img = create_heatmap_image_overlay(
                        global_heat_map,
                        attention,
                        image=image,
                        show_word=self.show_caption,
                        alpha=self.heatmap_blend_alpha,
                        batch_idx=image_idx,
                        opts=opts,
                    )

                    heatmap_images.append(img)

                    if self.save_images:
                        save_image(
                            img,
                            p.outpath_samples,
                            "daam",
                            grid=False,
                            p=p,
                        )

                if len(heatmap_images) / batch_size != len(self.attentions):
                    info(
                        f"Heatmap images ({len(heatmap_images)}) not matching "
                        + f"# of attentions ({len(self.attentions)})"
                    )

                self.heatmap_images[seed] = heatmap_images

            if len(self.heatmap_images[seed]) == 0:
                info("DAAM: Did not create any heatmap images.")

        self.try_unhook()

    @torch.no_grad()
    def postprocess(
        self,
        p: StableDiffusionProcessing,
        processed,
        attention_texts: str,
        enabled: bool,
        show_images: bool,
        save_images: bool,
        show_caption: bool,
        use_grid: bool,
        grid_per_image: bool,
        grid_layout: str,
        alpha: float,
        heatmap_image_scale: float,
        trace_each_layers: bool,
        layers_as_row: bool,
        **kwargs,
    ):
        self.try_unhook()
        debug("Postprocess...")
        if self.is_enabled(attention_texts, enabled) is False:
            debug("disabled...")
            return

        initial_info = None

        if initial_info is None:
            initial_info = processed.info

        # if layers_as_row:
        #     heatmap_images = []
        #     for k in sorted(self.heatmap_images.keys()):
        #         imgs += [
        #             self.heatmap_images[k][len(self.attentions) * i + j]
        #             for j in range(len(self.attentions))
        #         ]
        #     heatmap_images.extend(imgs)
        # else:

        # heatmap_images = self.heatmap_images.keys()

        debug(
            "Heatmap images: "
            + f"{[len(hm_imgs) for hm_imgs in self.heatmap_images.values()]} "
            + f"attentions {len(self.attentions)}"
        )
        debug(f"Images: {len(processed.images)}")

        debug(f"processed images: {processed.images}")

        for (seed, heatmap_images), img in zip(
            self.heatmap_images.items(), processed.images
        ):
            if processed.seed != seed:
                debug(f"INVALID SEED processed {processed.seed} {seed}")
            debug(f"Processing seed {seed} ")
            debug(f"heatmap_images {heatmap_images}")
            debug(f"img {img}")

            (
                images,
                infotexts,
                offset,
                grid_images_list,
            ) = compile_processed_image(
                img,
                heatmap_images,
                processed.infotexts[p.batch_index],
                offset=processed.index_of_first_image,
                grid_opts=GridOpts(
                    layout=GRID_LAYOUT_AUTO,
                    batch_size=p.batch_size,
                    n_iter=p.n_iter,
                    num_attention_words=len(self.attentions),
                    num_heatmap_images=len(self.heatmap_images.keys()),
                    layers_as_row=layers_as_row,
                ),
                use_grid=use_grid,
                grid_per_image=grid_per_image,
                show_images=show_images,
            )

            # save grid images
            if use_grid:
                for grid_images, batch_size, rows in grid_images_list:
                    grid_image = image_grid(
                        grid_images, batch_size=batch_size, rows=rows
                    )

                    if save_images:
                        save_image(
                            grid_image,
                            p.outpath_grids,
                            "grid_daam",
                            grid=True,
                            p=p,
                        )

                    if show_images:
                        images.append(grid_image)
                        # naively adding the first infotext to the grid image
                        infotexts.append(infotexts[0])
                        offset += 1

            debug(
                f"Images: {len(images)} Infotext: {len(infotexts)} Offset: {offset} Grid images: {len(grid_images_list)}"
            )
            debug(f"Images {images}")
            # debug(f"Infotext {infotexts}")

            # Add new images to the start of the processed image list
            processed.images[:0] = images
            processed.index_of_first_image += offset
            processed.infotexts[:0] = infotexts

        return processed

    def is_enabled(self, attention_texts, enabled):
        if self.enabled is False:
            return False

        if enabled is False:
            return False

        if attention_texts == "":
            return False

        return True

    def set_infotext_fields(self, p, params):
        pass
        # p.extra_generation_params.update(
        #     {
        #         f"AddNet Weight B {i+1}": weight_tenc,
        #     }
        # )

    def try_unhook(self):
        if self.trace is not None:
            try:
                self.trace.unhook()
                self.trace = None
            # Possibly not hooked and we are only attempting to unhook
            except RuntimeError:
                pass

    def debug(self, message):
        debug(message)

    def log(self, message):
        log(message)

    def error(self, err, message):
        error(err, message)

    def warning(self, err, message):
        warning(err, message)

    def __getattr__(self, attr):
        warning("unknown call", attr)
        # import traceback
        #
        # traceback.print_stack()


def on_script_unloaded():
    if shared.sd_model:
        for s in scripts.scripts_txt2img.alwayson_scripts:
            if isinstance(s, Script):
                s.try_unhook()
                break


def on_infotext_pasted(infotext, params):
    pass
    # if "AddNet Enabled" not in params:
    #     params["AddNet Enabled"] = "False"
    #
    # # TODO changing "AddNet Separate Weights" does not seem to work
    # if "AddNet Separate Weights" not in params:
    #     params["AddNet Separate Weights"] = "False"
    #
    # for i in range(MAX_MODEL_COUNT):
    #     # Convert combined weight into new format
    #     if f"AddNet Weight {i+1}" in params:
    #         params[f"AddNet Weight A {i+1}"] = params[f"AddNet Weight {i+1}"]
    #         params[f"AddNet Weight B {i+1}"] = params[f"AddNet Weight {i+1}"]
    #
    #     if f"AddNet Module {i+1}" not in params:
    #         params[f"AddNet Module {i+1}"] = "LoRA"
    #     if f"AddNet Model {i+1}" not in params:
    #         params[f"AddNet Model {i+1}"] = "None"
    #     if f"AddNet Weight A {i+1}" not in params:
    #         params[f"AddNet Weight A {i+1}"] = "0"
    #     if f"AddNet Weight B {i+1}" not in params:
    #         params[f"AddNet Weight B {i+1}"] = "0"
    #
    #     params[f"AddNet Weight {i+1}"] = params[f"AddNet Weight A {i+1}"]
    #
    #     if (
    #         params[f"AddNet Weight A {i+1}"]
    #         != params[f"AddNet Weight B {i+1}"]
    #     ):
    #         params["AddNet Separate Weights"] = "True"
    #
    #     # Convert potential legacy name/hash to new format
    #     params[f"AddNet Model {i+1}"] = str(
    #         model_util.find_closest_lora_model_name(
    #             params[f"AddNet Model {i+1}"]
    #         )
    #     )
    #
    #     addnet_xyz_grid_support.update_axis_params(
    #         i, params[f"AddNet Module {i+1}"], params[f"AddNet Model {i+1}"]
    #     )


script_callbacks.on_infotext_pasted(on_infotext_pasted)
