# DAAM Extension for Stable Diffusion Web UI

This is a port of [DAAM](https://github.com/castorini/daam) for Stable Diffusion Web UI.

![tmpp1ascudo](https://github.com/rockerBOO/sd-webui-daam/assets/15027/58d3fc2f-60cc-4a87-ba88-001b719399f0)

<!--toc:start-->

- [DAAM Extension for Stable Diffusion Web UI](#daam-extension-for-stable-diffusion-web-ui)
  - [Setup](#setup)
  - [How to use](#how-to-use)
  - [Sample](#sample)
  - [Development](#development)
  - [Contributions](#contributions)
  <!--toc:end-->

## Setup

![Screenshot 2023-12-01 at 21-59-10 Stable Diffusion](https://github.com/rockerBOO/sd-webui-daam/assets/15027/877a8159-89de-430f-ab7e-61bbb215a0c1)

## How to use

![Screenshot 2023-12-01 at 22-03-24 Stable Diffusion](https://github.com/rockerBOO/sd-webui-daam/assets/15027/489c7431-f020-4af0-939f-930543e21cd5)

An overlapping image with a heat map for each attention will be generated along with the original image.
Images will now be created in the default output directory.

Attention text is divided by commas, but multiple words without commas are recognized as a single sequence.
If you type "cat" for attention text, then all the tokens matching "cat" will be retrieved and combined into attention.
If you type "cute cat", only tokens with "cute" and "cat" in sequence will be retrieved and only their attention will be output.

## Sample

prompt : "a photograph of a cool, fantastic, awesome woman wearing sunglasses leaning over an old fashioned jukebox"

attention text: "woman, sunglasses, old fashioned jukebox"

![tmpwpx3s4ke](https://github.com/rockerBOO/sd-webui-daam/assets/15027/d306db4b-efe3-4f82-afbd-f86b95a4ad90)

![tmp1r00ktge](https://github.com/rockerBOO/sd-webui-daam/assets/15027/55beb061-dbdf-4e58-a07c-4f93c2ee7c50)

## Known issues

- Automatic1111 1.7 is not working properly. Something changed and trying to identify the issue. 
- Batch size works but isn't lining up with the right areas.
- Grids may have different images mixed together, especially bad in combo with batches.
- Certain resolutions may result in a `view of [16, 29, 39]` not working. Try changing the resolution. Please feel free to make a ticket with the error and resolution you were trying.

## Changelog

2023-12-02 — SDXL now supported

## Development

Using [ruff](https://docs.astral.sh/ruff/) for linting and formatting.
Tests using [pytest](https://pytest.org).

## Contributions

Issues and PRs welcome.

## Credits

@IAMAl — https://github.com/IAMAl — For support in making the attention heatmaps work at different aspect ratios and overall supporting me.
@kousw — https://github.com/kousw/stable-diffusion-webui-daam — For creating the initial extension
@toriato — https://github.com/toriato/stable-diffusion-webui-daam — For adapting for aspect ratio and batch sizes
@castorini — https://github.com/castorini/daam — For creating the DAAM attention heatmapping library
