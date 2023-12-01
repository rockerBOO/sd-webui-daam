import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch
from daam.heatmap import GlobalHeatMap
from PIL import Image
from webui_daam.image import (
    plot_overlay_heat_map,
    create_heatmap_image_overlay,
)

from webui_daam.tokenizer import Tokenizer

matplotlib.use("Agg")


@pytest.fixture
def sample_global_heatmap():
    # Sample GlobalHeatMap for testing
    tokenizer = lambda x: x.split()  # Example tokenizer function
    prompt = "Test prompt word"
    heat_maps = torch.randn((3, 5, 5))  # Example heat maps tensor
    return GlobalHeatMap(Tokenizer(tokenizer), prompt, [heat_maps] * 2)


@pytest.fixture
def sample_image():
    # Provide a sample image for testing
    return Image.new("RGB", (100, 100), color="white")


@pytest.fixture
def sample_heat_map():
    # Provide a sample heat map tensor for testing
    return torch.rand((100, 100))


def test_basic_functionality(sample_image, sample_heat_map):
    # Test basic functionality
    img = plot_overlay_heat_map(sample_image, sample_heat_map)

    assert isinstance(img, Image.Image)


def test_word_parameter(sample_image, sample_heat_map):
    # Test with a word parameter
    img = plot_overlay_heat_map(sample_image, sample_heat_map, word="TestWord")

    assert isinstance(img, Image.Image)


def test_output_file(sample_image, sample_heat_map):
    tmp_path = Path(tempfile.mkdtemp())

    # Test with an output file
    output_file_path = tmp_path / "output_plot.png"
    plot_overlay_heat_map(
        sample_image, sample_heat_map, out_file=output_file_path
    )
    assert output_file_path.is_file()


def test_crop_parameter_none(sample_image, sample_heat_map):
    # Test with crop=None
    img = plot_overlay_heat_map(sample_image, sample_heat_map, crop=None)

    assert isinstance(img, Image.Image)


def test_color_normalize_true(sample_image, sample_heat_map):
    # Test with color_normalize=True
    img = plot_overlay_heat_map(
        sample_image, sample_heat_map, color_normalize=True
    )

    assert isinstance(img, Image.Image)


def test_color_normalize_false(sample_image, sample_heat_map):
    # Test with color_normalize=False
    img = plot_overlay_heat_map(
        sample_image, sample_heat_map, color_normalize=False
    )

    assert isinstance(img, Image.Image)


def test_axis_parameter(sample_image, sample_heat_map):
    # Test with a specified axis
    fig, ax = plt.subplots()
    img = plot_overlay_heat_map(sample_image, sample_heat_map, ax=ax)
    assert ax.has_data()

    assert isinstance(img, Image.Image)


def test_alpha_parameter(sample_image, sample_heat_map):
    # Test with different alpha values

    img = plot_overlay_heat_map(sample_image, sample_heat_map, alpha=0.5)

    assert isinstance(img, Image.Image)


def test_basic_global_heatmap_functionality(
    sample_global_heatmap, sample_image
):
    # Test basic functionality
    img = create_heatmap_image_overlay(
        sample_global_heatmap, "word", sample_image
    )

    assert isinstance(img, Image.Image)


def test_show_word_true(sample_global_heatmap, sample_image):
    # Test with show_word=True
    img = create_heatmap_image_overlay(
        sample_global_heatmap, "word", sample_image, show_word=True
    )

    assert isinstance(img, Image.Image)


def test_show_word_false(sample_global_heatmap, sample_image):
    # Test with show_word=False
    img = create_heatmap_image_overlay(
        sample_global_heatmap, "word", sample_image, show_word=False
    )

    assert isinstance(img, Image.Image)


def test_global_heatmap_alpha_parameter(sample_global_heatmap, sample_image):
    # Test with different alpha values
    img = create_heatmap_image_overlay(
        sample_global_heatmap, "word", sample_image, alpha=0.5
    )

    assert isinstance(img, Image.Image)


def test_batch_idx_parameter(sample_global_heatmap, sample_image):
    # Test with different batch indices
    img = create_heatmap_image_overlay(
        sample_global_heatmap, "word", sample_image, batch_idx=1
    )

    assert isinstance(img, Image.Image)


def test_opts_parameter(sample_global_heatmap, sample_image):
    # Test with additional options (opts)
    img = create_heatmap_image_overlay(
        sample_global_heatmap,
        "word",
        sample_image,
        opts={
            "grid_background_color": "red",
            "grid_text_active_color": "green",
        },
    )

    assert isinstance(img, Image.Image)
