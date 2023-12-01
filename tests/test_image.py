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
    compile_processed_image,
    add_to_start,
)
from webui_daam.grid import GridOpts, GRID_LAYOUT_AUTO

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


# compile_processed_image
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


@pytest.fixture
def sample_heatmap_images():
    # Sample list of heatmap images for testing
    return [Image.new("RGB", (100, 100), color="red") for _ in range(3)]


@pytest.fixture
def sample_infotexts():
    # Sample list of infotext for testing
    return ["infotext1"]


@pytest.fixture
def sample_grid_opts():
    # Sample GridOpts for testing
    return GridOpts(
        layout=GRID_LAYOUT_AUTO,
        batch_size=4,
        n_iter=3,
        num_attention_words=2,
        num_heatmap_images=2,
        layers_as_row=False,
    )


def test_compile_processed_image_default(
    sample_image, sample_heatmap_images, sample_infotexts, sample_grid_opts
):
    # Test compile_processed_image with default options
    images, infotexts, offset, grid_images = compile_processed_image(
        sample_image,
        sample_heatmap_images,
        sample_infotexts,
        0,
        sample_grid_opts,
    )
    assert images == [sample_image]
    assert infotexts == sample_infotexts
    assert offset == 0
    assert grid_images == []


def test_compile_processed_image_show_images(
    sample_image, sample_heatmap_images, sample_infotexts, sample_grid_opts
):
    # Test compile_processed_image with show_images=True
    images, infotexts, offset, grid_images = compile_processed_image(
        sample_image,
        sample_heatmap_images,
        sample_infotexts,
        0,
        sample_grid_opts,
        show_images=True,
    )
    assert images == sample_heatmap_images + [sample_image]
    assert infotexts == sample_infotexts
    assert offset == 4
    assert grid_images == []


def test_compile_processed_image_use_grid(
    sample_image, sample_heatmap_images, sample_infotexts, sample_grid_opts
):
    # Test compile_processed_image with use_grid=True
    images, infotexts, offset, grid_images = compile_processed_image(
        sample_image,
        sample_heatmap_images,
        sample_infotexts,
        0,
        sample_grid_opts,
        use_grid=True,
    )
    assert images == [sample_image]
    assert infotexts == sample_infotexts
    assert offset == 0
    assert len(grid_images) == 2


def test_compile_processed_image_use_grid_per_image(
    sample_image, sample_heatmap_images, sample_infotexts, sample_grid_opts
):
    # Test compile_processed_image with use_grid=True and grid_per_image=True
    images, infotexts, offset, grid_images = compile_processed_image(
        sample_image,
        sample_heatmap_images,
        sample_infotexts,
        0,
        sample_grid_opts,
        use_grid=True,
        grid_per_image=True,
    )
    assert len(images) == 1
    assert infotexts == sample_infotexts
    assert offset == 0
    assert len(grid_images) == 2


# ADD TO START
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


@pytest.fixture
def sample_images():
    # Sample list of images for testing
    return [Image.new("RGB", (100, 100), color="white") for _ in range(3)]


def test_add_to_start_single_image(sample_images):
    # Test add_to_start with a single image
    images, infotexts, offset = add_to_start(
        [], sample_images[0], [], "new_infotext", 0
    )
    assert images == [sample_images[0]]
    assert infotexts == ["new_infotext"]
    assert offset == 1


def test_add_to_start_list_of_images(sample_images):
    # Test add_to_start with a list of images
    images, infotexts, offset = add_to_start(
        [], sample_images, [], "new_infotext", 0
    )
    assert images == sample_images
    assert infotexts == ["new_infotext"]
    assert offset == 3


def test_add_to_start_multiple_calls(sample_images):
    # Test multiple calls to add_to_start
    images, infotexts, offset = add_to_start(
        [], sample_images[0], [], "new_infotext1", 0
    )
    images, infotexts, offset = add_to_start(
        images, sample_images[1:], infotexts, "new_infotext2", offset
    )
    assert images == sample_images[1:] + [sample_images[0]]
    assert infotexts == ["new_infotext2", "new_infotext1"]
    assert offset == 4


def test_add_to_start_empty_images_list(sample_images):
    # Test add_to_start with an empty images list
    images, infotexts, offset = add_to_start(
        [], sample_images, [], "new_infotext", 0
    )
    assert images == sample_images
    assert infotexts == ["new_infotext"]
    assert offset == 3


def test_add_to_start_empty_imgs_list(sample_images):
    # Test add_to_start with an empty imgs list
    images, infotexts, offset = add_to_start([], [], [], "new_infotext", 0)
    assert images == []
    assert infotexts == ["new_infotext"]
    assert offset == 0


def test_add_to_start_empty_infotexts_list(sample_images):
    # Test add_to_start with an empty infotexts list
    images, infotexts, offset = add_to_start([], sample_images[0], [], "", 0)
    assert images == [sample_images[0]]
    assert infotexts == [""]
    assert offset == 1


def test_add_to_start_offset(sample_images):
    # Test add_to_start with a non-zero offset
    images, infotexts, offset = add_to_start(
        [], sample_images[0], [], "new_infotext", 2
    )
    assert images == [sample_images[0]]
    assert infotexts == ["new_infotext"]
    assert offset == 3
