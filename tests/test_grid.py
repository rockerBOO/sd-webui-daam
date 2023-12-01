import pytest
from webui_daam.grid import (
    make_grid,
    GRID_LAYOUT_AUTO,
    GRID_LAYOUT_PREVENT_EMPTY,
    GRID_LAYOUT_BATCH_LENGTH_AS_ROW,
    GridOpts,
)
from PIL import Image


@pytest.fixture
def sample_image_list():
    # Sample list of images for testing
    return [Image.new("RGB", (100, 100), color="white") for _ in range(5)]


def test_make_grid_prevent_empty_layout(sample_image_list):
    # Test GRID_LAYOUT_PREVENT_EMPTY layout
    opts = GridOpts(
        GRID_LAYOUT_PREVENT_EMPTY,
        batch_size=4,
        n_iter=3,
        num_attention_words=2,
        num_heatmap_images=2,
        layers_as_row=False,
    )
    img_list, batch_size, rows = make_grid(sample_image_list, opts)
    assert img_list == sample_image_list
    assert batch_size == 4
    assert rows is None


def test_make_grid_batch_length_as_row_layout(sample_image_list):
    # Test GRID_LAYOUT_BATCH_LENGTH_AS_ROW layout
    opts = GridOpts(
        GRID_LAYOUT_BATCH_LENGTH_AS_ROW,
        batch_size=4,
        n_iter=3,
        num_attention_words=2,
        num_heatmap_images=2,
        layers_as_row=False,
    )
    img_list, batch_size, rows = make_grid(sample_image_list, opts)
    assert img_list == sample_image_list
    assert batch_size == 4
    assert rows == 12


def test_make_grid_auto_layout_single_element(sample_image_list):
    # Test GRID_LAYOUT_AUTO layout with a single element
    opts = GridOpts(
        GRID_LAYOUT_AUTO,
        batch_size=4,
        n_iter=3,
        num_attention_words=2,
        num_heatmap_images=2,
        layers_as_row=False,
    )
    img_list, batch_size, rows = make_grid([sample_image_list[0]], opts)
    assert img_list == [sample_image_list[0]]
    assert batch_size == 4
    assert rows == 12


def test_make_grid_auto_layout_multiple_elements(sample_image_list):
    # Test GRID_LAYOUT_AUTO layout with multiple elements
    opts = GridOpts(
        GRID_LAYOUT_AUTO,
        batch_size=4,
        n_iter=3,
        num_attention_words=2,
        num_heatmap_images=2,
        layers_as_row=False,
    )
    img_list, batch_size, rows = make_grid(sample_image_list, opts)
    assert img_list == sample_image_list
    assert batch_size == 4
    assert rows == 12


def test_make_grid_auto_layout_single_element_prevent_empty(sample_image_list):
    # Test GRID_LAYOUT_AUTO layout with a single element, should use GRID_LAYOUT_PREVENT_EMPTY
    opts = GridOpts(
        GRID_LAYOUT_AUTO,
        batch_size=4,
        n_iter=1,
        num_attention_words=2,
        num_heatmap_images=2,
        layers_as_row=False,
    )
    img_list, batch_size, rows = make_grid([sample_image_list[0]], opts)
    assert img_list == [sample_image_list[0]]
    assert batch_size == 4
    assert rows == 4


def test_make_grid_auto_layout_multiple_elements_prevent_empty(
    sample_image_list
):
    # Test GRID_LAYOUT_AUTO layout with multiple elements, should use GRID_LAYOUT_PREVENT_EMPTY
    opts = GridOpts(
        GRID_LAYOUT_AUTO,
        batch_size=4,
        n_iter=1,
        num_attention_words=2,
        num_heatmap_images=2,
        layers_as_row=False,
    )
    img_list, batch_size, rows = make_grid(sample_image_list, opts)
    assert img_list == sample_image_list
    assert batch_size == 4
    assert rows == 4


def test_make_grid_invalid_layout():
    # Test invalid grid layout
    opts = GridOpts(
        "invalid_layout",
        batch_size=4,
        n_iter=3,
        num_attention_words=2,
        num_heatmap_images=2,
        layers_as_row=False,
    )
    with pytest.raises(
        RuntimeError, match="Invalid grid layout: invalid_layout"
    ):
        make_grid([], opts)


def test_make_grid_batch_length_as_row_layout_layers_as_row(sample_image_list):
    # Test GRID_LAYOUT_BATCH_LENGTH_AS_ROW layout with layers_as_row set to True
    opts = GridOpts(
        GRID_LAYOUT_BATCH_LENGTH_AS_ROW,
        batch_size=4,
        n_iter=3,
        num_attention_words=5,
        num_heatmap_images=2,
        layers_as_row=True,
    )
    img_list, batch_size, rows = make_grid(sample_image_list, opts)
    assert img_list == sample_image_list
    assert batch_size == 5
    assert rows == 2


def test_make_grid_auto_layout_single_element_large_batch_size(
    sample_image_list
):
    # Test GRID_LAYOUT_AUTO layout with a single element and a large batch size
    opts = GridOpts(
        GRID_LAYOUT_AUTO,
        batch_size=10,
        n_iter=2,
        num_attention_words=2,
        num_heatmap_images=2,
        layers_as_row=False,
    )
    img_list, batch_size, rows = make_grid([sample_image_list[0]], opts)
    assert img_list == [sample_image_list[0]]
    assert batch_size == 10
    assert rows == 20


def test_make_grid_invalid_layout_uppercase():
    # Test invalid grid layout with uppercase characters
    opts = GridOpts(
        "INVALID_LAYOUT",
        batch_size=4,
        n_iter=3,
        num_attention_words=2,
        num_heatmap_images=2,
        layers_as_row=False,
    )
    with pytest.raises(
        RuntimeError, match="Invalid grid layout: INVALID_LAYOUT"
    ):
        make_grid([], opts)


def test_make_grid_invalid_layout_whitespace(sample_image_list):
    # Test invalid grid layout with leading and trailing whitespace
    opts = GridOpts(
        "  invalid_layout  ",
        batch_size=4,
        n_iter=3,
        num_attention_words=2,
        num_heatmap_images=2,
        layers_as_row=False,
    )
    with pytest.raises(
        RuntimeError, match="Invalid grid layout:   invalid_layout  "
    ):
        make_grid(sample_image_list, opts)
