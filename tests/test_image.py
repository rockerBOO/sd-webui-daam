from webui_daam.image import plot_overlay_heat_map
from PIL import Image
import torch


def test_plot_overlay_heat_map():
    img = Image.fromarray()
    heatmap = torch.rand(364, 640, 1)

    overlay_img = plot_overlay_heat_map(img, heatmap, "testing")

    assert isinstance(overlay_img, Image)
