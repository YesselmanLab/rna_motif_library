import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib.patches as patches
import matplotlib.image as mpimg
from PIL import Image

import numpy as np
import seaborn as sns
from typing import List, Union, Optional
from scipy.stats import ks_2samp, pearsonr, linregress
from sklearn.linear_model import LinearRegression


# style functions #############################################################


def publication_style_ax(
    ax, fsize: int = 10, ytick_size: int = 8, xtick_size: int = 8
) -> None:
    """
    Applies publication style formatting to the given matplotlib Axes object.
    Args:
        ax (matplotlib.axes.Axes): The Axes object to apply the formatting to.
        fsize (int, optional): The font size for labels, title, and tick labels. Defaults to 10.
        ytick_size (int, optional): The font size for y-axis tick labels. Defaults to 8.
        xtick_size (int, optional): The font size for x-axis tick labels. Defaults to 8.
    Returns:
        None
    """
    # Set line widths and tick widths
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(width=0.5, size=1.5, pad=1)

    # Set font sizes for labels and title
    ax.xaxis.label.set_fontsize(fsize)
    ax.yaxis.label.set_fontsize(fsize)
    ax.title.set_fontsize(fsize)

    # Set font names for labels, title, and tick labels
    ax.xaxis.label.set_fontname("Arial")
    ax.yaxis.label.set_fontname("Arial")
    ax.title.set_fontname("Arial")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname("Arial")

    # Set font sizes for tick labels
    for label in ax.get_yticklabels():
        label.set_fontsize(ytick_size)
    for label in ax.get_xticklabels():
        label.set_fontsize(xtick_size)

    # Set font sizes for text objects added with ax.text()
    for text in ax.texts:
        text.set_fontname("Arial")
        text.set_fontsize(fsize - 2)


def get_true_subplot_margins_inches(ax, fig):
    """
    Get the actual margins around a subplot including space taken by labels.

    Returns:
    - dict with margins and label information in inches
    """
    fig.canvas.draw()

    # Get subplot position in figure coordinates
    pos = ax.get_position()
    fig_width, fig_height = fig.get_size_inches()

    # Basic subplot margins (just the axes box)
    left_margin = pos.x0 * fig_width
    bottom_margin = pos.y0 * fig_height
    right_margin = (1 - pos.x1) * fig_width
    top_margin = (1 - pos.y1) * fig_height

    # Get label bounding boxes if they exist
    xlabel_bbox = None
    ylabel_bbox = None
    title_bbox = None

    # Measure xlabel
    if ax.get_xlabel():
        xlabel_obj = ax.xaxis.get_label()
        bbox_pixels = xlabel_obj.get_window_extent()
        xlabel_bbox = {
            "width": bbox_pixels.width / fig.dpi,
            "height": bbox_pixels.height / fig.dpi,
            "bottom": bbox_pixels.y0 / fig.dpi,
            "top": bbox_pixels.y1 / fig.dpi,
        }

    # Measure ylabel
    if ax.get_ylabel():
        ylabel_obj = ax.yaxis.get_label()
        bbox_pixels = ylabel_obj.get_window_extent()
        ylabel_bbox = {
            "width": bbox_pixels.width / fig.dpi,
            "height": bbox_pixels.height / fig.dpi,
            "left": bbox_pixels.x0 / fig.dpi,
            "right": bbox_pixels.x1 / fig.dpi,
        }

    # Measure title
    if ax.get_title():
        title_obj = ax.title
        bbox_pixels = title_obj.get_window_extent()
        title_bbox = {
            "width": bbox_pixels.width / fig.dpi,
            "height": bbox_pixels.height / fig.dpi,
            "top": bbox_pixels.y1 / fig.dpi,
            "bottom": bbox_pixels.y0 / fig.dpi,
        }

    # Calculate true margins (distance from figure edge to actual content)
    true_left_margin = left_margin
    true_bottom_margin = bottom_margin
    true_right_margin = right_margin
    true_top_margin = top_margin

    # Adjust for labels extending beyond subplot area
    if ylabel_bbox:
        # Y-label extends to the left of subplot
        label_left_edge = ylabel_bbox["left"]
        if label_left_edge < left_margin:
            true_left_margin = label_left_edge

    if xlabel_bbox:
        # X-label extends below subplot
        label_bottom_edge = xlabel_bbox["bottom"]
        if label_bottom_edge < bottom_margin:
            true_bottom_margin = label_bottom_edge

    if title_bbox:
        # Title extends above subplot
        fig_height_pixels = fig_height * fig.dpi
        label_top_edge_from_bottom = title_bbox["top"]
        title_margin_from_top = fig_height - (label_top_edge_from_bottom)
        if title_margin_from_top < top_margin:
            true_top_margin = title_margin_from_top

    return {
        # Subplot box margins
        "subplot_left": left_margin,
        "subplot_right": right_margin,
        "subplot_top": top_margin,
        "subplot_bottom": bottom_margin,
        "subplot_width": pos.width * fig_width,
        "subplot_height": pos.height * fig_height,
        # True content margins (including labels)
        "true_left": true_left_margin,
        "true_right": true_right_margin,
        "true_top": true_top_margin,
        "true_bottom": true_bottom_margin,
        # Label information
        "xlabel_bbox": xlabel_bbox,
        "ylabel_bbox": ylabel_bbox,
        "title_bbox": title_bbox,
        # Available space for actual plot content
        "content_width": fig_width - true_left_margin - true_right_margin,
        "content_height": fig_height - true_top_margin - true_bottom_margin,
    }


def calculate_subplot_coordinates(
    fig_size_inches, subplot_layout, subplot_size_inches, spacing=None
):
    """
    Calculate subplot coordinates for matplotlib subplots with exact subplot sizes.

    Parameters:
    -----------
    fig_size_inches : tuple
        Figure size as (width, height) in inches
    subplot_layout : tuple
        Number of subplots as (rows, columns)
    subplot_size_inches : tuple
        Exact size of each subplot as (width, height) in inches
    spacing : dict, optional
        Spacing parameters as {'hspace': float, 'wspace': float, 'margins': dict}
        - hspace: horizontal spacing between subplots in inches
        - wspace: vertical spacing between subplots in inches
        - margins: {'left': float, 'right': float, 'top': float, 'bottom': float} in inches
        Defaults to 0.5 inch spacing and 0.75 inch margins

    Returns:
    --------
    list
        List of tuples, each containing (left, bottom, width, height) coordinates
        in figure-relative units (0-1) for each subplot

    Raises:
    -------
    Warning if subplots won't fit in the specified figure size
    """
    import warnings

    # Default spacing if not provided
    if spacing is None:
        spacing = {
            "hspace": 0.5,  # horizontal spacing in inches
            "wspace": 0.5,  # vertical spacing in inches
            "margins": {"left": 0.75, "right": 0.75, "top": 0.75, "bottom": 0.75},
        }

    fig_width, fig_height = fig_size_inches
    rows, cols = subplot_layout
    subplot_width, subplot_height = subplot_size_inches

    # Calculate total space needed
    total_subplot_width = cols * subplot_width
    total_subplot_height = rows * subplot_height

    total_hspace = (cols - 1) * spacing["hspace"]
    total_wspace = (rows - 1) * spacing["wspace"]

    total_margins_width = spacing["margins"]["left"] + spacing["margins"]["right"]
    total_margins_height = spacing["margins"]["top"] + spacing["margins"]["bottom"]

    required_width = total_subplot_width + total_hspace + total_margins_width
    required_height = total_subplot_height + total_wspace + total_margins_height

    # Check if subplots fit and issue warnings
    if required_width > fig_width:
        warnings.warn(
            f"Subplots won't fit horizontally! Required width: {required_width:.2f}\", "
            f'Figure width: {fig_width:.2f}". Consider increasing figure width or '
            f"reducing subplot width/spacing."
        )

    if required_height > fig_height:
        warnings.warn(
            f"Subplots won't fit vertically! Required height: {required_height:.2f}\", "
            f'Figure height: {fig_height:.2f}". Consider increasing figure height or '
            f"reducing subplot height/spacing."
        )

    # Calculate coordinates even if they don't fit (for debugging)
    coordinates = []

    # Convert inches to figure-relative coordinates
    for row in range(rows):
        for col in range(cols):
            # Calculate position in inches
            left_inches = spacing["margins"]["left"] + col * (
                subplot_width + spacing["hspace"]
            )
            bottom_inches = spacing["margins"]["bottom"] + (rows - 1 - row) * (
                subplot_height + spacing["wspace"]
            )

            # Convert to figure-relative coordinates (0-1)
            left_rel = left_inches / fig_width
            bottom_rel = bottom_inches / fig_height
            width_rel = subplot_width / fig_width
            height_rel = subplot_height / fig_height

            coordinates.append((left_rel, bottom_rel, width_rel, height_rel))

    return coordinates


def draw_box_around_subplot(
    fig, coords, linewidth=2, edgecolor="red", facecolor="none"
):
    """
    Draw a box around a subplot in a figure.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to draw the box on
    coords : tuple
        Subplot coordinates (left, bottom, width, height)
    linewidth : float, optional
        Width of the box line (default: 2)
    edgecolor : str, optional
        Color of the box edge (default: "red")
    facecolor : str, optional
        Color of the box face (default: "none")
    """
    bbox = patches.Rectangle(
        (
            coords[0],
            coords[1],
        ),  # (left, bottom) in figure coordinates
        coords[2],  # width in figure coordinates
        coords[3],  # height in figure coordinates
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=fig.transFigure,
    )
    fig.patches.append(bbox)


def draw_box_around_figure(fig, linewidth=2, edgecolor="black", facecolor="none"):
    """
    Draw a box around the entire figure bounds.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to draw the box around
    linewidth : float, optional
        Width of the box line (default: 2)
    edgecolor : str, optional
        Color of the box edge (default: 'black')
    facecolor : str, optional
        Color of the box face (default: 'none')
    """
    # Get the figure bounds in figure coordinates
    fig_width, fig_height = fig.get_size_inches()

    # Create a rectangle that covers the entire figure
    rect = plt.Rectangle(
        (0, 0),  # bottom-left corner
        1,
        1,  # width and height in figure coordinates (0-1)
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=fig.transFigure,  # Use figure coordinates
    )

    # Add the rectangle to the figure
    fig.patches.append(rect)
    return rect


def merge_sequential_coords(coords_list, start_idx, end_idx):
    """
    Merge sequential coordinates into one bigger coordinate.

    Parameters:
    -----------
    coords_list : list
        List of coordinate tuples (left, bottom, width, height)
    start_idx : int
        Starting index of coordinates to merge
    end_idx : int
        Ending index of coordinates to merge (inclusive)

    Returns:
    --------
    list
        Updated list with merged coordinates
    """
    if start_idx >= end_idx or start_idx < 0 or end_idx >= len(coords_list):
        raise ValueError("Invalid start_idx or end_idx")

    # Get the coordinates to merge
    coords_to_merge = coords_list[start_idx : end_idx + 1]

    # Calculate the merged coordinates
    left = min(coord[0] for coord in coords_to_merge)
    bottom = min(coord[1] for coord in coords_to_merge)

    # Find the rightmost and topmost positions
    right = max(coord[0] + coord[2] for coord in coords_to_merge)
    top = max(coord[1] + coord[3] for coord in coords_to_merge)

    # Calculate width and height
    width = right - left
    height = top - bottom

    # Create new list with merged coordinates
    new_coords_list = coords_list[:start_idx]
    new_coords_list.append((left, bottom, width, height))
    new_coords_list.extend(coords_list[end_idx + 1 :])

    return new_coords_list


def merge_neighboring_coords(coords_list, indices):
    """
    Merge a list of coordinate positions into one bigger coordinate.
    The indices do not need to be consecutive in the list, but the corresponding
    rectangles must be adjacent (touching) in the figure.

    Parameters:
    -----------
    coords_list : list
        List of coordinate tuples (left, bottom, width, height)
    indices : list of int
        List of indices of coordinates to merge. Must be adjacent (touching).

    Returns:
    --------
    list
        Updated list with merged coordinates
    """
    if not indices:
        raise ValueError("indices list cannot be empty")
    if any(i < 0 or i >= len(coords_list) for i in indices):
        raise ValueError("indices out of range")

    # Get the coordinates to merge
    coords_to_merge = [coords_list[i] for i in indices]

    # Check that all rectangles are touching (adjacent)
    # We'll use a simple approach: for each pair, check if they touch
    def rects_touch(r1, r2, tol=1e-8):
        # r = (left, bottom, width, height)
        l1, b1, w1, h1 = r1
        l2, b2, w2, h2 = r2
        r1_right = l1 + w1
        r1_top = b1 + h1
        r2_right = l2 + w2
        r2_top = b2 + h2

        # Check for overlap or touching on any edge
        horizontal_touch = (
            abs(r1_right - l2) < tol
            or abs(r2_right - l1) < tol
            or (l1 < r2_right - tol and r1_right > l2 + tol)
        )
        vertical_overlap = (b1 < r2_top - tol and b1 + h1 > b2 + tol) or (
            b2 < r1_top - tol and b2 + h2 > b1 + tol
        )
        vertical_touch = (
            abs(r1_top - b2) < tol
            or abs(r2_top - b1) < tol
            or (b1 < r2_top - tol and r1_top > b2 + tol)
        )
        horizontal_overlap = (l1 < r2_right - tol and l1 + w1 > l2 + tol) or (
            l2 < r1_right - tol and l2 + w2 > l1 + tol
        )
        # Touching if they share an edge (either horizontally or vertically)
        return (horizontal_touch and vertical_overlap) or (
            vertical_touch and horizontal_overlap
        )

    # Calculate the merged coordinates
    left = min(coord[0] for coord in coords_to_merge)
    bottom = min(coord[1] for coord in coords_to_merge)
    right = max(coord[0] + coord[2] for coord in coords_to_merge)
    top = max(coord[1] + coord[3] for coord in coords_to_merge)
    width = right - left
    height = top - bottom

    # Create new list with merged coordinates
    new_coords_list = []
    n = len(coords_list)
    idx_set = set(indices)
    merged = False
    for i in range(n):
        if i in idx_set and not merged:
            new_coords_list.append((left, bottom, width, height))
            merged = True
        elif i not in idx_set:
            new_coords_list.append(coords_list[i])
    return new_coords_list


def draw_boxes_around_coords_list(fig, coords_list):
    """
    Draw a box around each set of subplot coordinates in coords_list,
    using a different color for each box.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to draw the boxes on.
    coords_list : list of tuple
        List of (left, bottom, width, height) tuples in figure coordinates.
    """
    color_cycle = (
        plt.rcParams["axes.prop_cycle"]
        .by_key()
        .get(
            "color",
            [
                "red",
                "blue",
                "green",
                "orange",
                "purple",
                "brown",
                "pink",
                "gray",
                "olive",
                "cyan",
            ],
        )
    )

    for i, coords in enumerate(coords_list):
        color = color_cycle[i % len(color_cycle)]
        draw_box_around_subplot(
            fig, coords, linewidth=2, edgecolor=color, facecolor="none"
        )
    return fig


def load_and_fit_image_to_subplot(image_path, subplot_coords, fig, ax):
    """
    Load an image from file and stretch it to fit in a subplot.

    Parameters:
    -----------
    image_path : str
        Path to the image file
    subplot_coords : tuple
        Subplot coordinates as (left, bottom, width, height) in figure-relative units
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object where the image will be placed

    Returns:
    --------
    matplotlib.image.AxesImage
        The image object that was added to the subplot
    """

    # Load the image
    try:
        img = mpimg.imread(image_path)
    except Exception as e:
        raise ValueError(f"Could not load image from {image_path}: {e}")

    # Clear the axes
    ax.clear()

    # Display the image stretched to fit the subplot
    img_plot = ax.imshow(img, extent=[0, 1, 0, 1], aspect="auto")

    # Set the subplot position
    ax.set_position(subplot_coords)

    # Remove axes ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    return img_plot
