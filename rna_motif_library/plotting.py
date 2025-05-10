import matplotlib.pyplot as plt
import matplotlib.axes as axes
import pandas as pd
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
