import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

from Tools.load_save_utils import _get_brain_acquisition_limits
from Tools.load_save_utils import load_parameter_names_units_limits
from .colormaps import COLORMAPS, COLORNORMS

# Set matplotlib parameters
plt.style.use('dark_background')
plt.rcParams['figure.constrained_layout.use'] = False
mpl.rcParams['figure.dpi'] = 300
figure_color = 'white'

# Load parameters units and limits from the JSON file in the Plot directory
PARAMETER_DISPLAYED_LABELS, PARAMETER_UNITS, _, _, PARAMETER_LIMS_COLORMAP = load_parameter_names_units_limits(
    os.path.dirname(__file__) + '/../..')

# Set font settings for titles
slice_title_kwargs = dict(fontsize=15, color=figure_color,
                          fontweight='demibold', fontfamily='serif')
param_title_kwargs = dict(fontsize=15, color=figure_color,
                          fontweight='demibold', fontfamily='serif')
figure_title_kwargs = dict(
    fontsize=25, color=figure_color, fontweight='bold', fontfamily='serif')


def plot_parameter_maps(
    parameter_maps: NDArray,
    label_parameters: List[str],
    slices: Union[List[int], str] = "all",
    title: str = "PARAMETER MAPS",
    crop_type: Optional[str] = None,
    n_rotations: int = 0,
    slice_titles=None,
    fig_size=None,
    path_to_data: str = "",
    file_name: Optional[str] = None,
    saving_format: str = "png"
) -> Figure:
    """ 
    Plot multiple parameter maps across one or several slices.

    Parameters
    ----------
    parameter_maps: 4d array of shape (n_parameters, n_x, n_y, n_z)
        Array countaining the parameter maps to plot. 
    label_parameters: List[str]
        List of label parameters. 
    slices: str, optional
        List of slices (in the z direction) to plot. Default to all slices. 
    title: str, optional
        Figure title. Default to "PARAMETER MAPS".  
    crop_type: Optional[str]
        Type of croping of the image. Possibilities are 
        - 'separate': n_x and n_y shapes are independently minimized while keeping the whole brain in the acquisition. 
        - 'equal': n_x and n_y shapes are EQUALLY (n_x = n_y) minimized while keeping the whole brain in the acquisition. 
        - None: no cropping, original n_x and n_y are kept. 
    n_rotations: Int
        Number of 90° rotations to display maps. Possible values are 0 (default), 1, 2 and 3. Default to 0. 
    path_to_data: str, optional
        Figure will be saved in path_to_data/figures. 
    file_name: str, optional
        If specified, the plot is saved with the given file_name. 
    saving_format: str, optional
        Extension for saving the figure. Default to 'png'. 
    """
    if n_rotations:
        parameter_maps = np.rot90(parameter_maps, n_rotations, axes=(1, 2))

    if crop_type:
        mask = ~np.isnan(parameter_maps[0])
        x_min, x_max, y_min, y_max = _get_brain_acquisition_limits(mask, crop_type)
        parameter_maps = parameter_maps[:, x_min:x_max+1, y_min:y_max+1]

    n_params, n_x, n_y, n_z = parameter_maps.shape

    if len(label_parameters) != n_params:
        print(f"Warning: {n_params} maps but {len(label_parameters)} labels.")
        n_params = min(n_params, len(label_parameters))

    if slices == "all":
        slices = list(range(n_z))

    n_slices = len(slices)
    fig_w = 0.5 + 2.5 * n_params if fig_size is None else fig_size[0]
    fig_h = 0.7 + 2.5 * n_x / n_y * n_slices - 1 if fig_size is None else fig_size[1]
    
    fig = plt.figure(figsize=(fig_w, fig_h))
    plt.subplots_adjust(bottom=0.15)

    grid = AxesGrid(fig, 111, (n_slices, n_params), cbar_mode="edge",
                    cbar_location="bottom", axes_pad=0.1, cbar_pad=0.1, share_all=True)

    for row, z in enumerate(slices):
        axes = grid[row * n_params:(row + 1) * n_params]
        title_row = slice_titles[row] if slice_titles else f"SLICE {z+1}"
        _plot_parameter_map_slice(parameter_maps[..., z], label_parameters, axes,
                                  slice_title=title_row, param_titles=(row == 0))

    for i, ax in enumerate(axes):
        param = label_parameters[i]
        _remove_ticks(ax)
        cax = grid.cbar_axes[i]
        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(norm=COLORNORMS[param], cmap=COLORMAPS[param]),
            cax=cax, ticks=PARAMETER_LIMS_COLORMAP[param],
            orientation="horizontal", extend="both"
        )
        cbar.set_ticklabels(PARAMETER_LIMS_COLORMAP[param], color=figure_color, fontsize=10)
        cbar.outline.set_edgecolor("none")

    if title:
        plt.suptitle(title, y=0.1, **figure_title_kwargs)

    if file_name:
        path_save = os.path.join(path_to_data, f"{file_name}.{saving_format}")
        os.makedirs(os.path.dirname(path_save), exist_ok=True)
        plt.savefig(path_save, format=saving_format, bbox_inches="tight")

    return fig


def _plot_parameter_map_slice(
    parameter_maps: NDArray,
    label_parameters: List[str],
    axes: List[Axes],
    slice_title: str = "",
    param_titles: bool = False,
    plot_colorbar: bool = False
) -> None:
    """ 
    Plot all parameter maps of one slice. 

    Parameters
    ----------
    parameter_maps: 3d array of shape (n_parameters, n_x, n_y)
        Array countaining the parameter maps to plot. 
    label_parameters: List[str]
        List of label parameters. 
    axes: List[mpl.axes.Axes] of length n_parameters
        List of axes used to plot the parameter maps. 
    slice_title: str, optional
        Label to add to the slice. Default does not add any title. 
    param_titles: bool, optional
        If True, add a title to the maps with associated label. Default to False. 
    plot_colorar: bool, optional
        If true, add the scaling colorbar to the plot. Default to False. 
    """
    if slice_title:
        axes[0].set_ylabel(slice_title, **slice_title_kwargs)

    for i, ax in enumerate(axes):
        param = label_parameters[i]
        _remove_ticks(ax)

        if param_titles:
            title = PARAMETER_DISPLAYED_LABELS[param]
            if PARAMETER_UNITS[param]:
                title += f" ({PARAMETER_UNITS[param]})"
            ax.set_title(title, **param_title_kwargs)

        ax.imshow(parameter_maps[i], cmap=COLORMAPS[param], aspect="equal",
                  vmin=PARAMETER_LIMS_COLORMAP[param][0],
                  vmax=PARAMETER_LIMS_COLORMAP[param][1],
                  interpolation="none")

        if plot_colorbar:
            plt.colorbar(mpl.cm.ScalarMappable(norm=COLORNORMS[param], cmap=COLORMAPS[param]),
                         ax=ax, ticks=PARAMETER_LIMS_COLORMAP[param],
                         shrink=0.75, location="bottom")



def _remove_ticks(ax: Axes) -> None:
    """Remove axis ticks and borders."""
    ax.tick_params(
        axis='both', which='both',
        bottom=False, top=False, left=False, right=False,
        labelbottom=False, labelleft=False
    )
    for spine in ax.spines.values():
        spine.set_color('black')