import math

import matplotlib.pyplot as plt


def process_ax(nrows: int,
               ncols: int,
               cur_row: int,
               cur_col: int,
               to_next_row: bool,
               axs: plt.axes,
               ylabel: str,
               force_ylabel: bool=False,
               ylabel_fontsize: int = 16):
    """
    This function processes a multi-plot figure and ensures that looping columns and rows of the figures happens
    correctly.
    :param nrows: number of rows of the figure
    :param ncols: number of columns of the figure
    :param cur_row: the current row in the figure
    :param cur_col: the current columns in the figure
    :param to_next_row: a flag to indicate whether current row has been completed or not
    :param axs: the subplots of the figure
    :param ylabel: the label to add to the y-axis
    :param force_ylabel: if set to false, then the label is set only to the first plot in each row
    :param ylabel_fontsize: the font size of the y-label
    :return: this function returns the current plot in the graph with the specified y-label as well as updating the
    information for curr_row, curr-col, and to_next_row
    """

    if nrows == 1 and ncols == 1:
        axs.set_ylabel(ylabel, fontsize=ylabel_fontsize)
        return axs, cur_row, cur_col, to_next_row
    if to_next_row:
        cur_row += 1
        cur_col = 0
        to_next_row = False
    if nrows > 1:
        ax = axs[cur_row, cur_col]
    else:
        ax = axs[cur_col]

    if force_ylabel:
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    elif cur_col == 0:
        ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

    cur_col += 1
    if cur_col == ncols:
        to_next_row = True
    return ax, cur_row, cur_col, to_next_row


def specify_x_positions_for_points(ax: plt.axes, models: list, molbert:bool):
    ax.set_xlim(0, math.ceil(len(models) / 2))
    pos = []
    start_cnt = 0.25
    for cnt in range(len(models)):
        if molbert:
            start_cnt += 0.05
        pos.append(start_cnt)
        start_cnt += 0.5
    return pos
