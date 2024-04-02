import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from matplotlib import colormaps
from matplotlib.ticker import FormatStrFormatter

from .utils import process_ax, specify_x_positions_for_points


def plot_performance_ranges(transformers: dict,
                            ml: dict,
                            dl: dict,
                            data_path: str,
                            models_colors: dict,
                            comparable_only: bool = True,
                            nrows: int = 3,
                            ncols: int = 3,
                            width: int = 17,
                            height: int = 12,
                            title: str = 'Performance Ranges From the Reviewed Articles',
                            fig_name: str = 'performance_ranges'):
    """
    This function outputs a plot for each dataset in classification and regression tasks. The plot contains three
    entries, transformer models, classical machine learning (ML) models and deep learning (DL) models. in each entry,
    a point (average performance over multiple runs) of the corresponding model for the dataset is plotted. The
    classical ML and DL points are obtained from the corresponding articles (therefore, they are plotted using the same
    mark as the corresponding transformer model)
    :param models_colors: a dictionary that assigns a color per model to use for plotting
    :param transformers: dictionary of two entries, classification and regression. each entry contains a pd.DataFrame
    of the performance of the transformer models on some datasets
    :param ml: dictionary of two entries, classification and regression. each entry contains a pd.DataFrame
    of the performance of the classical ML models on some datasets. These values are obtained from the corresponding
    transformer models.
    :param dl: dictionary of two entries, classification and regression. each entry contains a pd.DataFrame
    of the performance of the DL models on some datasets. These values are obtained from the corresponding
    transformer models.
    :param data_path: the path of the performance files
    :param comparable_only: a flag to specify which figure to output, the ML and DL models that are guaranteed to be
    tested on the same test set in the corresponding transformer model, or all transformer models with all their
    reported values.
    :param nrows: number of rows in the figure
    :param ncols: number of columns in the figure
    :param width: the figure's width
    :param height:the figure's height
    :param title: the figure's title
    :param fig_name: the figure's name to be saved
    :return: the functions saves a figure of the performance
    """

    fig, axs = plt.subplots(nrows, ncols, figsize=(width, height))

    if comparable_only:
        fig_name = f'{fig_name}_honest'

    # starting plotting
    cur_row = 0
    cur_col = 0
    to_next_row = False
    for ml_task, df in transformers.items():  # loop over classification and regression tasks
        if ml_task == 'classification':
            ylabel = 'ROC-AUC'
        else:
            ylabel = 'RMSE'

        # loop over datasets in each category. Last column is the categorical 'source_transformer'
        for ds_name in df.columns[0:-1]:
            ax, cur_row, cur_col, to_next_row = process_ax(nrows, ncols, cur_row, cur_col, to_next_row, axs, ylabel)

            # extract the values for the corresponding dataset
            transformers_ds = transformers[ml_task][ds_name]
            ml_ds = ml[ml_task][ds_name]
            dl_ds = dl[ml_task][ds_name]

            # identifying the transformer models that were tested for this dataset AND compared against ML/DL models
            source_transformers = transformers[ml_task].loc[transformers_ds.index, 'source_transformer'].to_list()
            source_transformers.extend(ml[ml_task].loc[ml_ds.index, 'source_transformer'])
            source_transformers.extend(dl[ml_task].loc[dl_ds.index, 'source_transformer'])

            # creat a dataframe with the selected columns from the different dataframes and prefix the col name with the
            # model category it came from
            model_categories = [f'Transformers {ds_name}', f'ML {ds_name}', f'DL {ds_name}']
            df = pd.concat([transformers_ds, ml_ds, dl_ds], axis=1)
            df.columns = model_categories
            df['source_transformer'] = source_transformers

            # models_colors are defined w.r.t the transformer models that performed ML/DL comparison. Therefore,
            # for Figure 2, the transformer models with no comparable ML/DL models are removed from the
            # plot
            df = df[df['source_transformer'].isin(list(models_colors.keys()))]

            # the current frame is wide (i.e., the metric is reported per category in an independent column). The melt
            # function aggregates all categories into one column and the corresponding metric into another column
            # (i.e., long format). seaborn's stripplot function requires the data to be in a long format when the 'hue'
            # argument is used.
            df = df.melt('source_transformer', var_name='category', value_name='metric', ignore_index=False).dropna()

            # seaborn.stripplot gets crazy when multiple indices with NA are present. So, the NA values were dropped
            # from df, and in the below, we re-add a NA value when a model category is missing (e.g., transformers, ML,
            # or DL). By ensuring the presence of all categories, we ensure that each column will be plotted in the same
            # place in the subplots.
            for model_category in model_categories:
                if model_category not in df['category'].to_list():
                    for curr_model in df['source_transformer'].unique().tolist():
                        new_row = pd.DataFrame({f'{curr_model}_temp': [curr_model, model_category, None]}).transpose()
                        new_row.columns = df.columns.to_list()
                        df = pd.concat([df, new_row])

            # sort the models' categories to ensure that they are plotted in the same place in each subplot
            df.sort_values('category', ascending=False, inplace=True)

            # The 'dodge' argument seperated the points from each model alongside the x-axis. When there are more than
            # two models, this feature makes the plot's visibility better.
            dodge = False
            if len(df['source_transformer'].unique().tolist()) > 2:
                dodge = True

            # plot each model category in a separate column
            sns.stripplot(data=df, x='category', y='metric', hue='source_transformer', ax=ax, edgecolor='black',
                          linewidth=0.3, jitter=True, dodge=dodge, palette=models_colors)

            # specifying the xtickslabels because seaborn only adds the xlabel. the xlabel is also removed because it is
            # not that informative
            labels = [label.split(' ')[0] for label in df['category'].unique().tolist()]
            labels_pos = np.arange(len(labels))
            ax.set(xticks=labels_pos, xlabel=None)
            ax.set_xticklabels(labels, fontsize=14)
            ax.legend().set_visible(False)

            # seaborn forces the ylabel on the plots, but we already add the ylabel for the first subplot in each row
            if cur_col - 1 != 0:
                ax.set_ylabel(None)

            # we used the melt function to convert wide frame to long. Here, the pivot function reverses the process. We
            # do so to have each model category in a separate column again. Therefore, the min and max functions will be
            # calculated for each column. We calculate the min and max for each category to add a dashed line at the min
            # and max values in each column in the plot for easier comparison.
            df = df.drop('source_transformer', axis=1).pivot(columns='category')['metric']
            df = df.reindex(sorted(df.columns, reverse=True), axis=1)
            min_vals = df.min()
            max_vals = df.max()
            hline_color = 'gainsboro'
            hline_length = 0.3
            linewidth = 2
            if ml_task == 'classification':
                shift = 0.01
            else:
                shift = 0.02
            for j in range(len(min_vals)):
                min_val = min_vals.iloc[j]
                max_val = max_vals.iloc[j]
                if min_val == max_val:
                    ax.hlines(max_val, j - hline_length, j + hline_length, colors=hline_color, linestyles='--',
                              linewidth=linewidth)
                    ax.text(j, max_val + 0.01, f'{max_val:.3f}', ha='center', va='bottom',
                            fontsize=10,
                            color='k')
                else:
                    ax.hlines(min_val, j - hline_length, j + hline_length, colors=hline_color, linestyles='--',
                              linewidth=linewidth)
                    ax.hlines(max_val, j - hline_length, j + hline_length, colors=hline_color, linestyles='--',
                              linewidth=linewidth)
                    ax.text(j, min_val - shift, f'{min_val:.3f}', ha='center', va='top', fontsize=10, color='k')
                    ax.text(j, max_val + 0.01, f'{max_val:.3f}', ha='center', va='bottom', fontsize=10,
                            color='k')

            # subplot title
            ax.set_title(ds_name, fontsize=16)
            if ml_task == 'classification':
                ax.set_ylim(.5, 1)
            else:
                ax.set_ylim(0, ax.get_ylim()[1] + 0.5)

    legend = [Line2D([0], [0], marker='o', color=mark, label=mod, lw=0)
              for mod, mark in models_colors.items()]
    fig.legend(handles=legend, loc='center right', fontsize=16, bbox_to_anchor=(1, 0.5))

    # adjust empty spaces around the figure
    plt.subplots_adjust(right=0.85, left=0.05, bottom=0.05, top=0.9, hspace=0.3)

    # Figure title
    fig.suptitle(title, fontsize=20)

    plt.savefig(os.path.join(data_path, f'{fig_name}.png'), dpi=600, bbox_inches='tight')

    plt.show()


def plot_molformer_by_size(data_path: str):
    """
    This function outputs a figure for the performance of MolFormer's different models on the regression and
    classification datasets. The models were trained with different weighted combinations of ZINC and ChEMBL.
    :param data_path: the path for the data files
    :return: this function saves the plotted figure into the specified data_path
    """
    sorting_col = 'size (B)'
    molformer_classification = pd.read_csv(os.path.join(data_path, f'molformer_classification.csv'),
                                           index_col=0).sort_values(by=sorting_col).drop(sorting_col, axis=1)
    molformer_regression = (pd.read_csv(os.path.join(data_path, f'molformer_regression.csv'),
                                        index_col=0).sort_values(by=sorting_col)
                            .drop([sorting_col, 'QM8 (MAE)', 'QM9 (MAE)'], axis=1))

    # This dataset was ignored because the reported values of the individual tasks varied dramatically and normalization
    # was not performed
    # molformer_qm9 = pd.read_csv(os.path.join(data_path, f'molformer_qm9.csv'),
    # index_col=0).sort_values(by=sorting_col).drop(sorting_col, axis=1)
    dfs = [molformer_classification, molformer_regression]  # , molformer_qm9
    fig, axs = plt.subplots(1, len(dfs), figsize=(15, 5))
    ylabels = {0: 'ROC-AUC', 1: 'RMSE', 2: 'MAE'}
    titles = {0: 'Classification (↑)', 1: 'Regression (↓)'}  # , 2: 'QM9 (Regression (↓))'}
    # text = {0: '(A)', 1: '(B)', 2: '(C)'}

    for i, df in enumerate(dfs):
        ax = axs[i]
        ax.set_ylabel(ylabels[i], fontsize=14)
        ax.set_title(titles[i], fontsize=14)
        df.plot(ax=ax, marker='o').legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=14)

    fig_title = 'MolFormer Performance by Pre-Train Dataset Size'
    fig_name = fig_title.lower().replace(' ', '_').replace('-', '')
    fig.suptitle(fig_title, fontsize=20)
    plt.subplots_adjust(top=0.8, wspace=0.5)
    plt.savefig(os.path.join(data_path, f'{fig_name}.png'), dpi=600, bbox_inches='tight')


def plot_chemberta_2(data_path: str):
    df = pd.read_csv(os.path.join(data_path, 'chemBERTa-2.csv'), index_col=0)
    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    handles = None
    legends = None
    for i, col in enumerate(df.columns[1:]):
        ax = axs[i]
        if i == 0:
            ax.set_ylabel('ROC-AUC', fontsize=14)
        ax.set_xlabel('Number of Molecules', fontsize=12)
        ax.set_title(col, fontsize=16)
        mlm = df[df['objective'] == 'MLM'][col].round(3)
        mtr = df[df['objective'] == 'MTR'][col].round(3)

        mlm.plot(ax=ax, marker='o', label='MLM')
        mtr.plot(ax=ax, marker='o', label='MTR')

        if i == 0:
            handles, legends = ax.get_legend_handles_labels()
        # ax.legend(['MLM', 'MTR'])

    fig.legend(handles, legends, loc='center right', fontsize=16, bbox_to_anchor=(1.01, 0.5))

    fig_title = 'ChemBERTa-2 Performance by Pre-Train Dataset Size'
    fig_name = fig_title.lower().replace(' ', '_').replace('-', '')
    fig.suptitle(f'{fig_title}\nClassification (↑)', fontsize=20)
    plt.subplots_adjust(top=0.75, wspace=0.2)
    plt.savefig(os.path.join(data_path, f'{fig_name}.png'), dpi=600, bbox_inches='tight')


def plot_performance_by_representation_or_objectives(data_path: str,
                                                     mol_bert: bool = True,
                                                     mat: bool = False,
                                                     k_bert: bool = False,
                                                     molbert: bool = False,
                                                     objectives: bool = False,
                                                     stats: str = 'SD',
                                                     leg_ncols: int = 2,
                                                     nrows: int = 1,
                                                     figsize: tuple[int, int] = (10, 4),
                                                     sharey: bool = False,
                                                     top: float = 0.65,
                                                     bottom: float = 0.05,
                                                     left: float = 0.1,
                                                     right: float = 0.95,
                                                     hspace: float = 0.1,
                                                     wspace: float = 0.5,
                                                     bbox_to_anchor: tuple[float, float] = (0.5, 0.86),
                                                     fig_name: str | None = None,
                                                     round_val: int = 2,
                                                     png_name: str = 'representation'):
    if objectives:
        png_name = 'Pre-training Objectives'
        leg_ncols = 4

    if mat:
        data = pd.read_csv(os.path.join(data_path, 'mat.csv'), index_col=0)
        ncols = 3
        model_name = 'MAT'
        if objectives:
            fig_name = 'MAT Ablation'
            round_val = 3
    elif molbert:
        data = pd.read_csv(os.path.join(data_path, 'molbert.csv'), index_col=0)
        ncols = 1
        model_name = 'MolBERT'
        figsize = (8, 5)
        leg_ncols = 3
        bbox_to_anchor = (0.5, 0.91)
        fig_name = 'MolBERT Ablation'
    elif k_bert:
        data = pd.read_csv(os.path.join(data_path, 'k-bert.csv'), index_col=0)
        ncols = 5
        nrows = 3
        figsize = (15, 12)
        sharey = True
        model_name = 'K-BERT'
        top = 0.9
        bottom = 0.05
        left = 0.1
        right = 0.95
        hspace = 0.2
        wspace = 0.2
        bbox_to_anchor = (0.5, 0.96)
        fig_name = 'K-BERT Ablation'
    elif mol_bert:
        data = pd.read_csv(os.path.join(data_path, 'mol-bert.csv'), index_col=0)
        ncols = 4
        sharey = True
        model_name = 'Mol-BERT'
        stats = 'SE'
    else:
        raise NotImplemented

    ds_s = [ds for ds in data.columns.to_list() if not ds.endswith(f'{stats}')]
    models = data.index.to_list()

    cmap = colormaps.get_cmap("tab10")
    models_colors = {model: cmap(i) for i, model in enumerate(models)}

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey)

    legend_handles = []
    legend_labels = []

    cur_row = 0
    cur_col = 0
    to_next_row = False
    for i, ds in enumerate(ds_s):
        ylabel = f'ROC-AUC $\pm$ {stats}'
        if mat:
            if ds == 'BBBP':
                ylabel = f'ROC-AUC $\pm$ {stats}'
            else:
                ylabel = f'RMSE $\pm$ {stats}'

        ax, cur_row, cur_col, to_next_row = process_ax(nrows,
                                                       ncols,
                                                       cur_row,
                                                       cur_col,
                                                       to_next_row,
                                                       axs,
                                                       ylabel,
                                                       ylabel_fontsize=12)

        pos = specify_x_positions_for_points(ax, models, molbert)

        for j, model in enumerate(models):
            val = data.loc[model, ds]
            err = data.loc[model, f"{ds}_{stats}"]

            color = models_colors[model]
            line = ax.errorbar(pos[j], val,
                               yerr=err,
                               capsize=4, markersize=4, linewidth=1, marker="o",
                               color=color)

            val_max = val + err
            val_min = val - err
            ax.text(pos[j], val_max, round(val_max, round_val), ha='center', va='bottom', fontsize=10, color='black')

            if mat:
                if ds == 'BBBP':
                    margin = 0.0065
                elif ds == 'ESOL':
                    margin = 0.012
                    if objectives:
                        margin = 0.0075
                else:
                    margin = 0.02
                    if objectives:
                        margin = 0.01
            elif molbert:
                margin = 0.04
            else:
                margin = 0.035
            ax.text(pos[j], val_min - margin, round(val_min, round_val), ha='center', va='bottom', fontsize=10,
                    color='black')

            if i == 0:
                legend_handles.append(line)
                legend_labels.append(model)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        # Remove xticks as they will be defined in legends
        ax.set_xticks([])

        # Set x label
        if mat and ds != 'BBBP':
            ax.set_title(f'{ds} (↓)', fontsize=12)
        else:
            ax.set_title(f'{ds} (↑)', fontsize=12)

        # change the upper ylim to 1.05 and leave the lower lim as is. This allows the text to stay within the plot
        # borders
        ylim_low, ylim_high = ax.get_ylim()
        if mat:
            if ylim_low < 0:
                ax.set_ylim(ylim_low - 0.04, ylim_high + 0.02)
            else:
                ax.set_ylim(ylim_low - 0.02, ylim_high + 0.02)
        elif molbert:
            ax.set_ylim(0.29, 0.85)
        else:
            ax.set_ylim(0.5, 1.0)

    fig.legend(handles=legend_handles,
               labels=legend_labels,
               loc='upper center',
               bbox_to_anchor=bbox_to_anchor,
               ncols=leg_ncols,
               fontsize=14)
    plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
    if fig_name is not None:
        plt.suptitle(fig_name, fontsize=20)
    plt.savefig(os.path.join(data_path, f'{model_name.lower()}_{png_name}.png'))
