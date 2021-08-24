import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.explanation.gradient.extractors import EntityGradientExtractor
import os
import pickle


def plot_grads(grad_data: dict, target_entity: str, title: str = None, out_plot_name: str = None,
               ignore_special: bool = False, max_y = None):
    assert isinstance(grad_data, dict), "Wrong data type for parameter 'grad_data'."
    params = ['all', 'all_grad', 'left', 'left_grad', 'right', 'right_grad']
    assert all([p in grad_data for p in params]), "Wrong data format for parameter 'grad_data'."
    entities = ['all', 'left', 'right']
    assert isinstance(target_entity, str), "Wrong data type for parameter 'target_entity'."
    assert target_entity in entities, f"Wrong target entity: {target_entity} not in {entities}."
    if title is not None:
        assert isinstance(title, str), "Wrong data type for parameter 'title'."

    plt.subplots(figsize=(20, 10))

    x = grad_data[target_entity]
    sep_idxs = list(np.where(np.array(x) == '[SEP]')[0])
    skip_idxs = [0] + sep_idxs
    if ignore_special:
        if x[0] == '[CLS]':
            x = [x[i] for i in range(len(x)) if i not in skip_idxs]

    # check for duplicated labels
    label_counts = pd.Series(x).value_counts()
    not_unique_labels = label_counts[label_counts > 1]
    if len(not_unique_labels) > 0:
        new_x = x.copy()
        for nul in list(not_unique_labels.index):
            nul_idxs = np.where(np.array(x) == nul)[0]
            for i, nul_idx in enumerate(nul_idxs, 1):
                new_x[nul_idx] = f'{nul}_{i}'  # concatenating the duplicated label with an incremental id
        x = new_x.copy()

    if isinstance(grad_data[f'{target_entity}_grad'], dict):
        for m in grad_data[f'{target_entity}_grad']:

            if m not in ['avg']:
                continue

            y = grad_data[f'{target_entity}_grad'][m]
            if ignore_special:
                if grad_data[target_entity][0] == '[CLS]':
                    y = [y[i] for i in range(len(y)) if i not in skip_idxs]

            yerr = None
            if f'{target_entity}_error_grad' in grad_data:
                yerr = grad_data[f'{target_entity}_error_grad']
            barlist = plt.bar(x, y, yerr=yerr, label=m)
            max_grads_idxs = np.array(y).argsort()[-4:][::-1]
            for max_grads_idx in max_grads_idxs:
                barlist[max_grads_idx].set_color('r')
    else:
        y = grad_data[f'{target_entity}_grad']
        if ignore_special:
            if grad_data[target_entity][0] == '[CLS]':
                y = [y[i] for i in range(len(y)) if i not in skip_idxs]

        yerr = None
        if f'{target_entity}_error_grad' in grad_data:
            yerr = grad_data[f'{target_entity}_error_grad']
        barlist = plt.bar(x, y, yerr=yerr)
        max_grads_idxs = np.array(y).argsort()[-4:][::-1]
        for max_grads_idx in max_grads_idxs:
            barlist[max_grads_idx].set_color('r')

    plt.xticks(rotation=90)
    if max_y:
        plt.ylim(0, max_y)
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.subplots_adjust(bottom=0.15)
    fig = plt.gcf()
    plt.show()
    decision = input("Press Enter to continue or press S to save the plot and continue...")
    if decision == "S":
        fig.savefig(out_plot_name, bbox_inches='tight')


def plot_batch_grads(grads_data: list, target_entity: str, title_prefix: str = None, out_plot_name: str = None,
                     ignore_special: bool = False):
    EntityGradientExtractor.check_extracted_grad(grads_data)

    # get max gradient to have an upper bound for the y-axis when plotting
    max_grad = 0
    for g in grads_data:
        text_units = g['grad'][f'{target_entity}']
        sep_idxs = list(np.where(np.array(text_units) == '[SEP]')[0])
        skip_idxs = [0] + sep_idxs
        if isinstance(g['grad'][f'{target_entity}_grad'], dict):
            x = g['grad'][f'{target_entity}_grad']['avg']
        else:
            x = g['grad'][f'{target_entity}_grad']
        if ignore_special:
            if text_units[0] == '[CLS]':
                x = [x[i] for i in range(len(x)) if i not in skip_idxs]
        max_g = np.max(x)
        if max_g > max_grad:
            max_grad = max_g

    for idx, grad_data in enumerate(grads_data):

        if grad_data is None:
            logging.info(f"No gradients for item {idx}.")
            continue

        grad = grad_data['grad']
        label = grad_data['label']
        prob = grad_data['prob']
        pred = grad_data['pred']
        title = f"gradients for item#{idx} - label: {label} - pred: {pred} - prob: {prob}"

        if title_prefix is not None:
            title = f'{title_prefix} {title}'

        plot_grads(grad, target_entity, title=title, out_plot_name=f'{out_plot_name}_{idx}.pdf',
                   ignore_special=ignore_special, max_y=max_grad)


def plot_multi_use_case_grads(conf, sampler_conf, fine_tune, grads_conf, use_cases, out_dir, grad_agg_metrics=['avg'],
                              plot_type='box', ignore_special: bool = True, out_plot_name: str = None):
    assert isinstance(use_cases, list), "Wrong data type for parameter 'use_cases'."
    assert len(use_cases) > 0, "Empty use case list."
    grad_agg_available_metrics = ['sum', 'avg', 'median', 'max']
    assert all([m in grad_agg_available_metrics for m in grad_agg_metrics]), "Wrong metric names."
    plot_types = ['box', 'error']
    assert plot_type in plot_types
    if plot_type == 'box':
        assert len(grad_agg_metrics) == 1, "Only one metric supported in the 'box' plot type."

    tok = conf['tok']
    size = sampler_conf['size']
    grad_text_unit = grads_conf['text_unit']
    grad_special_tokens = grads_conf['special_tokens']

    ncols = 4
    nrows = 3
    if len(use_cases) == 1:
        ncols = 1
        nrows = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 12), sharey=True)
    if len(use_cases) > 1:
        axes = axes.flat
    # loop over the use cases
    for idx, use_case in enumerate(use_cases):
        out_fname = f"{use_case}_{tok}_{size}_{fine_tune}_{grad_text_unit}_{grad_special_tokens}_ALL"
        data_path = os.path.join(out_dir, use_case, out_fname)
        # load grads data
        uc_grad = pickle.load(open(f"{data_path}.pkl", "rb"))
        # check grads data format
        EntityGradientExtractor.check_extracted_grad(uc_grad)
        first_data = None
        for item in uc_grad:
            if item is not None:
                first_data = item
                break
        if not first_data:
            continue
        text_unit_names = first_data['grad']['all']
        sep_idxs = list(np.where(np.array(text_unit_names) == '[SEP]')[0])
        skip_idxs = [0] + sep_idxs
        assert all([item['grad']['all'] == text_unit_names for item in uc_grad if item is not None])

        if len(use_cases) > 1:
            ax = axes[idx]
        else:
            ax = axes

        uc_plot_data = {}
        for item in uc_grad:
            if item is not None:
                for m in item['grad']['all_grad']:
                    if m in grad_agg_metrics:

                        x = item['grad']['all_grad'][m]
                        if ignore_special:
                            if text_unit_names[0] == '[CLS]':
                                x = [x[i] for i in range(len(x)) if i not in skip_idxs]

                        if m not in uc_plot_data:
                            uc_plot_data[m] = [x]
                        else:
                            uc_plot_data[m].append(x)

        if ignore_special:
            text_unit_names = [text_unit_names[i] for i in range(len(text_unit_names)) if i not in skip_idxs]

        columns = []
        num_columns = len(text_unit_names)
        if grad_special_tokens and not ignore_special:
            num_columns -= 3
        if not ignore_special:
            columns.append('[CLS]')
        half_columns = num_columns // 2
        for num in range(1, half_columns + 1):
            columns.append(f'l_{num}')
        if not ignore_special:
            columns.append('[SEP]')
        for num in range(1, half_columns + 1):
            columns.append(f'r_{num}')
        if not ignore_special:
            columns.append('[SEP]')

        for metric in uc_plot_data:

            uc_plot_metric_table = pd.DataFrame(uc_plot_data[metric], columns=columns)

            if plot_type == 'error':
                uc_plot_metric_table_stats = uc_plot_metric_table.describe()
                medians = uc_plot_metric_table_stats.loc['50%', :].values
                percs_25 = uc_plot_metric_table_stats.loc['25%', :].values
                percs_75 = uc_plot_metric_table_stats.loc['75%', :].values
                uc_plot_metric_data = {
                    'x': range(len(uc_plot_metric_table_stats.columns)),
                    'y': medians,
                    'yerr': [medians - percs_25, percs_75 - medians],
                }

                ax.errorbar(**uc_plot_metric_data, alpha=.75, fmt=':', capsize=3, capthick=1, label=metric)
                uc_plot_metric_data_area = {
                    'x': uc_plot_metric_data['x'],
                    'y1': percs_25,
                    'y2': percs_75
                }
                ax.fill_between(**uc_plot_metric_data_area, alpha=.25)
                ax.set_xticks(range(len(uc_plot_metric_table.columns)))
                ax.set_xticklabels(uc_plot_metric_table.columns)
                ax.legend()

            elif plot_type == 'box':
                uc_plot_metric_table.boxplot(ax=ax)
            else:
                raise NotImplementedError("Wrong plot type.")

        ax.set_title(use_case)
        if idx % ncols == 0:
            ax.set_ylabel('Gradient')
        ax.set_xlabel('Attributes')

    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    if out_plot_name:
       plt.savefig(out_plot_name, bbox_inches='tight')
    plt.show()


def _plot_top_grad_stats(stats_data: dict, out_plot_name: str = None, stacked=True, share_legend=True, ylabel=None):
    ncols = 4
    nrows = 3
    if len(stats_data) == 1:
        ncols = 1
        nrows = 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10), sharey=True)
    if len(stats_data) > 1:
        axes = axes.flat
    for idx, use_case in enumerate(stats_data):

        if len(stats_data) > 1:
            ax = axes[idx]
        else:
            ax = axes

        use_case_stats = stats_data[use_case]
        use_case_stats.plot(kind='bar', stacked=stacked, ax=ax, legend=not share_legend, rot=0)
        ax.set_title(use_case, fontsize=18)
        if idx % ncols == 0:
            if ylabel is not None:
                ax.set_ylabel(ylabel, fontsize=20)
        ax.xaxis.set_tick_params(labelsize=18)
        ax.yaxis.set_tick_params(labelsize=20)

        # for p in ax.patches:
        #     width, height = p.get_width(), p.get_height()
        #     x, y = p.get_xy()
        #     ax.text(x + width / 2,
        #             y + height / 2,
        #             '{:.0f} %'.format(height),
        #             horizontalalignment='center',
        #             verticalalignment='center')

    if share_legend:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.73, 0.08), ncol=4, fontsize=20)
    plt.subplots_adjust(wspace=0.1, hspace=0.4)
    if out_plot_name:
        plt.savefig(out_plot_name, bbox_inches='tight')
    plt.show()


def plot_top_grad_stats(stats_data: list, out_plot_name: str = None, stacked=True, share_legend=True, ylabel=None):
    for stats in stats_data:
        _plot_top_grad_stats(stats, out_plot_name=out_plot_name, stacked=stacked, share_legend=share_legend,
                             ylabel=ylabel)
