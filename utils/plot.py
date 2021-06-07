from matplotlib import pyplot as plt
import math
import seaborn as sns
from utils.result_collector import TestResultCollector


def plot_layers_heads_attention(attns, mask=None, out_file_name: str = None):
    x = attns.shape[0]
    y = attns.shape[1]

    if mask is not None:

        assert attns.shape[:2] == mask.shape

        nplots = mask.sum()
        plot_grid_size = math.floor(math.sqrt(nplots))
        if plot_grid_size * plot_grid_size == nplots:
            nrows, ncols = plot_grid_size, plot_grid_size
        else:
            nrows, ncols = plot_grid_size + 1, plot_grid_size + 1
    else:
        nrows = attns.shape[0]
        ncols = attns.shape[1]

    figsize = (10, 10)
    if nrows * ncols > 25:
        figsize = (20, 20)
    fig, axes = plt.subplots(nrows, ncols, sharey=True, figsize=figsize)

    count = 0
    for i in range(x):
        for j in range(y):

            plt_x = count // nrows
            plt_y = count % ncols

            if mask is not None:
                if mask[i][j] > 0:
                    ax = axes[plt_x][plt_y]
                    ax.set_title(f"L: {i}, H: {j}")
                    sns.heatmap(attns[i][j], ax=ax, cbar=False)
                    count += 1
            else:
                ax = axes[plt_x][plt_y]
                ax.set_title(f"L: {i}, H: {j}")
                sns.heatmap(attns[i][j], ax=ax, cbar=False)
                count += 1
    plt.subplots_adjust(hspace=0.5)

    if out_file_name:
        plt.savefig(out_file_name, bbox_inches='tight')

    plt.show()


def plot_results(results, tester, target_cats=None, plot_params=None, vmin=0, vmax=1):
    assert isinstance(results, dict)
    if target_cats is not None:
        assert isinstance(target_cats, list)
        assert len(target_cats) > 0

    if plot_params is None:
        plot_params = ['match_attr_attn_loc', 'match_attr_attn_over_mean',
                       'avg_attr_attn', 'attr_attn_3_last', 'attr_attn_last_1',
                       'attr_attn_last_2', 'attr_attn_last_3',
                       'avg_attr_attn_3_last', 'avg_attr_attn_last_1',
                       'avg_attr_attn_last_2', 'avg_attr_attn_last_3']

    for use_case in results:
        print(use_case)
        uc_res = results[use_case]

        for cat, cat_res in uc_res.items():

            if cat_res is None:
                continue

            if target_cats:
                if cat not in target_cats:
                    continue

            print(cat)
            assert isinstance(cat_res, TestResultCollector)

            tester.plot(cat_res, plot_params=plot_params, labels=True, vmin=vmin, vmax=vmax, title_prefix=use_case)


def plot_comparison(res1, res2, cmp_res, tester, cmp_vals, target_cats=None, plot_params=None):
    assert isinstance(res1, dict)
    assert isinstance(res2, dict)
    assert isinstance(cmp_res, dict)
    assert isinstance(cmp_vals, list)
    assert len(cmp_vals) == 2
    if target_cats is not None:
        assert isinstance(target_cats, list)
        assert len(target_cats) > 0

    if plot_params is None:
        plot_params = ['match_attr_attn_loc', 'match_attr_attn_over_mean',
                       'avg_attr_attn', 'attr_attn_3_last', 'attr_attn_last_1',
                       'attr_attn_last_2', 'attr_attn_last_3',
                       'avg_attr_attn_3_last', 'avg_attr_attn_last_1',
                       'avg_attr_attn_last_2', 'avg_attr_attn_last_3']

    for cat in set(res1).intersection(set(res2)):
        cat_res1 = res1[cat]
        cat_res2 = res2[cat]
        cmp_cat_res = cmp_res[cat]

        if cat_res1 is None or cat_res2 is None or cmp_cat_res is None:
            continue

        if target_cats:
            if cat not in target_cats:
                continue

        print(cat)
        assert isinstance(cat_res1, TestResultCollector)
        assert isinstance(cat_res2, TestResultCollector)
        assert isinstance(cmp_cat_res, TestResultCollector)

        tester.plot_comparison(cat_res1, cat_res2, cmp_cat_res, plot_params=plot_params, labels=True,
                               title_prefix=f"{cmp_vals[0]}_vs_{cmp_vals[1]}")


def plot_benchmark_results(results, tester, use_cases, target_cats=None, plot_params=None, title_prefix=None, vmin=0,
                           vmax=1):
    assert isinstance(results, dict)
    assert isinstance(use_cases, list)
    assert len(use_cases) > 0
    for use_case in use_cases:
        assert use_case in results
    if target_cats is not None:
        assert isinstance(target_cats, list)
        assert len(target_cats) > 0
    if title_prefix is not None:
        assert isinstance(title_prefix, str)

    first_use_case = use_cases[0]
    first_results = results[first_use_case]
    cats = list(first_results)

    if plot_params is None:
        plot_params = ['match_attr_attn_loc', 'match_attr_attn_over_mean',
                       'avg_attr_attn', 'attr_attn_3_last', 'attr_attn_last_1',
                       'attr_attn_last_2', 'attr_attn_last_3',
                       'avg_attr_attn_3_last', 'avg_attr_attn_last_1',
                       'avg_attr_attn_last_2', 'avg_attr_attn_last_3']

    for cat in cats:

        if target_cats:
            if cat not in target_cats:
                continue

        print(cat)

        for plot_param in plot_params:
            nrows = 3
            ncols = 4
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 15))

            title = f'{plot_param}'
            if title_prefix is not None:
                title = f'{title_prefix} {title}'
            fig.suptitle(title)

            for idx, use_case in enumerate(use_cases):
                ax = axes[idx // ncols][idx % ncols]
                ax.set_title(use_case)
                cat_res = results[use_case][cat]

                if cat_res is None:
                    continue

                assert isinstance(cat_res, TestResultCollector)

                labels = False
                if idx % ncols == 0:
                    labels = True

                tester.plot(cat_res, plot_params=[plot_param], ax=ax, labels=labels,
                            vmin=vmin, vmax=vmax)
            plt.subplots_adjust(wspace=0.01, hspace=0.3)
            plt.show()


def plot_agg_results(results, target_cats=None, title_prefix=None, xlabel=None, ylabel=None,
                     xticks=None, yticks=None, agg=False, vmin=0, vmax=0.5):
    assert isinstance(results, dict)
    if target_cats is not None:
        assert isinstance(target_cats, list)
        assert len(target_cats) > 0
    if title_prefix is not None:
        assert isinstance(title_prefix, str)

    for cat in results:

        if target_cats is not None:
            if cat not in target_cats:
                continue

        cat_res = results[cat]
        print(cat)

        for metric in cat_res:

            print(metric)

            assert isinstance(cat_res[metric], dict)
            assert len(cat_res[metric]) == 1
            res_id = list(cat_res[metric].keys())[0]

            figsize = (20, 10)
            if cat_res[metric][res_id].shape[1] < 3:
                figsize = (6, 10)
            fig, ax = plt.subplots(figsize=figsize)

            title = f'{res_id} {metric}'
            if title_prefix is not None:
                title = f'{title_prefix} {title}'
            fig.suptitle(title)

            sns.heatmap(cat_res[metric][res_id], annot=True, fmt='.2f', vmin=vmin, vmax=vmax,
                        xticklabels=xticks, ax=ax)
            plt.show()

            if agg:
                fig, ax = plt.subplots(figsize=(5, 4))
                fig.suptitle(f'{res_id} {metric} agg')
                sns.heatmap(cat_res[metric][res_id].mean(1).reshape((-1, 1)), annot=True,
                            fmt='.2f', vmin=vmin, vmax=vmax, ax=ax)
                plt.show()
