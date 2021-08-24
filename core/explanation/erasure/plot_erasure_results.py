import matplotlib.pyplot as plt
import pandas as pd
from core.explanation.erasure.extractors import AggregateDeltaPredictionScores


def plot_delta_scores(delta_scores, plot_target_data_type='all', plot_target_delta_metric='jsd',
                      ignore_single_token: bool = True, out_plot_name: str = None):

    if len(delta_scores) == 1:
        nrows = ncols = 1
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 12))
        axes = [ax]
    else:
        nrows = 3
        ncols = 4
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 12), sharey=True)
        axes = axes.flat

    for idx, use_case in enumerate(delta_scores):
        uc_delta_scores = delta_scores[use_case]
        delta_score_aggregator = AggregateDeltaPredictionScores(uc_delta_scores)
        delta_grouped_data = delta_score_aggregator.get_grouped_data()

        ax = axes[idx]
        group_plot_data = {}
        for method in delta_grouped_data:
            delta_scores_by_method = delta_grouped_data[method]

            if method == 'common_synonym':
                method = 'syn'

            for text_unit in delta_scores_by_method:
                text_unit_scores = delta_scores_by_method[text_unit]
                for pair_or_single in text_unit_scores:
                    pair_or_single_scores = text_unit_scores[pair_or_single]

                    if ignore_single_token:
                        if pair_or_single != 'pair':
                            continue

                    if text_unit == 'sent':
                        key = f'{method}\n{pair_or_single}'
                    else:
                        key = f'{method}\n{text_unit}\n{pair_or_single}'
                    group_plot_data[key] = pair_or_single_scores[plot_target_delta_metric][plot_target_data_type]

        # if experiment_conf['text_unit'] == 'attr' and len(group_plot_data) == len(sample.columns):
        #     aligned_group_plot_data = {}
        #     for col in sample.columns:
        #         aligned_key = None
        #         for key in group_plot_data:
        #             if col in key.split("&"):
        #                 aligned_key = key
        #                 break
        #         assert aligned_key is not None
        #         aligned_group_plot_data[col] = group_plot_data[aligned_key]
        #     group_plot_data = aligned_group_plot_data.copy()

        group_plot_data_tab = pd.DataFrame(group_plot_data)
        # replace 0 with the min value for each row
        group_plot_data_tab = group_plot_data_tab.apply(lambda x: x.replace(0, x[x > 0].min()), axis=1)
        group_plot_data_tab.boxplot(ax=ax)
        ax.set_title(use_case)
        ax.set_yscale('log')

        if idx % ncols == 0:
            ax.set_ylabel('Delta score distribution')

    plt.subplots_adjust(wspace=0.05, hspace=0.3)
    if out_plot_name is not None:
        plt.savefig(out_plot_name, bbox_inches='tight')
    plt.show()


def plot_erasure_method_hits(hit_stats: dict, data_categories: list = None):

    first_use_case = list(hit_stats.keys())[0]
    num_method_pairs = len(hit_stats[first_use_case])

    cat_map = {
        'all': 'all', 'all_pos': 'match', 'all_pred_pos': 'pred\nmatch',
        'all_neg': 'no-match', 'all_pred_neg': 'pred\nno-match',
        'tp': 'tp', 'tn': 'tn', 'fp': 'fp', 'fn': 'fn'
    }

    for i in range(num_method_pairs):

        if len(hit_stats) == 1 or (data_categories is not None and len(data_categories) == 1):
            nrows = ncols = 1
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 12))
            axes = [ax]
        else:
            nrows = 3
            ncols = 4
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(22, 10), sharey=True)
            axes = axes.flat

        single_plot_hits = {}
        for idx, use_case in enumerate(hit_stats):
            use_case_hit_stats = hit_stats[use_case][i]

            hits_to_plot = {}
            for cat in use_case_hit_stats:
                if data_categories is not None:
                    if cat not in data_categories:
                        continue

                if use_case_hit_stats[cat] is None:
                    continue

                hits_to_plot[cat] = use_case_hit_stats[cat]

            if len(hits_to_plot) == 1:
                single_cat = list(hits_to_plot.keys())[0]
                if single_cat not in single_plot_hits:
                    single_plot_hits[single_cat] = [hits_to_plot[single_cat]]
                else:
                    single_plot_hits[single_cat].append(hits_to_plot[single_cat])

            else:
                multi_cat_hits_table = pd.DataFrame(list(hits_to_plot.values()), index=list(hits_to_plot))
                multi_cat_hits_table = multi_cat_hits_table.rename(index=cat_map)
                multi_cat_hits_table.plot(kind='bar', stacked=True, ax=axes[idx], rot=0, legend=False)
                axes[idx].set_title(use_case, fontsize=18)
                if idx % ncols == 0:
                    axes[idx].set_ylabel('Hit percentage', fontsize=20)
                axes[idx].xaxis.set_tick_params(labelsize=16)
                axes[idx].yaxis.set_tick_params(labelsize=20)
                axes[idx].axhline(y=50, color='r', linestyle='-')

        if len(single_plot_hits) > 0:
            single_cat = list(single_plot_hits.keys())[0]
            single_plot_hits_table = pd.DataFrame(single_plot_hits[single_cat], index=list(hit_stats))
            single_plot_hits_table.plot(kind='bar', stacked=True, ax=axes[0])
            axes[0].set_title(f'Erasure hits for the data category {cat_map[single_cat]}')
            axes[0].set_ylabel('Hit percentage')
            axes[0].axhline(y=50, color='r', linestyle='-')

        else:
            handles, labels = axes[idx].get_legend_handles_labels()
            fig.legend(handles, labels, bbox_to_anchor=(0.66, 0.08), ncol=2, fontsize=18)
            plt.subplots_adjust(wspace=0.05, hspace=0.5)

        plt.show()



