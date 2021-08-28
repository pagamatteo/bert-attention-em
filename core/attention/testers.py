import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils.result_collector import TestResultCollector
from utils.test_utils import ConfCreator
from utils.plot import plot_left_to_right_heatmap


class GenericAttributeAttentionTest(object):
    """
    This class analyzes the attention paid by some model on matching attributes by
    examining its attention maps.
    It produces the following results:
    - lr_match_attr_attn_loc: mask that displays for each layer and head if the
    attention paid by the model on the corresponding attributes of the left and
    right entities is greater than the average attention between each pair of
    attributes
    - rl_match_attr_attn_loc: mask that displays for each layer and head if the
    attention paid by the model on the corresponding attributes of the right and
    left entities is greater than the average attention between each pair of
    attributes
    - match_attr_attn_loc: mask obtained by getting the maximum values between the
    previous two masks
    - match_attr_attn_over_mean: above-average attention paid by the model on each
    pair of attributes of the two entities
    """

    def __init__(self, permute: bool = False, model_attention_grid: tuple = (12, 12), ignore_special: bool = True):
        assert isinstance(permute, bool), "Wrong data type for parameter 'permute'."
        assert isinstance(model_attention_grid, tuple), "Wrong data type for parameter 'model_attention_grid'."
        assert len(model_attention_grid) == 2, "'model_attention_grid' has to specify two dimensions."
        assert model_attention_grid[0] > 0 and model_attention_grid[
            1] > 0, "Wrong value for parameter 'model_attention_grid'."
        assert isinstance(ignore_special, bool), "Wrong data type for parameter 'ignore_special'."

        self.permute = permute
        self.model_attention_grid = model_attention_grid
        self.ignore_special = ignore_special
        self.result_names = ['lr_match_attr_attn_loc', 'rl_match_attr_attn_loc',
                             'match_attr_attn_loc']

        self.property_mask_res = ['match_attr_attn_over_mean', 'avg_attr_attn']
        self.result_names += self.property_mask_res

        mask = np.zeros(model_attention_grid)
        attr_attn_3_last = mask.copy()
        attr_attn_3_last[-3:, :] = 1
        attr_attn_last_1 = mask.copy()
        attr_attn_last_1[-1, :] = 1
        attr_attn_last_2 = mask.copy()
        attr_attn_last_2[-2, :] = 1
        attr_attn_last_3 = mask.copy()
        attr_attn_last_3[-3, :] = 1
        self.cond_prop_mask_res = {'attr_attn_3_last': attr_attn_3_last,
                                   'attr_attn_last_1': attr_attn_last_1,
                                   'attr_attn_last_2': attr_attn_last_2,
                                   'attr_attn_last_3': attr_attn_last_3,
                                   'avg_attr_attn_3_last': attr_attn_3_last.copy(),
                                   'avg_attr_attn_last_1': attr_attn_last_1.copy(),
                                   'avg_attr_attn_last_2': attr_attn_last_2.copy(),
                                   'avg_attr_attn_last_3': attr_attn_last_3.copy(), }
        self.result_names += list(self.cond_prop_mask_res)

    def _test_attr_attention(self, attn_map: np.ndarray):

        assert isinstance(attn_map, np.ndarray), "Wrong data type for parameter 'attn_map'."

        res = {}
        n = attn_map.shape[0] // 2

        # extract the attention between corresponding left-to-right and
        # right-to-left attributes
        lr_match_attr_attn = np.array([])
        rl_match_attr_attn = np.array([])
        for idx in range(n):
            if self.permute:
                lr = attn_map[idx, n - idx].item()
                rl = attn_map[idx - n, idx].item()
            else:
                lr = attn_map[idx, n + idx].item()
                rl = attn_map[idx + n, idx].item()
            lr_match_attr_attn = np.append(lr_match_attr_attn, lr)
            rl_match_attr_attn = np.append(rl_match_attr_attn, rl)

        # check if these attention scores are greater than the average score
        # if all these attention scores are over the mean then output 1 in the mask
        m = attn_map.mean().item()
        res['lr_match_attr_attn_loc'] = int((lr_match_attr_attn > m).sum() >= n - 1)
        res['rl_match_attr_attn_loc'] = int((rl_match_attr_attn > m).sum() >= n - 1)

        # diff_attr_attn_lr = np.array([])
        # diff_attr_attn_rl = np.array([])
        # for idx in range(n):
        #   lr = np.concatenate(
        #       (attn_map[idx, n : n + idx], attn_map[idx, n + idx + 1:])
        #       ).mean().item()
        #   rl = np.concatenate(
        #       (attn_map[n : n + idx, idx], attn_map[n + idx + 1:, idx])
        #       ).mean().item()
        #   diff_attr_attn_lr = np.append(diff_attr_attn_lr, lr)
        #   diff_attr_attn_rl = np.append(diff_attr_attn_rl, rl)
        # res['diff_lr'] = diff_attr_attn_lr
        # res['diff_rl'] = diff_attr_attn_rl

        # save the mask that indicates which pair of attributes generates an
        # attention greater than the average score
        res['match_attr_attn_over_mean'] = attn_map > m
        res['attr_attn_3_last'] = attn_map > m
        res['attr_attn_last_1'] = attn_map > m
        res['attr_attn_last_2'] = attn_map > m
        res['attr_attn_last_3'] = attn_map > m
        res['avg_attr_attn'] = attn_map.copy()
        res['avg_attr_attn_3_last'] = attn_map.copy()
        res['avg_attr_attn_last_1'] = attn_map.copy()
        res['avg_attr_attn_last_2'] = attn_map.copy()
        res['avg_attr_attn_last_3'] = attn_map.copy()

        return res

    def test(self, left_entity: pd.Series, right_entity: pd.Series,
             attn_params: dict):

        assert isinstance(left_entity, pd.Series), "Wrong data type for parameter 'left_entity'."
        assert isinstance(right_entity, pd.Series), "Wrong data type for parameter 'right_entity'."
        assert isinstance(attn_params, dict), "Wrong data type for parameter 'attn_params'."
        assert 'attns' in attn_params, "Attention maps parameter non found."
        assert 'text_units' in attn_params, "Text units non found."

        attr_attns = attn_params['attns']
        text_units = attn_params['text_units']

        if attr_attns is None:
            return None

        n_layers = attr_attns.shape[0]
        n_heads = attr_attns.shape[1]
        n_attrs = attr_attns[0][0].shape[0]
        if self.ignore_special is True and text_units is not None and text_units[0] == '[CLS]':
            sep_idxs = np.where(np.array(text_units) == '[SEP]')[0]
            if len(sep_idxs) > 1:                   # attr-pair mode
                n_attrs -= len(sep_idxs) * 2 + 1
            else:                                   # sent-pair mode
                n_attrs -= 3    # [CLS] + 2 x [SEP]
        assert self.model_attention_grid == (n_layers, n_heads)

        # initialize the result collector
        res_collector = TestResultCollector()
        for result_name in self.result_names:
            if result_name in self.property_mask_res or result_name in self.cond_prop_mask_res:
                res_collector.save_result(np.zeros((n_attrs, n_attrs)), result_name)
            else:
                res_collector.save_result(np.zeros((n_layers, n_heads)), result_name)

        # loop over the attention maps and analyze them
        for layer in range(n_layers):
            for head in range(n_heads):
                attn_map = attr_attns[layer][head]

                # (optional) remove special tokens
                if self.ignore_special is True and text_units is not None and text_units[0] == '[CLS]':
                    complete_text_units = text_units + text_units[1:]
                    sep_idxs = list(np.where(np.array(complete_text_units) == '[SEP]')[0])

                    valid_idxs = np.array(list(set(range(1, len(attn_map))).difference(sep_idxs)))
                    attn_map = attn_map[valid_idxs][:, valid_idxs]

                # analyze the current attention map
                test_res = self._test_attr_attention(attn_map)

                # save the results in the collector
                for result_name in test_res:
                    if result_name in self.property_mask_res:
                        res_collector.transform_result(result_name,
                                                       lambda x: x + test_res[result_name])
                    elif result_name in self.cond_prop_mask_res:
                        if self.cond_prop_mask_res[result_name][layer][head]:
                            res_collector.transform_result(result_name,
                                                           lambda x: x + test_res[result_name])
                    elif result_name in self.result_names:
                        res_collector.update_result_value(layer, head,
                                                          test_res[result_name],
                                                          result_name)
                    else:
                        ValueError("No result name found.")

        # update/add some results
        res_collector.combine_results('lr_match_attr_attn_loc',
                                      'rl_match_attr_attn_loc',
                                      lambda x, y: np.maximum(x, y),
                                      'match_attr_attn_loc')

        for result_name in self.property_mask_res:
            res_collector.transform_result(result_name,
                                           lambda x: x / (n_layers * n_heads))
        for result_name in self.cond_prop_mask_res:
            res_collector.transform_result(result_name,
                                           lambda x: x / (self.cond_prop_mask_res[result_name].sum()))

        return res_collector

    def _check_result_params(self, result: dict):
        assert isinstance(result, dict), "Wrong data type for parameter 'result'."

        params = self.result_names
        assert np.sum([param in result for param in params]) == len(params)

    def plot(self, res_collector: TestResultCollector, plot_params: list = None,
             out_dir=None, out_file_name_prefix=None, title_prefix=None, ax=None, labels=None,
             vmin=0, vmax=1, plot_type='simple'):

        assert isinstance(res_collector, TestResultCollector), "Wrong data type for parameter 'res_collector'."
        result = res_collector.get_results()
        self._check_result_params(result)
        if plot_params is not None:
            assert isinstance(plot_params, list)
            assert len(plot_params) > 0
            for plot_param in plot_params:
                assert plot_param in result
        assert isinstance(plot_type, str)
        assert plot_type in ['simple', 'advanced']

        for param, score in result.items():

            if plot_params is not None:
                if param not in plot_params:
                    continue

            if '_loc' not in param and plot_type == 'advanced':
                map = ConfCreator().use_case_map
                assert out_file_name_prefix is not None
                plot_left_to_right_heatmap(score, vmin=vmin, vmax=vmax, title=map[title_prefix], is_annot=True,
                                           out_file_name=f'{out_file_name_prefix}_{param}.png')

            else:

                if ax is None:
                    fig, new_ax = plt.subplots(figsize=(10, 5))
                    title = param
                    if title_prefix is not None:
                        title = f'{title_prefix}_{title}'
                    fig.suptitle(title)

                else:
                    new_ax = ax

                assert new_ax is not None

                score = score.mean(axis=1).reshape((-1, 1))
                _ = sns.heatmap(score, annot=True, fmt='.1f', ax=new_ax, vmin=vmin, vmax=vmax)
                ylabel = 'layers'
                xlabel = 'heads'
                if param in self.property_mask_res:
                    xlabel, ylabel = 'attributes', 'attributes'

                new_ax.set_xlabel(xlabel)

                if labels:
                    new_ax.set_ylabel(ylabel)
                else:
                    new_ax.set_yticks([])

                if out_dir is not None:
                    if out_file_name_prefix is not None:
                        out_plot_file_name = '{}_{}.pdf'.format(out_file_name_prefix, param)
                    else:
                        out_plot_file_name = '{}.pdf'.format(param)

                    if ax is None:
                        plt.savefig(os.path.join(out_dir, out_plot_file_name), bbox_inches='tight')

                if ax is None:
                    plt.show()

    def plot_comparison(self, res_coll1: TestResultCollector, res_coll2: TestResultCollector,
                        cmp_res_coll: TestResultCollector, plot_params: list = None, out_dir=None,
                        out_file_name_prefix=None, title_prefix=None, labels=None):

        assert isinstance(res_coll1, TestResultCollector), "Wrong data type for parameter 'res_coll1'."
        res1 = res_coll1.get_results()
        self._check_result_params(res1)
        assert isinstance(res_coll2, TestResultCollector), "Wrong data type for parameter 'res_coll2'."
        res2 = res_coll2.get_results()
        self._check_result_params(res2)
        assert isinstance(cmp_res_coll, TestResultCollector), "Wrong data type for parameter 'cmp_res_coll'."
        cmp_res = cmp_res_coll.get_results()
        self._check_result_params(cmp_res)

        if plot_params is not None:
            assert isinstance(plot_params, list)
            assert len(plot_params) > 0
            for plot_param in plot_params:
                assert plot_param in res1
                assert plot_param in res2
                assert plot_param in cmp_res

        for param in res1:
            score1 = res1[param]
            score2 = res2[param]
            cmp_score = cmp_res[param]

            if plot_params is not None:
                if param not in plot_params:
                    continue

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            title = param
            if title_prefix is not None:
                title = f'{title_prefix}_{title}'
            fig.suptitle(title)

            _ = sns.heatmap(score1, annot=True, fmt='.1f', ax=axes[0], vmin=0, vmax=1)
            _ = sns.heatmap(score2, annot=True, fmt='.1f', ax=axes[1], vmin=0, vmax=1)
            _ = sns.heatmap(cmp_score, annot=True, fmt='.1f', ax=axes[2], vmin=-0.5, vmax=0.5)

            # fig1, ax1 = plt.subplots(1, 1, figsize=(5, 4))
            # sns.heatmap(cmp_score, annot=True, fmt='.1f', ax=ax1, vmin=-0.5, vmax=0.5)
            # ax1.set_xlabel('heads')
            # ax1.set_ylabel('layers')
            # plt.savefig(f'{title}.pdf', bbox_inches='tight')

            ylabel = 'layers'
            xlabel = 'heads'
            if param in self.property_mask_res:
                xlabel, ylabel = 'attributes', 'attributes'

            for ax in axes:
                ax.set_xlabel(xlabel)

            if labels:
                for ax in axes:
                    ax.set_ylabel(ylabel)
            else:
                for ax in axes:
                    ax.set_yticks([])

            if out_dir is not None:
                if out_file_name_prefix is not None:
                    out_plot_file_name = '{}_{}.pdf'.format(out_file_name_prefix, param)
                else:
                    out_plot_file_name = '{}.pdf'.format(param)

                plt.savefig(os.path.join(out_dir, out_plot_file_name), bbox_inches='tight')

            plt.show()
