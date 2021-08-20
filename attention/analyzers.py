import copy

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from utils.result_collector import TestResultCollector
from attention.extractors import AttributeAttentionExtractor, WordAttentionExtractor, AttentionExtractor
import numpy as np
from utils.result_collector import BinaryClassificationResultsAggregator
from scipy.stats import entropy


class AttentionMapAnalyzer(object):
    """
    This class applies the analyzes implemented by appropriate tester classes to
    the attention maps extracted by an appropriate attention extractor class.
    It automatically categorizes the results of such analyzes into the following
    categories:
    - all: all the records of the dataset
    - true_match: ground truth match records
    - true_non_match: ground truth non-match records
    - pred_match: match records predicted by the model
    - pred_non_match: non-match records predicted by the model
    It accepts multiple classes as input for the application of different tests.
    Such classes have to implement the following interface:
    - test(left_entity, right_entity, attn_params): applies some tests on the
        attention maps (integrated with additional params) and returns the results
    """

    def __init__(self, attn_extractor, testers: list):

        assert isinstance(testers, list), "Wrong data type for parameter 'testers'."
        assert len(testers) > 0, "Empty tester list."

        self.attn_extractor = attn_extractor
        self.testers = testers
        res = {
            'all': None,
            'true_match': None,
            'true_non_match': None,
            'pred_match': None,
            'pred_non_match': None,
            'tp': None,
            'tn': None,
            'fp': None,
            'fn': None
        }

        self.res_history = [copy.deepcopy(res) for _ in range(len(self.testers))]
        self.counts = {
            'all': 0,
            'true_match': 0,
            'true_non_match': 0,
            'pred_match': 0,
            'pred_non_match': 0,
            'tp': 0,
            'tn': 0,
            'fp': 0,
            'fn': 0
        }
        self.preds = []
        self.labels = []
        self.text_units = []

    def __len__(self):
        return len(self.attn_extractor)

    def _save_result(self, res_history: dict, result: (dict, TestResultCollector),
                     result_name: str):

        assert isinstance(res_history, dict), "Wrong data type for parameter 'res_history'."
        assert isinstance(result, (dict, TestResultCollector)), "Wrong data type for parameter 'result'."
        assert isinstance(result_name, str), "Wrong data type for parameter 'result_name'."
        assert result_name in res_history

        if res_history[result_name] is None:  # init the result collector

            res_history[result_name] = copy.deepcopy(result)

        else:  # cumulative sum of the results

            if isinstance(result, dict):

                for key in result:
                    assert key in res_history[result_name]
                    assert isinstance(res_history[result_name][key], TestResultCollector)
                    assert isinstance(result[key], TestResultCollector)

                    # res_history[result_name][key].add_collector(result[key])
                    res_history[result_name].transform_collector(result, transform_fn=lambda x, y: x + y)

            elif isinstance(result, TestResultCollector):

                assert isinstance(res_history[result_name], TestResultCollector)

                # res_history[result_name].add_collector(result)
                res_history[result_name].transform_collector(result, transform_fn=lambda x, y: x + y)

        self.counts[result_name] += 1

    def _save(self, results: list, label: int, pred: int, category=None):

        assert isinstance(results, list), "Wrong data type for parameter 'results'."
        assert len(results) == len(self.testers), "Length mismatch between 'results' and 'self.testers'."
        assert isinstance(label, int), "Wrong data type for parameter 'label'."
        if pred is not None:
            assert isinstance(pred, int), "Wrong data type for parameter 'pred'."

        for tester_idx in range(len(self.testers)):
            tester_res_history = self.res_history[tester_idx]
            tester_res = results[tester_idx]

            self._save_result(tester_res_history, tester_res, 'all')

            if label == 1:  # match row

                self._save_result(tester_res_history, tester_res, 'true_match')

                if pred is not None:
                    if pred == 1:   # true positive
                        self._save_result(tester_res_history, tester_res, 'pred_match')
                        self._save_result(tester_res_history, tester_res, 'tp')
                    else:           # false negative
                        self._save_result(tester_res_history, tester_res, 'pred_non_match')
                        self._save_result(tester_res_history, tester_res, 'fn')

            else:  # non-match row

                self._save_result(tester_res_history, tester_res, 'true_non_match')

                if pred is not None:
                    if pred == 1:   # false positive
                        self._save_result(tester_res_history, tester_res, 'pred_match')
                        self._save_result(tester_res_history, tester_res, 'fp')
                    else:           # true negative
                        self._save_result(tester_res_history, tester_res, 'pred_non_match')
                        self._save_result(tester_res_history, tester_res, 'tn')

            if category is not None:
                if category not in tester_res_history:
                    tester_res_history[category] = copy.deepcopy(tester_res)
                else:
                    self._save_result(tester_res_history, tester_res, category)

    def __getitem__(self, idx: int):

        left_entity, right_entity, attn_params = self.attn_extractor[idx]

        assert isinstance(left_entity, pd.Series), "Wrong data type for parameter 'left_entity'."
        assert isinstance(right_entity, pd.Series), "Wrong data type for parameter 'right_entity'."
        assert isinstance(attn_params, dict), "Wrong data type for parameter 'attn_params'."
        assert 'preds' in attn_params, "predictions not found."
        assert 'labels' in attn_params, "labels not found."
        assert 'text_units' in attn_params, "'text_units' not found."

        label = attn_params['labels'].item()
        pred = None
        if attn_params['preds'] is not None:
            pred = attn_params['preds'].item()
        text_units = attn_params['text_units']
        category = None
        if 'category' in attn_params:
            category = attn_params['category']

        tester_results = []
        for tester_id, tester in enumerate(self.testers):
            result = tester.test(left_entity, right_entity, attn_params)

            if result is not None:
                assert isinstance(result, (dict, TestResultCollector))
                if isinstance(result, dict):
                    for key in result:
                        assert isinstance(result[key], TestResultCollector)

            tester_results.append(result)

        return tester_results, label, pred, category, text_units

    def analyze(self, idx: int):
        return self[idx]

    def get_labels_and_preds(self):
        return self.labels, self.preds

    def get_text_units(self):
        return self.text_units

    def analyze_all(self):

        # retrieve all the results
        for row_tester_results, row_label, row_pred, row_category, row_text_units in tqdm(self):
            assert len(row_tester_results) == len(self.testers)
            if row_tester_results[0] is not None:
                self._save(row_tester_results, row_label, row_pred, row_category)
                self.labels.append(row_label)
                self.preds.append(row_pred)
                self.text_units.append(row_text_units)

        assert len(self.res_history) == len(self.testers)

        # result post-processing
        # now the results are stored and categorized in the 'res_history' variable
        for tester_res in self.res_history:

            assert isinstance(tester_res, dict)

            for category in tester_res:
                tester_res_by_cat = tester_res[category]

                if tester_res_by_cat is not None:

                    assert isinstance(tester_res_by_cat, (dict, TestResultCollector))

                    if isinstance(tester_res_by_cat, dict):

                        for key in tester_res_by_cat:
                            assert isinstance(tester_res_by_cat[key], TestResultCollector)
                            assert len(tester_res_by_cat[key]) > 0

                            # normalization
                            tester_res_by_cat[key].transform_all(lambda x: x / self.counts[category])

                    elif isinstance(tester_res_by_cat, TestResultCollector):

                        assert len(tester_res_by_cat) > 0

                        # normalization
                        tester_res_by_cat.transform_all(lambda x: x / self.counts[category])

        return copy.deepcopy(self.res_history)


class AttrToClsAttentionAnalyzer(object):

    @staticmethod
    def group_or_aggregate(attn_results: list, target_categories: list = None, agg_metric: str = None):

        AttributeAttentionExtractor.check_batch_attn_features(attn_results)

        attrs = None
        records_cls_attn = []
        for attn_res in attn_results:
            attn_params = attn_res[2]

            if attn_params['attns'] is None:
                continue

            attn_text_units = attn_params['text_units']
            label = attn_params['labels'].item()
            pred = attn_params['preds'].item()
            assert attn_text_units[0] == '[CLS]' and attn_text_units[-1] == '[SEP]'
            if attrs is None:
                attrs = [f'{lr}{attr}' for lr in ['l_', 'r_'] for attr in attn_text_units[1:-1]]
            else:
                assert attrs == [f'{lr}{attr}' for lr in ['l_', 'r_'] for attr in attn_text_units[1:-1]]
            text_unit_idxs = [i for i, tu in enumerate(attn_text_units) if tu not in ['[CLS]', '[SEP]']]
            text_unit_idxs += [len(attn_text_units) - 1 + i for i in text_unit_idxs]

            # select only the last layer
            attns = attn_params['attns'][-1]
            # get an average attention map by aggregating along the heads belonging to the last layer
            attns = np.mean(attns, axis=0)
            # select only the row related to the CLS token (i.e., the first row of the attention map)
            attns = attns[0]
            # filter out the attention of other special tokens
            attns = attns[text_unit_idxs]

            record_cls_attn = {
                'label': label,
                'pred': pred,
                'attn': attns
            }
            records_cls_attn.append(record_cls_attn)

        aggregator = BinaryClassificationResultsAggregator('attn', target_categories=target_categories)
        grouped_cls_attn, _, _, _ = aggregator.add_batch_data(records_cls_attn)

        if agg_metric is not None:
            attr2cls_attn = aggregator.aggregate(agg_metric)
        else:
            attr2cls_attn = {}
            for cat in grouped_cls_attn:
                if grouped_cls_attn is not None:
                    attr2cls_attn[cat] = pd.DataFrame(grouped_cls_attn[cat], columns=attrs)

        return attr2cls_attn

    @staticmethod
    def analyze_multi_results(attr2cls_attn: dict, analysis_type: str):

        assert isinstance(attr2cls_attn, dict)
        assert isinstance(analysis_type, str)
        assert analysis_type in ['entropy']

        if analysis_type == 'entropy':

            def get_entropy(distribution):
                return entropy(distribution, base=2)

            out_data = []
            for use_case in attr2cls_attn:
                uc_attr2cls_attn = attr2cls_attn[use_case]
                AttrToClsAttentionAnalyzer.check_attr_to_cls_attn_results(uc_attr2cls_attn, agg=True)

                entropy_by_cat = {}
                for cat in uc_attr2cls_attn:
                    uc_cat_attn = uc_attr2cls_attn[cat]
                    entropy_by_cat[cat] = get_entropy(uc_cat_attn['mean'])

                out_data.append(entropy_by_cat)

            out_data = pd.DataFrame(out_data, index=attr2cls_attn.keys())

        else:
            raise NotImplementedError()

        return out_data

    @staticmethod
    def check_attr_to_cls_attn_results(attr2cls_attn: dict, agg: bool = False):
        assert isinstance(attr2cls_attn, dict), "Wrong results data type."
        err_msg = 'Wrong results format.'
        assert all([k in BinaryClassificationResultsAggregator.categories for k in attr2cls_attn.keys()]), err_msg
        for cat in attr2cls_attn:
            attr2cls_attn_by_cat = attr2cls_attn[cat]
            if attr2cls_attn_by_cat is not None:
                if not agg:
                    assert isinstance(attr2cls_attn_by_cat, pd.DataFrame), err_msg
                else:
                    assert isinstance(attr2cls_attn_by_cat, dict), err_msg
                    assert all([p in attr2cls_attn_by_cat for p in ['mean', 'std']]), err_msg
                    for metric in attr2cls_attn_by_cat:
                        assert isinstance(attr2cls_attn_by_cat[metric], np.ndarray), err_msg

    @staticmethod
    def plot_attr_to_cls_attn_entropy(entropy_res: pd.DataFrame, save_path: str = None):
        assert isinstance(entropy_res, pd.DataFrame)

        entropy_res = entropy_res.rename(columns={'all_pred_pos': 'match', 'all_pred_neg': 'non_match'})
        entropy_res.plot.bar(figsize=(12, 4))
        plt.ylabel('Entropy')
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

        plt.show()

    @staticmethod
    def plot_attr_to_cls_attn(attr2cls_attn, ax=None, title=None):
        AttrToClsAttentionAnalyzer.check_attr_to_cls_attn_results(attr2cls_attn)

        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 12))

        for cat in attr2cls_attn:
            attr2cls_attn_by_cat = attr2cls_attn[cat]
            attr2cls_table_stats = attr2cls_attn_by_cat.describe()
            medians = attr2cls_table_stats.loc['50%', :].values
            percs_25 = attr2cls_table_stats.loc['25%', :].values
            percs_75 = attr2cls_table_stats.loc['75%', :].values
            plot_data = {
                'x': range(len(attr2cls_table_stats.columns)),
                'y': medians,
                'yerr': [medians - percs_25, percs_75 - medians],
            }

            plot_cat = cat
            if cat == 'all_pred_pos':
                plot_cat = 'match'
            if cat == 'all_pred_neg':
                plot_cat = 'non-match'

            ax.errorbar(**plot_data, alpha=.75, fmt=':', capsize=3, capthick=1, label=plot_cat)
            # plot_data_area = {
            #     'x': plot_data['x'],
            #     'y1': percs_25,
            #     'y2': percs_75
            # }
            # ax.fill_between(**plot_data_area, alpha=.25)
            ax.set_xticks(range(len(attr2cls_attn_by_cat.columns)))
            ax.set_xticklabels(attr2cls_attn_by_cat.columns, rotation=45)
            ax.legend()

        if title is not None:
            ax.set_title(title)
        ax.set_xlabel('Attributes')

    @staticmethod
    def plot_multi_attr_to_cls_attn(attr2cls_attn: dict, save_path: str = None):

        assert isinstance(attr2cls_attn, dict)

        ncols = 4
        nrows = 3
        if len(attr2cls_attn) == 1:
            ncols = 1
            nrows = 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 12), sharey=True)
        if len(attr2cls_attn) > 1:
            axes = axes.flat
        # loop over the use cases
        for idx, use_case in enumerate(attr2cls_attn):
            if len(attr2cls_attn) == 1:
                ax = axes
            else:
                ax = axes[idx]
            AttrToClsAttentionAnalyzer.plot_attr_to_cls_attn(attr2cls_attn[use_case], ax=ax, title=use_case)
            if idx % ncols == 0:
                ax.set_ylabel('Attribute to [CLS] attention')

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.02, hspace=0.8)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

        plt.show()


class EntityToEntityAttentionAnalyzer(object):

    def __init__(self, attn_data: list, text_unit: str, tokenization: str):

        assert isinstance(text_unit, str)
        assert text_unit in ['attr', 'word', 'token']
        assert isinstance(tokenization, str)
        assert tokenization in ['sent_pair', 'attr_pair']

        if text_unit == 'attr':
            AttributeAttentionExtractor.check_batch_attn_features(attn_data)
        elif text_unit == 'word':
            WordAttentionExtractor.check_batch_attn_features(attn_data)
        else:
            AttentionExtractor.check_batch_attn_features(attn_data)

        self.attn_data = []
        for attn_item in attn_data:
            attn_features = attn_item[2]

            attn_values = attn_features['attns']
            if text_unit == 'token':
                attn_text_units = attn_features['tokens']
                attn_values = np.concatenate(attn_values)
                if '[PAD]' in attn_text_units:
                    pad_idx = attn_text_units.index('[PAD]')
                    attn_text_units = attn_text_units[:pad_idx]
                    attn_values = attn_values[:, :, :pad_idx, :pad_idx]
            else:
                attn_text_units = attn_features['text_units']
                if text_unit == 'attr':
                    attn_text_units = attn_text_units + attn_text_units[1:]

            attn_row = {
                'attns': attn_values,
                'text_units': attn_text_units,
                'label': attn_features['labels'].item(),
                'pred': attn_features['preds'].item(),
            }

            self.attn_data.append(attn_row)

        self.text_unit = text_unit
        self.tokenization = tokenization

    def analyze(self, analysis_type: str, ignore_special: bool = True, target_categories: list = None):

        assert isinstance(analysis_type, str)
        assert analysis_type in ['same_entity', 'cross_entity']
        assert isinstance(ignore_special, bool)

        entity_to_entity_attn = []
        for attn_item in self.attn_data:

            # get an average attention map for each layer by averaging all the heads that refer to the same layer
            attns = np.mean(attn_item['attns'], axis=1)

            # find the [SEP] token used to delimit the two entities
            sep_idxs = np.where(np.array(attn_item['text_units']) == '[SEP]')[0]
            if self.tokenization == 'sent_pair':
                entity_delimit = attn_item['text_units'].index('[SEP]')     # get first occurrence of the [SEP] token
            else:
                # in the attr-pair tokenization the [SEP] token is also used to delimit the attributes
                entity_delimit = sep_idxs[((len(sep_idxs) - 1) // 2) + 1]

            # select the top attention scores for each layer-wise attention map
            top_attns = np.zeros((attns.shape[0], attns.shape[1], attns.shape[2]))
            for layer in range(attns.shape[0]):
                layer_attn_map = attns[layer]
                thr = np.quantile(layer_attn_map, 0.8)
                top_layer_attn_map = layer_attn_map >= thr
                top_attns[layer] = top_layer_attn_map

            # count the number of attention scores that passed the previous test in a 'same_entity' or 'cross_entity'
            # perspective

            left_target_idxs = list(range(entity_delimit + 1))
            right_target_idxs = list(range(entity_delimit, top_attns.shape[1]))
            if ignore_special:
                left_target_idxs = left_target_idxs[1:]     # remove [CLS]
                left_target_idxs = sorted(list(set(left_target_idxs).difference(set(sep_idxs))))    # remove [SEP]s
                right_target_idxs = sorted(list(set(right_target_idxs).difference(set(sep_idxs))))  # remove [SEP]s

            e2e_attn = np.zeros(top_attns.shape[0])
            for layer in range(top_attns.shape[0]):

                if analysis_type == 'same_entity':
                    left_hits = top_attns[layer, left_target_idxs, :][:, left_target_idxs].sum()
                    right_hits = top_attns[layer, right_target_idxs, :][:, right_target_idxs].sum()

                else:   # cross_entity
                    left_hits = top_attns[layer, left_target_idxs, :][:, right_target_idxs].sum()
                    right_hits = top_attns[layer, right_target_idxs, :][:, left_target_idxs].sum()

                total_normalized_hits = (left_hits + right_hits) / (len(left_target_idxs) * len(right_target_idxs) * 2)
                e2e_attn[layer] = total_normalized_hits

            entity_to_entity_attn.append({
                'score': e2e_attn,
                'label': attn_item['label'],
                'pred': attn_item['pred'],
            })

        aggregator = BinaryClassificationResultsAggregator('score', target_categories=target_categories)
        grouped_e2e_attn, _, _, _ = aggregator.add_batch_data(entity_to_entity_attn)

        e2e_attn_results = {}
        for cat in grouped_e2e_attn:
            if grouped_e2e_attn[cat] is not None:
                e2e_attn_results[cat] = pd.DataFrame(grouped_e2e_attn[cat],
                                                     columns=range(1, self.attn_data[0]['attns'].shape[0] + 1))

        return e2e_attn_results

    @staticmethod
    def check_entity_to_entity_attn_results(e2e_results: dict):
        assert isinstance(e2e_results, dict), "Wrong results data type."
        err_msg = 'Wrong results format.'
        # assert all([k in BinaryClassificationResultsAggregator.categories for k in e2e_results.keys()]), err_msg
        for cat in e2e_results:
            e2e_attn_by_cat = e2e_results[cat]
            if e2e_attn_by_cat is not None:
                assert isinstance(e2e_attn_by_cat, pd.DataFrame), err_msg

    @staticmethod
    def plot_entity_to_entity_attn(e2e_attn, ax=None, title=None):
        EntityToEntityAttentionAnalyzer.check_entity_to_entity_attn_results(e2e_attn)

        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 12))

        for cat in e2e_attn:
            e2e_attn_by_cat = e2e_attn[cat]
            e2e_table_stats = e2e_attn_by_cat.describe()
            medians = e2e_table_stats.loc['50%', :].values
            percs_25 = e2e_table_stats.loc['25%', :].values
            percs_75 = e2e_table_stats.loc['75%', :].values
            plot_data = {
                'x': range(len(e2e_table_stats.columns)),
                'y': medians,
                'yerr': [medians - percs_25, percs_75 - medians],
            }

            plot_cat = cat
            if cat == 'all_pred_pos':
                plot_cat = 'match'
            if cat == 'all_pred_neg':
                plot_cat = 'non-match'

            ax.errorbar(**plot_data, alpha=.75, fmt=':', capsize=3, capthick=1, label=plot_cat)
            # plot_data_area = {
            #     'x': plot_data['x'],
            #     'y1': percs_25,
            #     'y2': percs_75
            # }
            # ax.fill_between(**plot_data_area, alpha=.25)
            ax.set_xticks(range(len(e2e_attn_by_cat.columns)))
            ax.set_xticklabels(e2e_attn_by_cat.columns)
            ax.legend()

        if title is not None:
            ax.set_title(title)
        ax.set_xlabel('Layers')

    @staticmethod
    def plot_multi_entity_to_entity_attn(e2e_results, save_path: str = None):

        assert isinstance(e2e_results, dict)

        ncols = 4
        nrows = 3
        if len(e2e_results) == 1:
            ncols = 1
            nrows = 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 12), sharey=True)
        if len(e2e_results) > 1:
            axes = axes.flat
        # loop over the use cases
        for idx, use_case in enumerate(e2e_results):
            if len(e2e_results) == 1:
                ax = axes
            else:
                ax = axes[idx]
            EntityToEntityAttentionAnalyzer.plot_entity_to_entity_attn(e2e_results[use_case], ax=ax, title=use_case)
            if idx % ncols == 0:
                ax.set_ylabel('Entity to entity attention')

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.02, hspace=0.3)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

        plt.show()
