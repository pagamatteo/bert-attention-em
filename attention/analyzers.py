import copy
import pandas as pd
from tqdm import tqdm
from utils.result_collector import TestResultCollector


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

                    res_history[result_name][key].add_collector(result[key])

            elif isinstance(result, TestResultCollector):

                assert isinstance(res_history[result_name], TestResultCollector)

                res_history[result_name].add_collector(result)

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
