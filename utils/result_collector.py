import numpy as np
import copy


class TestResultCollector(object):

    def __init__(self):
        self.results = {}

    def __len__(self):
        return len(self.results)

    def get_results(self):
        return self.results

    def get_result(self, result_id: str):

        assert isinstance(result_id, str), "Wrong data type for parameter 'result_id'."

        if result_id in self.results:
            return self.results[result_id]
        return None

    def save_result(self, result, result_id: str):

        assert isinstance(result_id, str), "Wrong data type for parameter 'result_id'."

        self.results[result_id] = result

    def update_result_value(self, x: int, y: int, val, result_id: str):

        assert isinstance(x, int), "Wrong data type for parameter 'x'."
        assert isinstance(y, int), "Wrong data type for parameter 'y'."
        assert isinstance(result_id, str), "Wrong data type for parameter 'result_id'."
        assert result_id in self.results, f"No result id {result_id} found."
        assert isinstance(self.results[result_id], np.ndarray)
        assert x < self.results[result_id].shape[0], "Row index out of bounds."
        assert y < self.results[result_id].shape[1], "Column index out of bounds."

        self.results[result_id][x, y] = val

    def transform_result(self, result_id: str, transform_fn):

        assert isinstance(result_id, str), "Wrong data type for parameter 'result_id'."
        assert result_id in self.results

        self.results[result_id] = transform_fn(self.results[result_id])

    def transform_all(self, transform_fn):

        assert len(self) > 0, "No results found."

        for result_id in self.results:
            self.transform_result(result_id, transform_fn)

    def combine_results(self, res_id1: str, res_id2: str, comb_fn,
                        out_res_id: str):

        assert isinstance(res_id1, str), "Wrong data type for parameter 'res_id1'."
        assert isinstance(res_id2, str), "Wrong data type for parameter 'res_id2'."
        assert res_id1 in self.results, f"No result id {res_id1} found."
        assert res_id2 in self.results, f"No result id {res_id2} found."
        assert isinstance(out_res_id, str), "Wrong data type for parameter 'out_res_id'."

        res1 = self.results[res_id1]
        res2 = self.results[res_id2]
        self.results[out_res_id] = comb_fn(res1, res2)

    def __copy__(self):
        trc = TestResultCollector()
        trc.results = copy.deepcopy(self.results)

        return trc

    def __deepcopy__(self, memo):

        return self.__copy__()

    def transform_collector(self, res_collector, transform_fn):

        assert isinstance(res_collector, type(self)), "Wrong data type for parameter 'res_collector'."
        assert len(self) == len(res_collector), "Result collectors not compatible: different len."
        input_results = res_collector.get_results()
        assert set(self.results) == set(input_results), "Result collectors not compatible: different result ids."

        for result_name in list(self.results):

            if isinstance(self.results[result_name], np.ndarray) and isinstance(input_results[result_name], np.ndarray):
                left_shape = self.results[result_name].shape
                right_shape = input_results[result_name].shape
                if left_shape == right_shape:
                    self.results[result_name] = transform_fn(self.results[result_name], input_results[result_name])
                else:
                    max_row_dim = max(left_shape[0], right_shape[0])
                    max_col_dim = max(left_shape[1], right_shape[1])

                    extended_left = np.zeros((max_row_dim, max_col_dim))
                    extended_left[:left_shape[0], :left_shape[1]] = self.results[result_name][:]

                    extended_right = np.zeros((max_row_dim, max_col_dim))
                    extended_right[:right_shape[0], :right_shape[1]] = input_results[result_name][:]

                    self.results[result_name] = transform_fn(extended_left, extended_right)


class BinaryClassificationResultsAggregator(object):

    def __init__(self, data_key: str, label_col: str = 'label', pred_col: str = 'pred',
                 target_categories: list = ['all']):

        assert isinstance(data_key, str), "Wrong data type for parameter 'data_key'."
        assert isinstance(label_col, str), "Wrong data type for parameter 'label_col'."
        assert isinstance(pred_col, str), "Wrong data type for parameter 'pred_col'."
        assert isinstance(target_categories, list), "Wrong data type for parameter 'target_categories'."
        assert len(target_categories) > 0, "Empty target categories."

        self.data_key = data_key
        self.label_col = label_col
        self.pred_col = pred_col
        self.res_collector = TestResultCollector()
        self.categories = ['all', 'all_pos', 'all_neg', 'all_pred_pos', 'all_pred_neg', 'tp', 'tn', 'fp', 'fn']
        assert all(
            [c in self.categories for c in target_categories]), "Wrong data format for parameter 'target_categories'."
        self.target_categories = target_categories
        self.agg_metrics = ['mean']

    def _check_data_format(self, data: dict):
        if data is not None:
            assert isinstance(data, dict), "Wrong data type for parameter 'data'."
            params = [self.data_key, self.label_col, self.pred_col]
            assert all([p in data for p in params]), "Wrong data format for parameter 'data'."

        return True

    def _add_data_by_category(self, data: list, cat: str):
        assert isinstance(data, list), "Wrong data type for parameter 'data'."
        assert len(data) > 0, "Empty data."
        assert isinstance(cat, str), "Wrong data type for parameter 'cat'."
        assert cat in self.categories, "Wrong data format for parameter 'cat'."

        if self.res_collector.get_result(cat) is None:
            self.res_collector.save_result(data, cat)
        else:
            self.res_collector.transform_result(cat, transform_fn=lambda x: x + data)

    def add_batch_data(self, batch: list):
        assert isinstance(batch, list), "Wrong data type for parameter 'batch'."
        assert len(batch) > 0, "Empty data."
        assert all([self._check_data_format(data) for data in batch]), "Wrong data format for parameter 'batch'."

        def _add_to_group(group, x, key):
            if key not in group:
                group[key] = [x]
            else:
                group[key].append(x)
            return group

        grouped_data = {}
        for data in batch:
            if data is None:
                continue

            label = data[self.label_col]
            pred = data[self.pred_col]
            values = data[self.data_key]

            grouped_data = _add_to_group(grouped_data, values, 'all')

            if label == 1:
                grouped_data = _add_to_group(grouped_data, values, 'all_pos')

                if pred == 1:
                    grouped_data = _add_to_group(grouped_data, values, 'tp')
                    grouped_data = _add_to_group(grouped_data, values, 'all_pred_pos')
                else:
                    grouped_data = _add_to_group(grouped_data, values, 'fn')
                    grouped_data = _add_to_group(grouped_data, values, 'all_pred_neg')

            else:
                grouped_data = _add_to_group(grouped_data, values, 'all_neg')

                if pred == 1:
                    grouped_data = _add_to_group(grouped_data, values, 'fp')
                    grouped_data = _add_to_group(grouped_data, values, 'all_pred_pos')
                else:
                    grouped_data = _add_to_group(grouped_data, values, 'tn')
                    grouped_data = _add_to_group(grouped_data, values, 'all_pred_neg')

        for cat in grouped_data:
            if cat in self.target_categories:
                self._add_data_by_category(grouped_data[cat], cat)

    def aggregate(self, metric: str):
        assert isinstance(metric, str), "Wrong data type for parameter 'metric'."
        assert metric in self.agg_metrics, f"Wrong metric: {metric} not in {self.agg_metrics}."

        out_data = {}
        if metric == 'mean':
            mean_res_collector = copy.deepcopy(self.res_collector)
            mean_res_collector.transform_all(lambda x: np.array(x).mean(axis=0))
            mean_data = mean_res_collector.get_results()

            std_res_collector = copy.deepcopy(self.res_collector)
            std_res_collector.transform_all(lambda x: np.array(x).std(axis=0))
            std_data = std_res_collector.get_results()

            for key in mean_data:
                out_data[key] = {'mean': mean_data[key], 'std': std_data[key]}

        for cat in self.target_categories:
            if cat not in out_data:
                out_data[cat] = None

        return out_data
