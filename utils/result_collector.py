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

    def add_collector(self, res_collector):

        assert isinstance(res_collector, type(self)), "Wrong data type for parameter 'res_collector'."
        assert len(self) == len(res_collector), "Result collectors not compatible: different len."
        input_results = res_collector.get_results()
        assert set(self.results) == set(input_results), "Result collectors not compatible: different result ids."

        for result_name in list(self.results):

            if isinstance(self.results[result_name], np.ndarray) and isinstance(input_results[result_name], np.ndarray):
                left_shape = self.results[result_name].shape
                right_shape = input_results[result_name].shape
                if left_shape == right_shape:
                    self.results[result_name] += input_results[result_name]
                else:
                    max_row_dim = max(left_shape[0], right_shape[0])
                    max_col_dim = max(left_shape[1], right_shape[1])

                    extended_left = np.zeros((max_row_dim, max_col_dim))
                    extended_left[:left_shape[0], :left_shape[1]] = self.results[result_name][:]

                    extended_right = np.zeros((max_row_dim, max_col_dim))
                    extended_right[:right_shape[0], :right_shape[1]] = input_results[result_name][:]

                    self.results[result_name] = extended_left + extended_right
