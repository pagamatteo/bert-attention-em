from models.em_dataset import EMDataset
import pandas as pd


class Sampler(object):
    """
    This class implements some techniques for sampling rows from an EM dataset.
    """

    def __init__(self, dataset: EMDataset, permute: bool = False):

        assert isinstance(dataset, EMDataset), "Wrong data type for parameter 'dataset'."
        assert isinstance(permute, bool), "Wrong data type for parameter 'permute'."

        self.dataset = dataset
        self.data = self.dataset.get_complete_data()
        self.dataset_params = self.dataset.get_params()
        self.permute = permute

    def _get_data_by_label(self, label_val: int, size: int = None, seed: int = 42):

        assert isinstance(label_val, int), "Wrong data type for parameter 'label_val'."
        if size is not None:
            assert isinstance(size, int), "Wrong data type for parameter 'size'."
        assert isinstance(seed, int), "Wrong data type for parameter 'seed'."

        label_col = self.dataset_params['label_col']

        assert label_val in list(self.data[label_col].unique()), "Wrong value for parameter 'label_val'."

        out_data = self.data[self.data[label_col] == label_val]

        if size is not None:
            assert size <= len(out_data)

            out_data = out_data.sample(size, random_state=seed)

        return out_data

    def _create_dataset(self, data: pd.DataFrame, params: dict):

        assert isinstance(data, pd.DataFrame), "Wrong data type for parameter 'data'."
        assert isinstance(params, dict), "Wrong data type for parameter 'params'."
        param_names = ["model_name", "label_col", "left_prefix", "right_prefix", "max_len", "verbose", "tokenization"]
        assert all([p in params for p in param_names]), "Missing some parameters from 'params'."

        model_name = params["model_name"]
        label_col = params["label_col"]
        left_prefix = params["left_prefix"]
        right_prefix = params["right_prefix"]
        max_len = params["max_len"]
        verbose = params["verbose"]
        tokenization = params["tokenization"]

        return EMDataset(data, model_name, tokenization=tokenization, label_col=label_col, left_prefix=left_prefix,
                         right_prefix=right_prefix, max_len=max_len, verbose=verbose, permute=self.permute)

    def get_match_data(self, size: int = None, seed: int = 42):

        match_data = self._get_data_by_label(1, size, seed)

        dataset_params = self.dataset_params.copy()
        dataset_params["verbose"] = True

        return self._create_dataset(match_data, dataset_params)

    def get_non_match_data(self, size: int = None, seed: int = 42):

        non_match_data = self._get_data_by_label(0, size, seed)

        dataset_params = self.dataset_params.copy()
        dataset_params["verbose"] = True

        return self._create_dataset(non_match_data, dataset_params)

    def get_balanced_data(self, size: int = None, seeds: list = [42, 42]):

        assert isinstance(seeds, list), "Wrong data type for parameter 'seeds'."
        assert len(seeds) == 2, "Only two classes are supported."

        match_seed = seeds[0]
        non_match_seed = seeds[1]

        match_data = self._get_data_by_label(1, size, match_seed)
        if size is not None:
            non_match_data = self._get_data_by_label(0, size, non_match_seed)
        else:
            non_match_data = self._get_data_by_label(0, len(match_data), non_match_seed)

        out_data = pd.concat([match_data, non_match_data])

        dataset_params = self.dataset_params.copy()
        dataset_params["verbose"] = True
        # dataset_params["categories"] = ([1] * len(match_data)) + ([0] * len(match_data))

        return self._create_dataset(out_data, dataset_params)


if __name__ == '__main__':
    from utils.data_collector import DataCollector
    import os

    uc = "Structured_Fodors-Zagats"

    data_type = 'train'
    # data_type = 'test'
    # data_type = 'valid'

    # dataset params
    model_name = 'bert-base-uncased'
    tok = 'sent_pair'
    # tok = 'attr'
    # tok = 'attr_pair'
    label_col = 'label'
    left_prefix = 'left_'
    right_prefix = 'right_'
    max_len = 128

    # sampler params
    perm = False
    # size = None
    size = 2
    target_class = 'both'
    # target_class = 0
    # target_class = 1

    # download the data
    data_collector = DataCollector()
    use_case_dir = data_collector.get_data(uc)

    # data selection
    if data_type == 'train':
        dataset_path = os.path.join(use_case_dir, "train.csv")
    elif data_type == 'test':
        dataset_path = os.path.join(use_case_dir, "test.csv")
    else:
        dataset_path = os.path.join(use_case_dir, "valid.csv")

    data = pd.read_csv(dataset_path)
    dataset = EMDataset(data, model_name, tokenization=tok, label_col=label_col, left_prefix=left_prefix,
                        right_prefix=right_prefix, max_len=max_len)

    sampler = Sampler(dataset, permute=perm)

    sample = None
    if target_class == 'both':
        sample = sampler.get_balanced_data(size=size)
    elif target_class == 0:
        sample = sampler.get_non_match_data(size=size)
    elif target_class == 1:
        sample = sampler.get_match_data(size=size)

    if sample is not None:
        print("Num. samples: {}".format(len(sample)))

        first_row = sample[0]
        first_left_entity = first_row[0]
        first_right_entity = first_row[1]
        first_input_ids = first_row[2]['input_ids']
        first_row_text = dataset.tokenizer.convert_ids_to_tokens(first_input_ids)
        first_label = first_row[2]['labels']
        print("\nFIRST ROW")
        print(first_left_entity)
        print(first_right_entity)
        print(first_row_text)
        print(first_label)
        print("Num. sentences: {}".format(len(first_row[2]['token_type_ids'].unique())))

        last_row = sample[-1]
        last_left_entity = last_row[0]
        last_right_entity = last_row[1]
        last_input_ids = last_row[2]['input_ids']
        last_row_text = dataset.tokenizer.convert_ids_to_tokens(last_input_ids)
        last_label = last_row[2]['labels']
        print("\nLAST ROW")
        print(last_left_entity)
        print(last_right_entity)
        print(last_row_text)
        print(last_label)
        print("Num. sentences: {}".format(len(last_row[2]['token_type_ids'].unique())))

