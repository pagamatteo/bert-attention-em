import os
import pandas as pd
from transformers import AutoModel, AutoModelForSequenceClassification

from utils.data_collector import DataCollector
from models.em_dataset import EMDataset
from utils.data_selection import Sampler
from fine_tuning.advanced_fine_tuning import MatcherTransformer


def get_use_case(use_case: str):

    data_collector = DataCollector()
    use_case_dir = data_collector.get_data(use_case)

    return use_case_dir


def get_dataset(conf: dict):

    assert isinstance(conf, dict), "Wrong data type for parameter 'conf'."
    params = ['use_case', 'data_type', 'model_name', 'tok', 'label_col', 'left_prefix', 'right_prefix', 'max_len',
              'verbose', 'permute']
    assert all([p in conf for p in params])
    assert isinstance(conf['data_type'], str), "Wrong data type for parameter 'data_type'."
    assert conf['data_type'] in ['train', 'test', 'valid'], "Wrong value for parameter 'data_type'."

    use_case = conf['use_case']
    data_type = conf['data_type']
    model_name = conf['model_name']
    tok = conf['tok']
    label_col = conf['label_col']
    left_prefix = conf['left_prefix']
    right_prefix = conf['right_prefix']
    max_len = conf['max_len']
    verbose = conf['verbose']
    permute = conf['permute']

    use_case_data_dir = get_use_case(use_case)

    if data_type == 'train':
        dataset_path = os.path.join(use_case_data_dir, "train.csv")
    elif data_type == 'test':
        dataset_path = os.path.join(use_case_data_dir, "test.csv")
    else:
        dataset_path = os.path.join(use_case_data_dir, "valid.csv")

    data = pd.read_csv(dataset_path)
    dataset = EMDataset(data, model_name, tokenization=tok, label_col=label_col, left_prefix=left_prefix,
                        right_prefix=right_prefix, max_len=max_len, verbose=verbose, permute=permute)

    return dataset


def get_sample(dataset: EMDataset, size: int = None, target_class='both', permute: bool = False,
               seeds: list = [42, 42]):

    assert isinstance(target_class, (str, int)), "Wrong data type for parameter 'target_class'."
    assert target_class in ['both', 0, 1], "Wrong value for parameter 'target_class'."
    assert isinstance(seeds, list), "Wrong data type for parameter 'seeds'."
    assert len(seeds) == 2, "Wrong value for parameter 'seeds'."

    sampler = Sampler(dataset, permute=permute)

    if target_class == 'both':
        sample = sampler.get_balanced_data(size=size, seeds=seeds)
    elif target_class == 0:
        sample = sampler.get_non_match_data(size=size, seed=seeds[0])
    elif target_class == 1:
        sample = sampler.get_match_data(size=size, seed=seeds[1])

    return sample


def get_model(model_name: str, fine_tune: str = None, model_path: str = None):

    assert isinstance(model_name, str), "Wrong data type for parameter 'model_name'."
    if fine_tune is not None:
        assert isinstance(fine_tune, str), "Wrong data type for parameter 'fine_tune'."
        assert fine_tune in ['simple', 'advanced'], "Wrong value for parameter 'fine_tune'."
        assert model_path is not None, "If 'fine_tune' is not null, provide a model path."
    if model_path is not None:
        assert isinstance(model_path, str), "Wrong data type for parameter 'model_path'."
        assert os.path.exists(model_path), "Wrong value for parameter 'model_path'."

    if not fine_tune:
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
    else:
        if fine_tune == 'simple':
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        elif fine_tune == 'advanced':
            model = MatcherTransformer.load_from_checkpoint(checkpoint_path=model_path)

    return model