from utils.general import get_dataset, get_model, get_sample
from attention.extractors import AttributeAttentionExtractor
import os


PROJECT_DIR = os.path.abspath('..')
MODELS_DIR = os.path.join(PROJECT_DIR, 'results', 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results', 'analysis')


def test_dataset(conf: dict):

    dataset = get_dataset(conf)
    verbose = conf['verbose']

    row = dataset[0]
    left_entity = None
    right_entity = None
    features = None

    if verbose:
        left_entity = row[0]
        right_entity = row[1]
        features = row[2]
    else:
        features = row

    assert features is not None

    if left_entity is not None:
        print(left_entity)

    if right_entity is not None:
        print(right_entity)

    row_text = dataset.tokenizer.convert_ids_to_tokens(features['input_ids'])
    row_label = features['labels']
    print(row_text)
    print(row_label)
    print("Num. sentences: {}".format(len(features['token_type_ids'].unique())))


def test_sampler(conf: dict, sampler_conf: dict):

    dataset = get_dataset(conf)

    size = sampler_conf['size']
    target_class = sampler_conf['target_class']
    seeds = sampler_conf['seeds']
    sample = get_sample(dataset, size, target_class, conf['permute'], seeds)

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


def test_extractor(conf: dict, sampler_conf: dict, fine_tune: str):

    dataset = get_dataset(conf)
    use_case = conf['use_case']
    tok = conf['tok']
    model_name = conf['model_name']

    if fine_tune is not None:
        model_path = os.path.join(MODELS_DIR, fine_tune, f"{use_case}_{tok}_tuned")
    else:
        model_path = None
    model = get_model(model_name, fine_tune, model_path)

    size = sampler_conf['size']
    target_class = sampler_conf['target_class']
    seeds = sampler_conf['seeds']
    sample = get_sample(dataset, size, target_class, conf['permute'], seeds)

    attr_attn_extractor = AttributeAttentionExtractor(sample, model)
    results = attr_attn_extractor.extract_all()

    for res in results:
        left = res[0]
        right = res[1]
        features = res[2]
        print(left)
        print(right)
        print(features.keys())
        print("-" * 10)


if __name__ == '__main__':

    conf = {
        'use_case': "Structured_Fodors-Zagats",
        'data_type': 'train',                       # 'train', 'test', 'valid'
        'model_name': 'bert-base-uncased',
        'tok': 'sent_pair',                         # 'sent_pair', 'attr', 'attr_pair'
        'label_col': 'label',
        'left_prefix': 'left_',
        'right_prefix': 'right_',
        'max_len': 128,
        'permute': False,
        'verbose': False,
    }

    sampler_conf = {
        'size': 2,
        'target_class': 'both',                     # 'both', 0, 1
        'seeds': [42, 42],                          # [42 -> class 0, 42 -> class 1]
    }

    # test_dataset(conf)

    # test_sampler(conf, sampler_conf)

    fine_tune = 'advanced'                            # None, 'simple', 'advanced'
    test_extractor(conf, sampler_conf, fine_tune)
