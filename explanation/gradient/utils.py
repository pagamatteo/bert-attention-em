import pandas as pd
from explanation.gradient.extractors import EntityGradientExtractor
import numpy as np


# def get_words_sorted_by_gradient(left_entity: pd.Series, right_entity: pd.Series, features,
#                                  entity_grad_extr: EntityGradientExtractor, num_words: int = None):
#     assert isinstance(left_entity, pd.Series), "Wrong data type for parameter 'left_entity'."
#     assert isinstance(right_entity, pd.Series), "Wrong data type for parameter 'right_entity'."
#     params = ["input_ids", "attention_mask", "token_type_ids"]
#     assert all([p in features for p in params]), "Wrong data format for parameter 'features'."
#     assert isinstance(entity_grad_extr, EntityGradientExtractor), "Wrong data type for parameter 'entity_grad_extr'."
#     if num_words is not None:
#         assert isinstance(num_words, int), "Wrong data type for parameter 'num_words'."
#
#     data = [[left_entity, right_entity, features]]
#     grad_params = entity_grad_extr.extract(data)
#     assert isinstance(grad_params, list)
#     assert len(grad_params) == 1
#     row_grad_params = grad_params[0]
#     assert isinstance(row_grad_params, dict)
#     assert 'grad' in row_grad_params
#     row_grad_data = row_grad_params['grad']
#     assert isinstance(row_grad_data, dict)
#     assert 'all' in row_grad_data
#     assert 'all_grad' in row_grad_data
#     row_grad_words = row_grad_data['all']
#     row_grads = row_grad_data['all_grad']
#     assert isinstance(row_grad_words, list)
#     assert isinstance(row_grads, list)
#     sorted_words = sorted(zip(row_grad_words, row_grads), key=lambda x: x[1], reverse=True)
#
#     out_data = []
#     for word, grad in sorted_words:
#         if word.startswith('l_'):
#             w = word.replace('l_', '')
#             out_data.append({'left': w, 'right': None})
#
#         elif word.startswith('r_'):
#             w = word.replace('r_', '')
#             out_data.append({'left': None, 'right': w})
#
#         else:
#             ValueError("Gradient word without prefix.")
#
#         if num_words is not None and len(out_data) == num_words:
#             break
#
#     return out_data

def get_words_sorted_by_gradient(left_entity: pd.Series, right_entity: pd.Series, record_idx: int, grads: list,
                                 grad_agg_metric: str = 'max', num_words: int = None):
    assert isinstance(left_entity, pd.Series), "Wrong data type for parameter 'left_entity'."
    assert isinstance(right_entity, pd.Series), "Wrong data type for parameter 'right_entity'."
    assert isinstance(record_idx, int), "Wrong data type for parameter 'record_idx'."
    EntityGradientExtractor.check_extracted_grad(grads)
    assert record_idx < len(grads), "Record idx out of range."
    assert isinstance(grad_agg_metric, str), "Wrong data type for parameter 'grad_agg_metric'."
    if num_words is not None:
        assert isinstance(num_words, int), "Wrong data type for parameter 'num_words'."

    # check that the input record matches with the record_idx-th record in the gradient list
    grad_record = grads[record_idx]['grad']
    grad_record_tokens = grad_record['all']
    grad_record_values = grad_record['all_grad'][grad_agg_metric]
    if grad_record_tokens[0] == '[CLS]':
        sep_idxs = list(np.where(np.array(grad_record_tokens) == '[SEP]')[0])
        skip_idxs = [0] + sep_idxs
        grad_record_tokens = [grad_record_tokens[i] for i in range(len(grad_record_tokens)) if i not in skip_idxs]
        grad_record_values = [grad_record_values[i] for i in range(len(grad_record_values)) if i not in skip_idxs]
    left_sent = ''
    for attr, val in left_entity.iteritems():
        left_sent += f'{str(val)} '
    left_sent = left_sent[:-1]  # remove last space
    right_sent = ''
    for attr, val in right_entity.iteritems():
        right_sent += f'{str(val)} '
    right_sent = right_sent[:-1]  # remove last space
    assert f'{left_sent} {right_sent}' == ' '.join([t.replace('l_', '').replace('r_', '') for t in grad_record_tokens])

    sorted_words = sorted(zip(grad_record_tokens, grad_record_values), key=lambda x: x[1], reverse=True)

    out_data = []
    for word, grad in sorted_words:
        if word.startswith('l_'):
            w = word.replace('l_', '')
            out_data.append({'left': w, 'right': None})

        elif word.startswith('r_'):
            w = word.replace('r_', '')
            out_data.append({'left': None, 'right': w})

        else:
            ValueError("Gradient word without prefix.")

        if num_words is not None and len(out_data) == num_words:
            break

    return out_data


def get_pair_words_sorted_by_gradient(left_entity: pd.Series, right_entity: pd.Series, record_idx: int, grads: list,
                                      grad_agg_metric: str = 'max', num_words: int = None):
    assert isinstance(left_entity, pd.Series), "Wrong data type for parameter 'left_entity'."
    assert isinstance(right_entity, pd.Series), "Wrong data type for parameter 'right_entity'."
    assert isinstance(record_idx, int), "Wrong data type for parameter 'record_idx'."
    EntityGradientExtractor.check_extracted_grad(grads)
    assert record_idx < len(grads), "Record idx out of range."
    assert isinstance(grad_agg_metric, str), "Wrong data type for parameter 'grad_agg_metric'."
    if num_words is not None:
        assert isinstance(num_words, int), "Wrong data type for parameter 'num_words'."

    # check that the input record matches with the record_idx-th record in the gradient list
    grad_record = grads[record_idx]['grad']
    left_grad_tokens = grad_record['left']
    left_grad_values = grad_record['left_grad'][grad_agg_metric]
    right_grad_tokens = grad_record['right']
    right_grad_values = grad_record['right_grad'][grad_agg_metric]
    left_first_attr = str(left_entity.iloc[0]).split()
    left_grad_first_attr = [left_grad_tokens[i][2:] for i in range(min(len(left_first_attr), len(left_grad_tokens)))]
    left_first_attr = left_first_attr[:len(left_grad_first_attr)]
    right_first_attr = str(right_entity.iloc[0]).split()
    right_grad_first_attr = [right_grad_tokens[i][2:] for i in range(min(len(right_first_attr), len(right_grad_tokens)))]
    right_first_attr = right_first_attr[:len(right_grad_first_attr)]
    assert left_first_attr == left_grad_first_attr and right_first_attr == right_grad_first_attr

    sorted_left_words = sorted(zip(left_grad_tokens, left_grad_values), key=lambda x: x[1], reverse=True)
    sorted_right_words = sorted(zip(right_grad_tokens, right_grad_values), key=lambda x: x[1], reverse=True)

    out_data = []
    for left_grad_data, right_grad_data in zip(sorted_left_words, sorted_right_words):
        left_word = left_grad_data[0][2:]
        right_word = right_grad_data[0][2:]
        out_data.append({'left': left_word, 'right': right_word})

        if num_words is not None and len(out_data) == num_words:
            break

    return out_data
