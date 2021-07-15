import pandas as pd
from explanation.gradient.extractors import EntityGradientExtractor


def get_words_sorted_by_gradient(left_entity: pd.Series, right_entity: pd.Series, features,
                                 entity_grad_extr: EntityGradientExtractor, num_words: int = None):
    assert isinstance(left_entity, pd.Series), "Wrong data type for parameter 'left_entity'."
    assert isinstance(right_entity, pd.Series), "Wrong data type for parameter 'right_entity'."
    params = ["input_ids", "attention_mask", "token_type_ids"]
    assert all([p in features for p in params]), "Wrong data format for parameter 'features'."
    assert isinstance(entity_grad_extr, EntityGradientExtractor), "Wrong data type for parameter 'entity_grad_extr'."
    if num_words is not None:
        assert isinstance(num_words, int), "Wrong data type for parameter 'num_words'."

    data = [[left_entity, right_entity, features]]
    grad_params = entity_grad_extr.extract(data)
    assert isinstance(grad_params, list)
    assert len(grad_params) == 1
    row_grad_params = grad_params[0]
    assert isinstance(row_grad_params, dict)
    assert 'grad' in row_grad_params
    row_grad_data = row_grad_params['grad']
    assert isinstance(row_grad_data, dict)
    assert 'all' in row_grad_data
    assert 'all_grad' in row_grad_data
    row_grad_words = row_grad_data['all']
    row_grads = row_grad_data['all_grad']
    assert isinstance(row_grad_words, list)
    assert isinstance(row_grads, list)
    sorted_words = sorted(zip(row_grad_words, row_grads), key=lambda x: x[1], reverse=True)

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

