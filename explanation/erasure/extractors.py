from models.em_dataset import EMDataset
import pandas as pd
from utils.nlp import simple_tokenization_and_clean
import torch
from torch.nn.functional import softmax
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
import logging


class DeltaPredictionExtractor(object):
    def __init__(self, model, tokenizer, word_selector_fns: list, delta_metrics: list, max_len: int = 128,
                 only_left_word: bool = False, only_right_word: bool = False):

        assert isinstance(word_selector_fns, list), "Wrong data type for parameter 'word_selector_fns'."
        assert len(word_selector_fns) > 0, "Empty word_selector_fns."
        assert isinstance(delta_metrics, list), "Wrong data type for parameter 'delta_metrics'."
        assert len(delta_metrics) > 0, "Empty delta_metrics."
        assert all([isinstance(m, str) for m in delta_metrics]), "Wrong data format for parameter 'delta_metrics'."
        available_metrics = {'jsd': jensenshannon, 'tvd': lambda x, y: (sum(abs(x - y)) / 2).item()}
        assert all([m in available_metrics for m in delta_metrics]), f"Delta metric not in {list(available_metrics)}."
        assert isinstance(max_len, int), "Wrong data type for parameter 'max_len'."
        assert isinstance(only_left_word, bool), "Wrong data type for parameter 'only_left_word'."
        assert isinstance(only_right_word, bool), "Wrong data type for parameter 'only_right_word'."

        self.model = model
        self.tokenizer = lambda x, y: tokenizer(x, y, padding='max_length', truncation=True, return_tensors="pt",
                                                max_length=max_len, add_special_tokens=True,
                                                pad_to_max_length=True, return_attention_mask=True)
        self.word_selector_fns = word_selector_fns
        self.delta_metrics = delta_metrics
        self.delta_fns = [available_metrics[dm] for dm in delta_metrics]
        self.only_left_word = only_left_word
        self.only_right_word = only_right_word

    def extract(self, data: EMDataset):
        assert isinstance(data, EMDataset), "Wrong data type for parameter 'data'."

        def get_pred(model, features):

            new_features = {}
            for key in ['input_ids', 'attention_mask', 'token_type_ids']:
                val = features[key]
                if val.ndim == 1:
                    val = val.unsqueeze(0)
                new_features[key] = val

            with torch.no_grad():
                outputs = model(input_ids=new_features['input_ids'], attention_mask=new_features['attention_mask'],
                                token_type_ids=new_features['token_type_ids'])
            logits = outputs['logits']
            probs = softmax(logits, dim=-1).squeeze(0)
            pred = torch.argmax(logits, dim=1).squeeze(0).item()

            return probs, pred

        def get_delta_scores(metric_names, metric_fns, score1, score2, pred1, pred2, label):
            out = {}
            for dm_idx in range(len(metric_names)):
                delta_metric = metric_names[dm_idx]
                delta_fn = metric_fns[dm_idx]
                out[delta_metric] = delta_fn(score1, score2)
            out['diff'] = (score2[label] - score1[label]).item()
            out['flip'] = pred1 != pred2

            return out

        out_data = [
            [] for _ in range(len(self.word_selector_fns))
        ]

        for idx, (left_entity, right_entity, features) in tqdm(enumerate(data)):
            original_label = features['labels'].item()
            original_sent1 = features['sent1']
            original_sent2 = features['sent2']
            left_sent = ' '.join([str(val) for val in left_entity if not pd.isnull(val)])
            left_words = simple_tokenization_and_clean(left_sent)
            right_sent = ' '.join([str(val) for val in right_entity if not pd.isnull(val)])
            right_words = simple_tokenization_and_clean(right_sent)

            # loop over all the word selector functions and try to retrieve some pair of words to remove
            word_pairs_to_remove = []
            for ws_idx in range(len(self.word_selector_fns)):
                word_selector_fn = self.word_selector_fns[ws_idx]

                # select the pair of words to remove
                if word_selector_fn.__code__.co_argcount == 2:      # the function expects only two params: (left and right words)
                    word_pair_to_remove = word_selector_fn(left_words, right_words)
                elif word_selector_fn.__code__.co_argcount == 3:    # the function needs also a seed for random choices
                    word_pair_to_remove = None
                    found = False
                    attempt = 0
                    while not found:
                        word_pair_to_remove = word_selector_fn(left_words, right_words, idx + attempt)

                        if word_pair_to_remove is None:
                            break

                        already_exists = False
                        for wptr in word_pairs_to_remove:
                            if len(set(wptr.values()).intersection(set(word_pair_to_remove.values()))) > 0:
                                already_exists = True
                                break
                        if not already_exists:
                            found = True

                        if attempt == 100:
                            logging.info("No word pairs uniqueness preserved.")
                            break
                        attempt += 1
                else:
                    raise ValueError("Too many arguments expected by the word_selector_fn.")

                if word_pair_to_remove is None:  # no word pair to remove
                    continue

                assert isinstance(word_pair_to_remove, dict), "word_pair_to_remove is not a dict."
                assert all(
                    [p in word_pair_to_remove for p in ['left', 'right']]), "word_pair_to_remove has a wrong format."
                assert all([isinstance(w, str) for w in word_pair_to_remove]), "word_pair_to_remove has a wrong format."

                word_pairs_to_remove.append(word_pair_to_remove)

            # if not all the word selector functions have extracted some words to remove, skip the current row
            if len(word_pairs_to_remove) != len(self.word_selector_fns):
                continue

            # get the model prediction for the original pair of sentences
            original_probs, original_pred = get_pred(self.model, features)

            for ws_idx, word_pair_to_remove in enumerate(word_pairs_to_remove):
                # remove the selected words from the pair of sentences
                removed_sent1 = original_sent1.split()
                removed_sent1.remove(word_pair_to_remove['left'])
                removed_sent1 = ' '.join(removed_sent1)
                removed_sent2 = original_sent2.split()
                removed_sent2.remove(word_pair_to_remove['right'])
                removed_sent2 = ' '.join(removed_sent2)

                # get the model prediction for a pair of sentences where the selected words have been removed
                pair_removed_features = self.tokenizer(removed_sent1, removed_sent2)
                pair_removed_probs, pair_removed_pred = get_pred(self.model, pair_removed_features)

                # get delta score and save
                pair_delta_res = {'label': original_label, 'pred': original_pred, 'probs': original_probs,
                                  'target_words': word_pair_to_remove, 'idx': idx}
                pair_delta_scores = get_delta_scores(self.delta_metrics, self.delta_fns, original_probs,
                                                     pair_removed_probs, original_pred, pair_removed_pred,
                                                     original_label)
                pair_delta_res.update(pair_delta_scores)

                # get the model prediction for a pair of sentences where only the left-selected word has been removed
                if self.only_left_word:
                    left_removed_features = self.tokenizer(removed_sent1, original_sent2)
                    left_removed_probs, left_removed_pred = get_pred(self.model, left_removed_features)

                    # get delta score and save
                    left_delta_res = pair_delta_res.copy()
                    left_delta_scores = get_delta_scores(self.delta_metrics, self.delta_fns, original_probs,
                                                         left_removed_probs, original_pred, left_removed_pred,
                                                         original_label)
                    left_delta_res.update(left_delta_scores)
                else:
                    left_delta_res = None

                # get the model prediction for a pair of sentences where only the right-selected word has been removed
                if self.only_right_word:
                    right_removed_features = self.tokenizer(original_sent1, removed_sent2)
                    right_removed_probs, right_removed_pred = get_pred(self.model, right_removed_features)

                    # get delta score and save
                    right_delta_res = pair_delta_res.copy()
                    right_delta_scores = get_delta_scores(self.delta_metrics, self.delta_fns, original_probs,
                                                          right_removed_probs, original_pred, right_removed_pred,
                                                          original_label)
                    right_delta_res.update(right_delta_scores)
                else:
                    right_delta_res = None

                all_delta_res = {
                    'pair': pair_delta_res,
                    'left': left_delta_res,
                    'right': right_delta_res,
                }

                out_data[ws_idx].append(all_delta_res)

        return out_data
