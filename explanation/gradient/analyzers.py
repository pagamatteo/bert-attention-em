from explanation.gradient.extractors import EntityGradientExtractor
from utils.result_collector import BinaryClassificationResultsAggregator
import numpy as np
import string
import re
import spacy
from spacy.tokenizer import Tokenizer
import pandas as pd


class TopKGradientAnalyzer(object):
    def __init__(self, grads_data: list, topk: int, metric: str = 'avg', target_entity: str = 'all'):
        EntityGradientExtractor.check_extracted_grad(grads_data)
        assert isinstance(topk, int), "Wrong data type for parameter 'topk'."
        assert isinstance(metric, str), "Wrong data type for parameter 'metric'."
        assert metric in EntityGradientExtractor.grad_agg_fns, "Wrong metric."

        sel_grads_data = []
        for g in grads_data:
            item = {
                'label': g['label'],
                'pred': g['pred'],
                'grad': {'text_units': g['grad'][target_entity], 'values': g['grad'][f'{target_entity}_grad'][metric]},
            }
            sel_grads_data.append(item)
        self.grads_data = sel_grads_data
        self.topk = topk
        self.analysis_types = ['pos', 'str_type']
        self.pos_model = spacy.load('en_core_web_sm')
        self.pos_model.tokenizer = Tokenizer(self.pos_model.vocab, token_match=re.compile(r'\S+').match)

    def analyze(self, analysis_type: str, target_categories: list = None, ignore_special: bool = True):
        assert isinstance(analysis_type, str), "Wrong data type for parameter 'analysis_type'."
        assert analysis_type in self.analysis_types, f"Wrong type: {analysis_type} not in {self.analysis_types}."

        def get_topk_text_units(x, ignore_special, topk):
            text_units = np.array(x['text_units'])
            sep_idxs = list(np.where(text_units == '[SEP]')[0])
            skip_idxs = [0] + sep_idxs
            g = x['values']
            if ignore_special:
                if text_units[0] == '[CLS]':
                    g = [g[i] for i in range(len(g)) if i not in skip_idxs]
                    text_units = [text_units[i].replace("l_", "").replace("r_", "") for i in range(len(text_units)) if i not in skip_idxs]
            topk_text_unit_idxs = np.array(g).argsort()[-topk:][::-1]
            return text_units, topk_text_unit_idxs

        def get_pos_tag(word):
            pos_tag = word.pos_

            # adjust Spacy's pos tags
            if any(c.isdigit() for c in word.text) and pos_tag != 'NUM':
                pos_tag = 'NUM'
            if not word.text.isalpha() and pos_tag == 'PROPN':
                pos_tag = 'X'
            if word.text == "'" and pos_tag != 'PUNCT':
                pos_tag = 'PUNCT'

            # aggregate Spacy's pos tags
            if pos_tag in ['ADJ', 'ADV', 'AUX', 'NOUN', 'PROPN', 'VERB']:
                pos_tag = 'TEXT'
            elif pos_tag in ['SYM', 'PUNCT']:
                pos_tag = 'PUNCT'
            elif pos_tag in ['NUM']:
                pos_tag = 'NUM&SYM'
            else:
                pos_tag = 'CONN'
            return pos_tag

        aggregator = BinaryClassificationResultsAggregator('grad', target_categories=target_categories)
        aggregator.add_batch_data(self.grads_data)
        agg_grads_data = aggregator.get_results()

        out_data = {}
        for cat in agg_grads_data:
            cat_grads_data = agg_grads_data[cat]

            if cat_grads_data is None:
                continue

            cat_out_data = {}
            for grad_data in cat_grads_data:
                # print("--------")
                text_units, topk_text_unit_idxs = get_topk_text_units(grad_data, ignore_special, self.topk)

                if analysis_type == 'str_type':
                    stats_cat = ['alpha', 'punct', 'num', 'no-alpha']
                    for topk_text_unit_idx in topk_text_unit_idxs:
                        tu = text_units[topk_text_unit_idx]
                        if tu in string.punctuation:
                            text_cat = 'punct'
                        elif any(c.isdigit() for c in tu):
                            text_cat = 'num'
                        elif not tu.isalpha():
                            text_cat = 'no-alpha'
                        elif tu.isalpha():
                            text_cat = 'alpha'
                        else:
                            text_cat = 'other'
                        # print(tu, text_cat)
                        if text_cat not in cat_out_data:
                            cat_out_data[text_cat] = 1
                        else:
                            cat_out_data[text_cat] += 1

                elif analysis_type == 'pos':
                    stats_cat = ['TEXT', 'PUNCT', 'NUM&SYM', 'CONN']
                    sent = ' '.join(text_units)
                    sent = self.pos_model(sent)
                    for word_idx, word in enumerate(sent):
                        if word_idx in topk_text_unit_idxs:
                            pos_tag = get_pos_tag(word)
                            # print(word, pos_tag)
                            if pos_tag not in cat_out_data:
                                cat_out_data[pos_tag] = 1
                            else:
                                cat_out_data[pos_tag] += 1
                else:
                    raise NotImplementedError()

            tot_count = np.sum(list(cat_out_data.values()))
            out_data[cat] = {k: int((v / tot_count) * 100) for k, v in cat_out_data.items()}

        stats = pd.DataFrame(out_data)
        stats = stats.rename(columns={'all_pos': 'match', 'all_neg': 'non_match'})
        stats = stats.T
        stats = stats[stats_cat]

        return stats
