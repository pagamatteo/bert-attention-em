import pandas as pd
import numpy as np
import torch
from models.em_dataset import EMDataset
from tqdm import tqdm


class AttentionExtractor(object):
    """
    This class extracts the attention maps generated by the input model on the
    provided data.
    The following parameters are returned for each record of the dataset:
    - attns: the attention maps
    - preds: the prediction generated by the model
    - tokens: the record word pieces
    - all the other parameters provided in the record features
    """

    def __init__(self, dataset: EMDataset, model):

        assert isinstance(dataset, EMDataset), "Wrong data type for parameter 'dataset'."

        self.dataset = dataset
        self.tokenizer = dataset.tokenizer
        self.model = model

        self.model.eval()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        left_entity, right_entity, features = self.dataset[idx]

        assert isinstance(left_entity, pd.Series), "Wrong data type for parameter 'left_entity'."
        assert isinstance(right_entity, pd.Series), "Wrong data type for parameter 'right_entity'."
        assert 'input_ids' in features, "'input_ids' not found."
        assert 'attention_mask' in features, "'attention_mask' not found."
        assert 'token_type_ids' in features, "'token_type_ids' not found."

        input_ids = features['input_ids']
        attention_mask = features['attention_mask']
        token_type_ids = features['token_type_ids']

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
            logits = outputs['logits']
            attns = outputs['attentions']

        preds = torch.argmax(logits, axis=1)

        param = {}
        # remove useless features and preserve other features
        useless_features = ['input_ids']
        for f in features:
            if f not in useless_features:
                param[f] = features[f]
        param["tokens"] = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        param["attns"] = attns
        param["preds"] = preds

        return left_entity, right_entity, param


class AttributeAttentionExtractor(AttentionExtractor):
    """
    This class creates the attribute-level attention maps by aggregating the
    word-piece-level attention maps generated by the input model on the provided
    data.
    The following parameters are returned for each record of the dataset:
    - attns: the attribute-level attention maps
    - preds: the prediction generated by the model
    - tokens: the record word pieces
    - text_units: the attributes describing the single entity in the record
    - all the other parameters provided in the record features
    """

    def __init__(self, dataset: EMDataset, model):

        super().__init__(dataset, model)
        self.dataset_len = len(dataset)
        self.tokenizer = dataset.tokenizer
        self.max_len = dataset.max_len
        self.attrs = dataset.columns
        self.tokenization = dataset.tokenization
        self.invalid_attr_attn_maps = 0

    def _get_head_attr_attn(self, head_attn: torch.Tensor, left_idxs: list,
                            right_idxs: list):

        assert isinstance(head_attn, torch.Tensor), "Wrong data type for parameter 'head_attn'."
        assert head_attn.ndim == 2, "Attention map is not bi-dimensional"
        assert isinstance(left_idxs, list), "Wrong data type for parameter 'left_idxs'."
        assert isinstance(right_idxs, list), "Wrong data type for parameter 'right_idxs'."
        assert len(left_idxs) == len(right_idxs) == len(
            self.attrs), "The number of attribute attention indexes is different from the number of attributes."

        all_idxs = left_idxs + right_idxs
        head_attn = head_attn.detach().numpy()

        # head_attn (n x n) -> softmax applied at column level
        assert head_attn.sum(1).sum() == head_attn.shape[0]

        # sum attention scores by attribute along the columns in order to obtain for
        # each token its attribute attentions
        attr_to_word_attn = np.empty((head_attn.shape[0], len(all_idxs)))
        for idx, (attr_start, attr_end) in enumerate(all_idxs):
            attr_to_word_attn[:, idx] = head_attn[:, attr_start:attr_end].sum(1)

        # normalize the attention scores by columns to make them to sum to 1 (as the
        # original softmax)
        min_by_cols = attr_to_word_attn.min(1).reshape((-1, 1))
        max_by_cols = attr_to_word_attn.max(1).reshape((-1, 1))
        attr_to_word_attn = (attr_to_word_attn - min_by_cols) / (max_by_cols - min_by_cols)
        attr_to_word_attn /= attr_to_word_attn.sum(1).reshape((-1, 1))

        assert attr_to_word_attn.sum(1).sum() == attr_to_word_attn.shape[0], ""

        # average the attention scores that refer to the same attribute along the
        # rows in order to obtain for each attribute its attribute attentions
        attr_to_attr_attn = np.empty((len(all_idxs), len(all_idxs)))
        for idx, (attr_start, attr_end) in enumerate(all_idxs):
            attr_to_attr_attn[idx, :] = attr_to_word_attn[attr_start:attr_end, :].mean(0)

        # normalize the attention scores by columns to make them to sum to 1 (as the
        # original softmax)
        min_by_cols = attr_to_attr_attn.min(1).reshape((-1, 1))
        max_by_cols = attr_to_attr_attn.max(1).reshape((-1, 1))
        attr_to_attr_attn = (attr_to_attr_attn - min_by_cols) / (max_by_cols - min_by_cols)
        attr_to_attr_attn /= attr_to_attr_attn.sum(1).reshape((-1, 1))

        assert int(round(attr_to_attr_attn.sum(1).sum())) == attr_to_attr_attn.shape[0]

        return attr_to_attr_attn

    def _get_sent_word_idxs(self, offsets: list):

        assert isinstance(offsets, list), "Wrong data type for parameter 'offsets'."
        assert len(offsets) > 0, "No offsets provided."

        # aggregate all tokens of the sentence that refer to the same word
        # these tokens can be detected by searching for adjacent offsets from the
        # `offset_mapping` parameter
        tokens_to_sent_offsets = offsets[:]
        tokens_by_word = []  # this list will aggregate the token offsets by word
        prec_token_offsets = None
        tokens_in_word = []  # this list will accumulate all the tokens that refer to a target word
        words_offsets = []  # this list will store for each word the range of token idxs that refer to it
        for ix, token_offsets in enumerate(tokens_to_sent_offsets):

            # special tokens (e.g., [CLS], [SEP]) do not refer to any words
            # their offsets are equal to (0, 0)
            if token_offsets == [0, 0]:

                # save all the tokens that refer to the previous word
                if len(tokens_in_word) > 0:
                    l = int(np.sum([len(x) for x in tokens_by_word]))
                    words_offsets.append((l, l + len(tokens_in_word)))
                    tokens_by_word.append(tokens_in_word)
                    prec_token_offsets = None
                    tokens_in_word = []

                l = int(np.sum([len(x) for x in tokens_by_word]))
                # words_offsets.append((l, l + 1))
                tokens_by_word.append([token_offsets])
                continue

            if prec_token_offsets is None:
                tokens_in_word.append(token_offsets)
            else:
                # if the offsets of the current and previous tokens are adjacent then they
                # refer to the same word
                if prec_token_offsets[1] == token_offsets[0]:
                    tokens_in_word.append(token_offsets)
                else:
                    # the current token refers to a new word

                    # save all the tokens that refer to the previous word
                    l = int(np.sum([len(x) for x in tokens_by_word]))
                    words_offsets.append((l, l + len(tokens_in_word)))
                    tokens_by_word.append(tokens_in_word)

                    tokens_in_word = [token_offsets]

            prec_token_offsets = token_offsets

        # Note that 'words_offsets' contains only real word offsets, i.e. offsets
        # for special tokens (e.g., [CLS], [SEP], [PAD]), except for the [UNK]
        # token, are omitted

        return words_offsets

    def _get_pair_sent_word_idxs(self, encoded_pair_sent):

        assert 'offset_mapping' in encoded_pair_sent, "'encoded_pair_sent' doesn't include the 'offset_mapping' param."

        # split the offset mappings at sentence level by exploting the [SEP] which
        # is identified with the offsets [0, 0] (as any other special tokens)
        offsets = encoded_pair_sent['offset_mapping'].squeeze(0).tolist()
        sep_idx = offsets[1:].index([0, 0])  # ignore the [CLS] token at the index 0
        left_offsets = offsets[:sep_idx + 2]
        right_offsets = offsets[sep_idx + 1:]

        left_word_idxs = self._get_sent_word_idxs(left_offsets)
        right_word_idxs = self._get_sent_word_idxs(right_offsets)

        return left_word_idxs, right_word_idxs

    def _get_entity_pair_attr_idxs(self, left_entity: pd.Series,
                                   right_entity: pd.Series, features: dict):

        assert isinstance(left_entity, pd.Series), "Wrong data type for parameter 'left_entity'."
        assert isinstance(right_entity, pd.Series), "Wrong data type for parameter 'right_entity'."
        assert isinstance(features, dict), "Wrong data type for parameter 'features'."

        sent1 = ""
        sent2 = ""
        left_attr_len_map = []
        for attr, attr_val in left_entity.iteritems():
            sent1 += "{} ".format(str(attr_val))
            left_attr_len_map.append(len(str(attr_val).split()))
        right_attr_len_map = []
        for attr, attr_val in right_entity.iteritems():
            sent2 += "{} ".format(str(attr_val))
            right_attr_len_map.append(len(str(attr_val).split()))
        sent1 = sent1[:-1]
        sent2 = sent2[:-1]
        n_words = len(sent1.split()) + len(sent2.split())

        encoded = self.tokenizer(sent1, sent2, padding='max_length', truncation=True,
                                 return_tensors="pt", max_length=self.max_len,
                                 add_special_tokens=True, pad_to_max_length=True,
                                 return_attention_mask=False,
                                 return_offsets_mapping=True)

        left_word_idxs, right_word_idxs = self._get_pair_sent_word_idxs(encoded)

        cum_len = 0
        left_attr_idxs = []
        last_left_attr_idx = None
        left_trunc = False
        for left_attr_len in left_attr_len_map:

            if left_trunc:
                left_attr_idxs.append(None)
            else:

                left_attr_words = left_word_idxs[cum_len: cum_len + left_attr_len]
                assert len(left_attr_words) <= left_attr_len
                if len(left_attr_words) < left_attr_len or len(left_attr_words) == 0:
                    left_trunc = True

                if len(left_attr_words) > 0:
                    left_attr_start_idx = left_attr_words[0][0]
                    left_attr_end_idx = left_attr_words[-1][1]
                    left_attr_idxs.append((left_attr_start_idx, left_attr_end_idx))
                    last_left_attr_idx = left_attr_end_idx

                    cum_len += left_attr_len
                else:
                    left_attr_idxs.append(None)

        assert last_left_attr_idx is not None

        cum_len = 0
        right_attr_idxs = []
        right_trunc = False
        for iix, right_attr_len in enumerate(right_attr_len_map):

            if right_trunc:
                right_attr_idxs.append(None)
            else:

                right_attr_words = right_word_idxs[cum_len: cum_len + right_attr_len]
                assert len(right_attr_words) <= right_attr_len
                if len(right_attr_words) < right_attr_len or len(right_attr_words) == 0:
                    right_trunc = True

                if len(right_attr_words) > 0:
                    right_attr_start_idx = right_attr_words[0][0]
                    right_attr_end_idx = right_attr_words[-1][1]
                    right_attr_idxs.append((right_attr_start_idx + last_left_attr_idx,
                                            right_attr_end_idx + last_left_attr_idx))

                    cum_len += right_attr_len
                else:
                    right_attr_idxs.append(None)

        return left_attr_idxs, right_attr_idxs

    def _check_attr_idxs_consistency(self, left_idxs, right_idxs, tokens):

        left_trunc = False
        right_trunc = False
        assert len(left_idxs) == len(self.attrs)
        assert len(right_idxs) == len(self.attrs)
        assert left_idxs[0][0] == 1  # ignore [CLS] token

        # check left attribute indexes consistency
        sep_idx = tokens.index('[SEP]')
        last_left_valid_attr = None
        idx = 0
        while last_left_valid_attr is None:
            last_left_valid_attr = left_idxs[-(1 + idx)]
            idx += 1
        assert last_left_valid_attr[1] == sep_idx
        if idx > 1:
            left_trunc = True

        assert last_left_valid_attr[1] + 1 == right_idxs[0][0]  # ignore [SEP] token

        # check right attribute indexes consistency
        last_sep_idx = tokens[sep_idx + 1:].index('[SEP]') + sep_idx + 1
        last_right_valid_attr = None
        idx = 0
        while last_right_valid_attr is None:
            last_right_valid_attr = right_idxs[-(1 + idx)]
            idx += 1
        assert last_right_valid_attr[1] == last_sep_idx
        if idx > 1:
            assert last_right_valid_attr[1] == self.max_len - 1
            right_trunc = True

        return left_trunc or right_trunc

    def _get_attr_idxs(self, num_entity_attrs: int, tokens: np.ndarray,
                       offset=None):
        assert isinstance(num_entity_attrs, int), "Wrong data type for parameter 'num_entity_attrs'."
        assert num_entity_attrs > 0, "Wrong value for parameter 'num_entity_attrs' (>0)."
        assert isinstance(tokens, np.ndarray), "Wrong data type for parameter 'tokens'."
        assert len(tokens) > 0, "No tokens provided: empty array."

        tokens_by_attr = np.where(tokens == '[SEP]')[0]
        if offset is not None:
            tokens_by_attr += offset
        truncation = False
        idxs = None

        if len(tokens_by_attr) < num_entity_attrs:
            truncation = True

            return idxs, truncation

        idxs = []
        start = 0
        if offset is not None:
            start = offset
        for ix in range(num_entity_attrs):
            idxs.append((start + 1, tokens_by_attr[ix]))
            start = tokens_by_attr[ix]

        if np.array([(item[1] - item[0]) == 0 for item in idxs]).sum() > 0:
            truncation = True

        return idxs, truncation

    def _get_attr_pair_idxs(self, left_entity: pd.Series, right_entity: pd.Series,
                            features: dict):
        assert isinstance(left_entity, pd.Series), "Wrong data type for parameter 'left_entity'."
        assert isinstance(right_entity, pd.Series), "Wrong data type for parameter 'right_entity'."
        assert isinstance(features, dict), "Wrong data type for parameter 'features'."
        tokens = features['tokens']
        token_type_ids = features['token_type_ids'].squeeze(0)
        attention_mask = features['attention_mask'].squeeze(0)

        # print([(ix, r) for ix, r in enumerate(tokens)])

        # get the tokens associated to left and right entities
        valid_token_type_ids = token_type_ids[attention_mask == 1]
        assert set(valid_token_type_ids.detach().numpy()) == {0, 1}
        sent_split_idx = np.where(valid_token_type_ids == 1)[0]
        assert isinstance(sent_split_idx, np.ndarray)
        assert len(sent_split_idx) > 0
        sent_split_idx = sent_split_idx[0]
        assert valid_token_type_ids[sent_split_idx:].sum() == len(valid_token_type_ids) - sent_split_idx

        left_tokens = np.array(tokens[:sent_split_idx])
        right_tokens = np.array(['[CLS]'] + tokens[sent_split_idx:])  # add fake [CLS] token only for alignment

        # get attribute indexes from the left entity
        left_idxs, left_trunc = self._get_attr_idxs(len(left_entity), left_tokens)
        left_last_idx = None
        if left_idxs is not None:
            left_last_idx = left_idxs[-1][1]

        # get attribute indexes from the right entity
        right_idxs, right_trunc = self._get_attr_idxs(len(right_entity),
                                                      right_tokens,
                                                      offset=left_last_idx)

        return left_idxs, right_idxs, left_trunc or right_trunc

    def _get_attr_attn(self, left_entity: pd.Series, right_entity: pd.Series,
                       features: dict):

        # check data types
        assert isinstance(left_entity, pd.Series), "Wrong data type for parameter 'left_entity'."
        assert isinstance(right_entity, pd.Series), "Wrong data type for parameter 'right_entity'."
        assert isinstance(features, dict), "Wrong data type for parameter 'features'."

        # 'features' should contain the attention maps, the preds and the tokens
        assert "attns" in features, "No 'attns' found in the features."
        assert "preds" in features, "No 'preds' found in the features."
        assert "tokens" in features, "No 'tokens' found in the features."
        assert "token_type_ids" in features, "No 'token_type_ids' found in the features."
        assert 'attention_mask' in features, "No 'attention_mask' found in the features."

        attns = features["attns"]
        n_layers = len(attns)
        n_heads = attns[0].shape[1]
        n_attrs = len(self.attrs) * 2

        # in order to aggregate the attention scores at attribute level it is needed
        # to know the word pieces that refer to the same attribute

        if self.tokenization == 'sent_pair':
            left_idxs, right_idxs = self._get_entity_pair_attr_idxs(left_entity,
                                                                    right_entity,
                                                                    features)

            # check attribute indexes consistency
            truncation = self._check_attr_idxs_consistency(left_idxs, right_idxs,
                                                           features["tokens"])
        elif self.tokenization == 'attr':
            # left_idxs, right_idxs, truncation = self._get_attr_idxs(left_entity,
            #                                                         right_entity,
            #                                                         features)
            idxs, truncation = self._get_attr_idxs(len(left_entity) +
                                                   len(right_entity),
                                                   np.array(features["tokens"]))
            if not truncation:
                left_idxs = idxs[:len(left_entity)]
                right_idxs = idxs[len(left_entity):]

        elif self.tokenization == 'attr_pair':
            left_idxs, right_idxs, truncation = self._get_attr_pair_idxs(left_entity,
                                                                         right_entity,
                                                                         features)

        if truncation:
            # print("TRUNC")
            # print(left_entity.values)
            # print(right_entity.values)
            # print(features["tokens"])

            # some attribute has been truncated, so don't produce the attribute-level
            # attention maps
            features['attns'] = None
            features['text_units'] = self.attrs

            self.invalid_attr_attn_maps += 1

            return left_entity, right_entity, features

        # else:
        #   print("OK")
        #   print(left_entity.values)
        #   print(right_entity.values)
        #   print("LEFT")
        #   for l in left_idxs:
        #     print(features["tokens"][l[0]:l[1]])
        #   print("RIGHT")
        #   for r in right_idxs:
        #     print(features["tokens"][r[0]:r[1]])
        #   print("\n\n")

        attr_attns = np.empty((n_layers, n_heads, n_attrs, n_attrs))
        for layer in range(n_layers):
            heads = attns[layer].squeeze(0)
            for head in range(n_heads):
                head_attn = heads[head]
                attr_attns[layer, head, :, :] = self._get_head_attr_attn(head_attn,
                                                                         left_idxs,
                                                                         right_idxs)

        # override word-piece-level with attribute-level attention maps
        features['attns'] = attr_attns
        features['text_units'] = self.attrs

        return left_entity, right_entity, features

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        left_entity, right_entity, features = super().__getitem__(idx)

        return self._get_attr_attn(left_entity, right_entity, features)

    def get_num_invalid_attr_attn_maps(self):
        return self.invalid_attr_attn_maps

    def extract(self, idx):
        return self[idx]

    def extract_all(self):

        attr_features = []
        for features in tqdm(self):
            attr_features.append(features)

        return attr_features