import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_sent_word_idxs(offsets: list):
    """
    This function returns the indexes of the words included in a sentence tokenized with BERT.

    :param offsets: offset_mapping parameter extracted from a BERT tokenizer
    :return: list of tuples that indicate the start end indexes of a word in a BERT-tokenized sentence
    """

    assert isinstance(offsets, list), "offsets param is not a list."
    assert len(offsets) > 0, "Empty offsets."

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


def get_sent_pair_word_idxs(sent1: str, sent2: str, tokenizer, max_len: int):
    """
    This function returns the indexes of the words included in a pair of sentences tokenized with BERT.

    :param sent1: first sentence
    :param sent2: second sentence
    :param tokenizer: BERT tokenizer
    :param max_len: max number of tokens included in the sentence pair
    :return: (sentence 1 word indexes, sentence 2 word indexes, tokens)
    """

    encoded_pair_sent = tokenizer(sent1, sent2, padding='max_length', truncation=True,
                                  return_tensors="pt", max_length=max_len,
                                  add_special_tokens=True, pad_to_max_length=True,
                                  return_attention_mask=False,
                                  return_offsets_mapping=True)

    tokens = tokenizer.convert_ids_to_tokens(encoded_pair_sent['input_ids'][0])

    # split the offset mappings at sentence level by exploting the [SEP] which
    # is identified with the offsets [0, 0] (as any other special tokens)
    offsets = encoded_pair_sent['offset_mapping'].squeeze(0).tolist()
    sep_idx = offsets[1:].index([0, 0])  # ignore the [CLS] token at the index 0
    left_offsets = offsets[:sep_idx + 2]
    right_offsets = offsets[sep_idx + 1:]

    left_word_idxs = get_sent_word_idxs(left_offsets)
    right_word_idxs = get_sent_word_idxs(right_offsets)

    return left_word_idxs, right_word_idxs, tokens


def get_entity_pair_attr_idxs(left_entity: pd.Series, right_entity: pd.Series, tokenizer, max_len: int):
    """
    This function returns the indexes of the attributes included in a pair of entities tokenized with BERT.
    Optionally the function can return also the word or token indexes.

    :param left_entity: Pandas' Series object containing the data of the first entity
    :param right_entity: Pandas' Series object containing the data of the second entity
    :param tokenizer: BERT tokenizer
    :param max_len: max number of tokens included in the sentence pair
    :return: dictionary containing data about attribute/word/token indexes for the input pair of entities
    """

    def _check_attr_idxs_consistency(left_idxs: list, right_idxs: list, tokens: list, num_attrs: int, max_len: int):

        left_trunc = False
        right_trunc = False
        assert len(left_idxs) == num_attrs
        assert len(right_idxs) == num_attrs
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
            assert last_right_valid_attr[1] == max_len - 1
            right_trunc = True

        return left_trunc or right_trunc

    assert isinstance(left_entity, pd.Series), "Wrong data type for parameter 'left_entity'."
    assert isinstance(right_entity, pd.Series), "Wrong data type for parameter 'right_entity'."
    assert isinstance(max_len, int), "Wrong data type for parameter 'max_len'."

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

    left_word_idxs, right_word_idxs, tokens = get_sent_pair_word_idxs(sent1, sent2, tokenizer, max_len)

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

    tokens = [t for t in tokens if t != tokenizer.pad_token]

    truncation = _check_attr_idxs_consistency(left_attr_idxs, right_attr_idxs, tokens, len(left_entity), max_len)
    if truncation:
        logging.info("Exceeded max len -> truncated attributes. No attribute data will be returned.")
        return None

    out_data = {
        'left_names': [f'l_{c}' for c in list(left_entity.index)],
        'right_names': [f'r_{c}' for c in list(right_entity.index)],
        'left_idxs': left_attr_idxs,
        'right_idxs': right_attr_idxs
    }

    return out_data
