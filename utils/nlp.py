import nltk
from nltk.corpus import wordnet
import random
import logging
nltk.download('wordnet')


def get_synonyms_from_sent(word, sent):
    synonyms = set([])
    for synset in wordnet.synsets(word):
        for lemma in synset.lemma_names():
            if lemma in sent and lemma != word:
                synonyms.add(lemma)
    return synonyms


def get_synonyms_from_sent_pair(words1, words2, topk: int = None):
    assert isinstance(words1, list), "Wrong data type for parameter 'words1'."
    assert all([isinstance(w, str) for w in words1]), "Wrong data format for parameter 'words1'."
    assert isinstance(words2, list), "Wrong data type for parameter 'words2'."
    assert all([isinstance(w, str) for w in words2]), "Wrong data format for parameter 'words2'."
    if topk is not None:
        assert isinstance(topk, int), "Wrong data type for parameter 'topk'."

    synonyms = []
    for word in words1:
        word_synonyms = get_synonyms_from_sent(word, words2)
        synonyms += [(word, syn) for syn in word_synonyms]

    # for word in words2:
    #     word_synonyms = get_synonyms_from_sent(word, words1)
    #     synonyms += [(syn, word) for syn in word_synonyms]

    synonyms = list(set(synonyms))
    synonyms = [{'left': syn[0], 'right': syn[1]} for syn in synonyms]

    if len(synonyms) == 0:
        return None

    if topk is None:
        out_data = synonyms
    elif topk == 1:
        out_data = synonyms[0]
    else:
        out_data = synonyms[:topk]

    return out_data


def get_random_words_from_sent_pair(words1, words2, num_words: int, exclude_synonyms: bool = False, seed: int = 42):
    assert isinstance(words1, list), "Wrong data type for parameter 'words1'."
    assert all([isinstance(w, str) for w in words1]), "Wrong data format for parameter 'words1'."
    assert isinstance(words2, list), "Wrong data type for parameter 'words2'."
    assert all([isinstance(w, str) for w in words2]), "Wrong data format for parameter 'words2'."
    assert isinstance(num_words, int), "Wrong data type for parameter 'num_words'."
    assert num_words <= len(words1) * len(words2), f"Too many words requested (max={len(words1) * len(words2)})."
    assert isinstance(exclude_synonyms, bool), "Wrong data type for parameter 'ignore_synonyms'."
    assert isinstance(seed, int), "Wrong data type for parameter 'seed'."

    random.seed(seed)
    words = []
    for i in range(num_words):
        w1 = random.choice(words1)
        w2 = random.choice(words2)

        if exclude_synonyms:
            attempt = 1
            while len(get_synonyms_from_sent(w1, [w2])) > 0 or w1 == w2:
                w1 = random.choice(words1)
                w2 = random.choice(words2)
                attempt += 1

                if attempt == 10:
                    break
            if attempt == 10:
                logging.info("Impossible to select not synonyms words.")

        words.append({'left': w1, 'right': w2})

    if num_words == 1:
        return words[0]
    return words


def get_common_words_from_sent_pair(words1, words2, num_words: int, seed: int = 42):
    assert isinstance(words1, list), "Wrong data type for parameter 'words1'."
    assert all([isinstance(w, str) for w in words1]), "Wrong data format for parameter 'words1'."
    assert isinstance(words2, list), "Wrong data type for parameter 'words2'."
    assert all([isinstance(w, str) for w in words2]), "Wrong data format for parameter 'words2'."
    assert isinstance(num_words, int), "Wrong data type for parameter 'num_words'."
    assert isinstance(seed, int), "Wrong data type for parameter 'seed'."

    common_words = set(words1).intersection(set(words2))
    if len(common_words) == 0:
        logging.info("No common words found.")
        return None

    assert num_words <= len(common_words), f"Too many words requested (max={len(common_words)})."

    random.seed(seed)
    words = set([])
    idx = 0
    while len(words) < num_words:
        word = random.choice(list(common_words))
        words.add(word)
        idx += 1

    words = list(words)

    if num_words == 1:
        return {'left': words[0], 'right': words[0]}
    return [{'left': w, 'right': w} for w in words]


def simple_tokenization_and_clean(text: str):
    assert isinstance(text, str), "Wrong data type for parameter 'text'."

    # remove non-alphabetical and short words
    return [word for word in text.split() if word.isalpha() and len(word) > 3]
