import os

import numpy
from toolz import sliding_window

from blocks import config
from blocks.datasets import Dataset
from blocks.datasets.streams import CachedDataStream


class TextFileState(object):
    pass


class TextFile(Dataset):
    r"""Reads text files and numberizes them given a dictionary.

    Parameters
    ----------
    files : list of str
        The names of the files in order which they should be read. Each
        file is expected to have a sentence per line.
    dictionary : str or dict
        Either the path to a Pickled dictionary mapping tokens to integers,
        or the dictionary itself. At the very least this dictionary must
        map the unknown word-token to an integer.
    bos_token : str or None, optional
        The beginning-of-sentence (BOS) token in the dictionary that
        denotes the beginning of a sentence. Is ``<S>`` by default. If
        passed ``None`` no beginning of sentence markers will be added.
    eos_token : str or None, optional
        The end-of-sentence (EOS) token is ``</S>`` by default, see
        ``bos_taken``.
    unk_token : str, optional
        The token in the dictionary to fall back on when a token could not
        be found in the dictionary.
    level : 'word' or 'character', optional
        If 'word' the dictionary is expected to contain full words. The
        sentences in the text file will be split at the spaces, and each
        word replaced with its number as given by the dictionary, resulting
        in each example being a single list of numbers. If 'character' the
        dictionary is expected to contain single letters as keys. A single
        example will be a list of character numbers, starting with the
        first non-whitespace character and finishing with the last one.
    preprocess : function, optional
        A function which takes a sentence (string) as an input and returns
        a modified string. For example ``str.lower`` in order to lowercase
        the sentence before numberizing.

    Examples
    --------
    >>> with open('sentences.txt', 'w') as f:
    ...     _ = f.write("This is a sentence\n")
    ...     _ = f.write("This another one")
    >>> dictionary = {'<UNK>': 0, '</S>': 1, 'this': 2, 'a': 3, 'one': 4}
    >>> text_data = TextFile(files=['sentences.txt'],
    ...                      dictionary=dictionary, bos_token=None,
    ...                      preprocess=str.lower)
    >>> for data in text_data.get_default_stream().get_epoch_iterator():
    ...     print(data)
    ([2, 0, 3, 0, 1],)
    ([2, 0, 4, 1],)

    .. doctest::
       :hide:

       >>> import os
       >>> os.remove('sentences.txt')

    """
    provides_sources = ('features',)
    default_scheme = None

    def __init__(self, files, dictionary, bos_token='<S>', eos_token='</S>',
                 unk_token='<UNK>', level='word', preprocess=None):
        self.files = files
        self.dictionary = dictionary
        if bos_token is not None and bos_token not in dictionary:
            raise ValueError
        self.bos_token = bos_token
        if eos_token is not None and eos_token not in dictionary:
            raise ValueError
        self.eos_token = eos_token
        if unk_token not in dictionary:
            raise ValueError
        self.unk_token = unk_token
        if level not in ('word', 'character'):
            raise ValueError
        self.level = level
        self.preprocess = preprocess
        super(TextFile, self).__init__()

    def open(self):
        state = TextFileState()
        state.current_index = 0
        state.file = self._open_file(state.current_index)
        return state

    def _open_file(self, partition_index):
        return open(self.files[partition_index])

    def get_data(self, state=None, request=None):
        if request is not None:
            raise ValueError
        while True:
            if state.file is None:
                raise StopIteration
            sentence = state.file.readline()
            if not sentence:
                state.file.close()
                state.file = None
                if state.current_index == len(self.files) - 1:
                    raise StopIteration
                else:
                    state.current_index += 1
                    state.file = self._open_file(state.current_index)
            else:
                break
        if self.preprocess is not None:
            sentence = self.preprocess(sentence)
        data = [self.dictionary[self.bos_token]] if self.bos_token else []
        if self.level == 'word':
            data += [self.dictionary.get(word, self.dictionary[self.unk_token])
                     for word in sentence.split()]
        else:
            data += [self.dictionary.get(char, self.dictionary[self.unk_token])
                     for char in sentence.strip()]
        data += [self.dictionary[self.eos_token]] if self.eos_token else []
        return (data,)


class NGramStream(CachedDataStream):
    """Return n-grams from a stream.

    This data stream wrapper takes as an input a data stream outputting
    batches of sentences. From these sentences n-grams of a fixed order
    (e.g. bigrams, trigrams, etc.) are extracted and returned. It also
    creates a ``targets`` data source. For each example, the target is the
    word immediately following that n-gram. It is normally used for
    language modeling, where we try to predict the next word from the
    previous n words.

    Parameters
    ----------
    ngram_order : int
        The order of the n-grams to output e.g. 3 for trigrams.
    data_stream : :class:`.DataStream` instance
        The data stream providing sentences. Each example is assumed to be
        a list of integers.
    target_source : str, optional
        This data stream adds a new source for the target words. By default
        this source is 'targets'.

    Notes
    -----
    This class inherits from :class:`.CachedDataStream` because it makes
    use of a cache to store the sentences from the wrapped data stream in.

    """
    def __init__(self, ngram_order, data_stream, target_source='targets',
                 iteration_scheme=None):
        if len(data_stream.sources) > 1:
            raise ValueError
        super(NGramStream, self).__init__(data_stream, iteration_scheme)
        self.sources = self.sources + (target_source,)
        self.ngram_order = ngram_order

    def get_data(self, request=None):
        features, targets = [], []
        for _, sentence in enumerate(self.cache[0]):
            features.append(list(
                sliding_window(self.ngram_order,
                               sentence[:-1]))[:request - len(features)])
            targets.append(
                sentence[self.ngram_order:][:request - len(targets)])
            self.cache[0][0] = self.cache[0][0][request:]
            if not self.cache[0][0]:
                self.cache[0].pop(0)
                if not self.cache[0]:
                    self._cache()
            if len(features) == request:
                break
        return tuple(numpy.asarray(data) for data in (features, targets))
