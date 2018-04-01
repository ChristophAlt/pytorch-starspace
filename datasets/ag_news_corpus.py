import os
import csv
import re

from torchtext import data


class AGNewsCorpus(data.Dataset):
    """
    AG News Corpus
    https://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html
    """

    urls = ['https://raw.githubusercontent.com/mhjabreel/CharCNN/master/data/ag_news_csv/train.csv',
            'https://raw.githubusercontent.com/mhjabreel/CharCNN/master/data/ag_news_csv/test.csv']
    name = 'ag_news'
    dirname = ''

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        examples = []

        with open(os.path.expanduser(path), 'r') as csv_file:
            for line in csv.reader(csv_file, quotechar='"', delimiter=','):
                label = line[0]

                text = ''
                for s in line[1:]:
                    text = text + ' ' + re.sub("^\s*(.-)\s*$", "%1", s).replace("\\n", "\n")

                examples.append(data.Example.fromlist([text, label], fields))

        super(AGNewsCorpus, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, root='.data',
               train='train.csv', test='test.csv', **kwargs):
        """Create dataset objects for splits of the AG news corpus dataset.
        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            root: Root dataset storage directory. Default is '.data'.
            train: The filename of the train data. Default: 'train.csv'.
            test: The filename of the test data, or None to not load the test
                set. Default: 'test.csv'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        return super(AGNewsCorpus, cls).splits(
            root=root, text_field=text_field, label_field=label_field,
            train=train, validation=None, test=test, **kwargs)

    @classmethod
    def iters(cls, batch_size=32, device=0, root='.data', vectors=None,
              **kwargs):
        """Create iterator objects for splits of the AG news corpus dataset.
        Arguments:
            batch_size: Batch_size
            device: Device to create batches on. Use - 1 for CPU and None for
                the currently active GPU device.
            root: The root directory that contains the trec dataset subdirectory
            vectors: one of the available pretrained vectors or a list with each
                element one of the available pretrained vectors (see Vocab.load_vectors)
            Remaining keyword arguments: Passed to the splits method.
        """
        TEXT = data.Field()
        LABEL = data.Field(sequential=False, unk_token=None)

        train, test = cls.splits(TEXT, LABEL, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)
        LABEL.build_vocab(train)

        return data.BucketIterator.splits(
            (train, test), batch_size=batch_size, device=device)
