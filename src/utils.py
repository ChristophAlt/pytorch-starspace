import random
import os
import pickle

from torchtext import data
from datasets.ag_news_corpus import AGNewsCorpus


def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""

    import os, errno

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


def train_validation_split(dataset, train_size, shuffle=True):
    examples = list(dataset.examples)
    
    if shuffle:
        random.shuffle(examples)
    
    split_idx = int(train_size * len(examples))
    train_examples, val_examples = examples[:split_idx], examples[split_idx:]
    
    train_dataset, val_dataset = data.Dataset(train_examples, dataset.fields), data.Dataset(val_examples, dataset.fields)
    
    train_dataset.sort_key = dataset.sort_key
    val_dataset.sort_key = dataset.sort_key
    
    return train_dataset, val_dataset


def get_batch_attribs(lhs_attr_name, rhs_attr_name):
    def func(batch):
        return getattr(batch, lhs_attr_name), getattr(batch, rhs_attr_name)
    return func


def get_fields(dataset_format):
    if dataset_format == 'ag_news':
        lhs_field = data.Field(batch_first=True, sequential=True, include_lengths=False, unk_token=None)
        rhs_field = data.Field(sequential=False, unk_token=None)
    else:
        raise NotImplementedError("Dataset format '%s' not supported yet!" % dataset_format)

    return lhs_field, rhs_field


def get_dataset_extractor(path, dataset_format, lhs_field, rhs_field):
    if dataset_format == 'ag_news':
        dataset = AGNewsCorpus(path, text_field=lhs_field, label_field=rhs_field)
        extractor_func = get_batch_attribs('text', 'label')
    else:
        raise NotImplementedError("Dataset format '%s' not supported yet!" % dataset_format)

    return dataset, extractor_func

def serialize_field_vocabs(model_path, lhs_field, rhs_field):
    idx = model_path.find('model.pt')
    path = model_path[:idx]

    with open(path + '_lhs_vocab.pkl', 'wb') as f:
        pickle.dump(lhs_field.vocab, f)

    with open(path + '_rhs_vocab.pkl', 'wb') as f:
        pickle.dump(rhs_field.vocab, f)

def deserialize_field_vocabs(model_path):
    idx = model_path.find('model.pt')
    path = model_path[:idx]

    with open(path + '_lhs_vocab.pkl', 'rb') as f:
        lhs_vocab = pickle.load(f)

    with open(path + '_rhs_vocab.pkl', 'rb') as f:
        rhs_vocab = pickle.load(f)

    return lhs_vocab, rhs_vocab

