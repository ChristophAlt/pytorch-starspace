import random
import os, errno
import pickle

from torchtext import data
from datasets.ag_news_corpus import AGNewsCorpus


def makedirs(name):
    """Helper to call os.makedirs(), does not throw an error if directory already exists.
        Args:
            name (str): Directory name to be created.
    
    """

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise
    return


def train_validation_split(dataset, train_size=None, validation_size=None, shuffle=True):
    """Splits dataset into train and validation set.

    Args:
        dataset (Dataset): The dataset to be split.
        train_size (float): Fraction of dataset to be added to the train set, in range (0, 1).
        validation_size (float): Fraction of dataset treated as the validation set, in range (0, 1).
        Mutual exclusive with train_size.
        shuffle (bool, optional): If true, shuffle dataset before splitting.

    Returns:
        tuple: training dataset, validation dataset.
    
    """

    if train_size is None and validation_size is None:
        raise ValueError('Either train_size or validation_size must be given')

    examples = list(dataset.examples)
    if shuffle:
        random.shuffle(examples)

    train_size = train_size or (1. - validation_size)
    split_idx = int(train_size * len(examples))

    train_examples, val_examples = examples[:split_idx], examples[split_idx:]
    
    train_dataset = data.Dataset(train_examples, dataset.fields)
    val_dataset = data.Dataset(val_examples, dataset.fields)

    train_dataset.sort_key, val_dataset.sort_key = dataset.sort_key, dataset.sort_key
    
    return train_dataset, val_dataset


def create_batch_extractor(lhs_name, rhs_name):
    """Creates an extractor that can be used to extract two attributes from an object.
    
    Depending on the dataset, the two entities required for StarSpace may be added to the 
    torchtext Batch object under different attributes. This function can be used to return the  
    attributes specific to the respective dataset.    

    Args:
        lhs_name (str): The first entities attribute name.
        rhs_name (str): The second entities attribute name.

    Returns:
        tuple: content of the lhs_name attribute, content of the rhs_name attribute.

    """

    def func(batch):
        return getattr(batch, lhs_name), getattr(batch, rhs_name)
    
    return func


def create_fields(dataset_format):
    """Creates torchtext fields for the two entitites required in StarSpace, 
    according to the given dataset format.

    Args:
        dataset_format (str): The format of the dataset or a specific dataset.

    Returns:
        tuple: torchtext.Field for the first entity, torchtext.Field for the second entity.

    Raises:
        ValueError: If unknown dataset_format is passed.

    """
    dataset_format = dataset_format.lower()

    if dataset_format == 'ag_news':
        lhs_field = data.Field(batch_first=True, sequential=True, include_lengths=False, unk_token=None)
        rhs_field = data.Field(sequential=False, unk_token=None)
    else:
        raise ValueError("Dataset format '%s' not supported yet!" % dataset_format)

    return lhs_field, rhs_field


def get_dataset_and_extractor(path, dataset_format, lhs_field, rhs_field):
    """Given the path and format of a dataset, read the dataset and create 
    the corresponding batch extractor.

    Args:
        path (str):
        dataset_format (str): The format of the dataset or a specific dataset.
        lhs_field (Field): The field used for processing the first entity.
        rhs_field (Field): The field used for processing the second entity.

    Returns:
        tuple: Created dataset, batch extractor for both entities.

    Raises:
        ValueError: If unknown dataset_format is passed.

    """

    dataset_format = dataset_format.lower()

    if dataset_format == 'ag_news':
        dataset = AGNewsCorpus(path, text_field=lhs_field, label_field=rhs_field)
        extractor = create_batch_extractor(lhs_name='text', rhs_name='label')
    else:
        raise ValueError("Dataset format '%s' not supported yet!" % dataset_format)

    return dataset, extractor

def save_vocab(field, path):
    """Saves the vocabulary of a torchtext Field to the given path.

    Args:
        field (Field): The field which vocabulary to be saved.
        path (str): The path where to save the vocabulary.

    """
    
    with open(path, 'wb') as f:
        pickle.dump(field.vocab, f)

def load_vocab(path):
    """Loads a vocabulary for a torchtext Field from the given path.

    Args:
        path (str): The path where to load the vocabulary from.

    Returns:
        Vocabulary: The vocabulary.

    """

    with open(path, 'rb') as f:
        return pickle.load(f)
