import random
import os

from tabulate import tabulate
from torchtext.data import Dataset

from argparse import ArgumentParser


class TableLogger(object):
    def __init__(self, headers):
        self.headers = headers
        self.header_logged = False
        self.tabular_data = []
        self.tablefmt = 'simple'
        
    def _align_entries(self, entries):
        dct = dict(entries)
        aligned = [dct.get(h, None) for h in self.headers]
        return aligned
        
    def log(self, *entries):
        tabular_data = self._align_entries(entries)
        self.tabular_data.append(tabular_data)
        table = tabulate(self.tabular_data, headers=self.headers, tablefmt=self.tablefmt)
        if self.header_logged:
            table = table.rsplit('\n', 2)[-1]
        else:
            self.header_logged = True
        print(table)


def get_args():
    parser = ArgumentParser(description='PyTorch StarSpace')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--d_embed', type=int, default=100)
    parser.add_argument('--max_vocab_size', type=int, default=100000)
    parser.add_argument('--n_negative', type=int, default=5)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='results')
    #parser.add_argument('--resume_snapshot', type=str, default='')
    args = parser.parse_args()
    return args

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
    
    train_dataset, val_dataset = Dataset(train_examples, dataset.fields), Dataset(val_examples, dataset.fields)
    
    train_dataset.sort_key = dataset.sort_key
    val_dataset.sort_key = dataset.sort_key
    
    return train_dataset, val_dataset
