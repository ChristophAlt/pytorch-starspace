import random

from torchtext.data import Dataset

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='PyTorch StarSpace')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--d_embed', type=int, default=100)
    parser.add_argument('--max_vocab_size', type=int, default=100000)
    parser.add_argument('--n_negative', type=int, default=5)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dev_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='results')
    #parser.add_argument('--resume_snapshot', type=str, default='')
    args = parser.parse_args()
    return args

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

def evaluate_on_test(model, dataset, batch_size=128):
    model.eval()
    n_correct, n_total = 0, 0
    for batch_idx, batch in enumerate(
            BucketIterator(dataset, batch_size, train=False)):
        predictions = model.predict(batch)
        #predictions = output[Port.PREDICTION]
        accuracy(predictions, batch.label)

    print('Test accuracy:', 100. * n_correct / n_total)
