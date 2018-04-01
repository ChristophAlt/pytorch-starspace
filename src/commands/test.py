import os
import click
import torch

from torchtext import data

from src.utils import create_fields, load_vocab, get_dataset_and_extractor


@click.command()
@click.option('--test-file', type=click.Path(exists=True), required=True)
@click.option('--model-path', type=click.Path(exists=True), required=True)
@click.option('--dataset_format', type=click.Choice(['ag_news']), default=None)
@click.option('--gpu', type=int, default=0)
@click.option('--batch_size', type=int, default=64)
def test(model_path, dataset_format, test_file, gpu, batch_size):
    model_file = os.path.join(model_path, 'model.pt')
    lhs_vocab_file = os.path.join(model_path, 'lhs_vocab.pkl')
    rhs_vocab_file = os.path.join(model_path, 'rhs_vocab.pkl')

    lhs_field, rhs_field = create_fields(dataset_format)
    lhs_field.vocab = load_vocab(lhs_vocab_file)
    rhs_field.vocab = load_vocab(rhs_vocab_file)

    test, batch_extractor = get_dataset_and_extractor(test_file, dataset_format, lhs_field, rhs_field)

    test_iter = data.BucketIterator(test, batch_size=batch_size, device=gpu, train=False)

    n_rhs = len(rhs_field.vocab)

    model = torch.load(model_file, map_location=lambda storage, locatoin: storage.cuda(gpu))

    model.eval()
    # calculate accuracy on test set
    n_test_correct = 0
    for test_batch in test_iter:
        test_lhs, test_rhs = batch_extractor(test_batch)

        test_candidate_rhs = torch.autograd.Variable(torch.arange(0, n_rhs).long().expand(test_batch.batch_size, -1)) # B x n_output
        if test_lhs.is_cuda:
            test_candidate_rhs = test_candidate_rhs.cuda()
        test_lhs_repr, test_candidate_rhs_repr = model(test_lhs, test_candidate_rhs.view(test_batch.batch_size * n_rhs))  # B x dim, (B * n_output) x dim
        test_candidate_rhs_repr = test_candidate_rhs_repr.view(test_batch.batch_size, n_rhs, -1)  # B x n_output x dim
        similarity = model.similarity(test_lhs_repr, test_candidate_rhs_repr).squeeze(1)  # B x n_output
        n_test_correct += (torch.max(similarity, dim=-1)[1].view(test_rhs.size()).data == test_rhs.data).sum()
    test_acc = 100. * n_test_correct / len(test)
    print('Accuracy on test set: {:12.4f}'.format(test_acc))
