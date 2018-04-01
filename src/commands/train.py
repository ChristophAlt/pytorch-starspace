import os
import time
import glob
import torch
import click

from torchtext import data

from src.model import StarSpace, InnerProductSimilarity, MarginRankingLoss
from src.sampling import NegativeSampling
from src.logging import TableLogger
from src.utils import train_validation_split, makedirs, create_fields, \
    get_dataset_and_extractor, save_vocab


@click.command()
@click.option('--train-file', type=click.Path(exists=True), required=True)
@click.option('--dataset_format', type=click.Choice(['ag_news']), default=None)
@click.option('--validation_split', type=float, default=0.1)
@click.option('--epochs', type=int, default=10)
@click.option('--batch_size', type=int, default=64)
@click.option('--d_embed', type=int, default=100)
@click.option('--n_negative', type=int, default=5)
@click.option('--log_every', type=int, default=50)
@click.option('--lr', type=float, default=1e-3)
@click.option('--val_every', type=int, default=1000)
@click.option('--save_every', type=int, default=1000)
@click.option('--gpu', type=int, default=0)
@click.option('--save_path', type=str, default='results')
def train(train_file, dataset_format, epochs, batch_size, d_embed, n_negative, log_every, lr, val_every, save_every, gpu, save_path, validation_split):
    #print('Configuration:')

    torch.cuda.device(gpu)

    lhs_field, rhs_field = create_fields(dataset_format)
    train_dataset, batch_extractor = get_dataset_and_extractor(train_file, dataset_format, lhs_field, rhs_field)

    train, validation = train_validation_split(train_dataset, validation_size=validation_split)

    print('Num entity pairs train:', len(train))
    print('Num entity pairs validation:', len(validation))

    lhs_field.build_vocab(train)
    rhs_field.build_vocab(train)

    n_lhs = len(lhs_field.vocab)
    n_rhs = len(rhs_field.vocab)

    print('Num LHS features:', n_lhs)
    print('Num RHS features:', n_rhs)

    train_iter, val_iter = data.BucketIterator.splits(
        (train, validation), batch_size=batch_size, device=gpu)

    # TODO: implement loading from snapshot
    model = StarSpace(
        d_embed=d_embed,
        n_input=n_lhs,
        n_output=n_rhs,
        similarity=InnerProductSimilarity(),
        max_norm=20,
        aggregate=torch.sum)

    model.cuda()

    neg_sampling = NegativeSampling(n_output=n_rhs, n_negative=n_negative)

    criterion = MarginRankingLoss(margin=1., aggregate=torch.mean)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    logger = TableLogger(['time', 'epoch', 'iterations', 'loss', 'accuracy', 'val_accuracy'])
    
    makedirs(save_path)
    
    iterations = 0
    start = time.time()
    train_iter.repeat = False
    best_val_acc = -1

    for epoch in range(epochs):
        n_correct, n_total = 0, 0
        for batch in train_iter:
            
            model.train(); opt.zero_grad()

            iterations += 1

            # TODO: add input/output extractor function for different tasks
            # TODO: add prediction candidate generator function for different tasks
            # TODO: add correctness/accuracy function for different tasks
            # TODO: clean up accuracy computation
            
            # extract entity pairs from batch
            lhs, rhs = batch_extractor(batch)

            # get similarity for positive entity pairs
            lhs_repr, pos_rhs_repr = model(lhs, rhs)  # B x dim, B x dim
            positive_similarity = model.similarity(lhs_repr, pos_rhs_repr).squeeze(1)  # B x 1

            # get similarity for negative entity pairs
            n_samples = batch.batch_size * n_negative
            neg_rhs = neg_sampling.sample(n_samples)
            if lhs.is_cuda:
                neg_rhs = neg_rhs.cuda()
            _, neg_rhs_repr = model(output=neg_rhs)  # (B * n_negative) x dim
            neg_rhs_repr = neg_rhs_repr.view(batch.batch_size, n_negative, -1)  # B x n_negative x dim
            negative_similarity = model.similarity(lhs_repr, neg_rhs_repr).squeeze(1)  # B x n_negative

            # calculate accuracy of predictions in the current batch
            candidate_rhs = torch.autograd.Variable(torch.arange(0, n_rhs).long().expand(batch.batch_size, -1)) # B x n_output
            if lhs.is_cuda:
                candidate_rhs = candidate_rhs.cuda()
            _, candidate_rhs_repr = model(output=candidate_rhs.view(batch.batch_size * n_rhs))  # B x dim, (B * n_output) x dim
            candidate_rhs_repr = candidate_rhs_repr.view(batch.batch_size, n_rhs, -1)  # B x n_output x dim
            similarity = model.similarity(lhs_repr, candidate_rhs_repr).squeeze(1)  # B x n_output
            n_correct += (torch.max(similarity, dim=-1)[1].view(rhs.size()).data == rhs.data).sum()
            n_total += batch.batch_size
            train_acc = 100. * n_correct/n_total

            # calculate loss
            loss = criterion(positive_similarity, negative_similarity)

            loss.backward(); opt.step()

            # checkpoint model periodically
            if iterations % save_every == 0:
                snapshot_prefix = os.path.join(save_path, 'snapshot')
                snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc, loss.data[0], iterations)
                torch.save(model, snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

            # evaluate performance on validation set
            if iterations % val_every == 0:
                model.eval()

                # calculate accuracy on validation set
                n_val_correct = 0
                for val_batch_idx, val_batch in enumerate(val_iter):
                    val_lhs, val_rhs = batch_extractor(val_batch)

                    val_candidate_rhs = torch.autograd.Variable(torch.arange(0, n_rhs).long().expand(val_batch.batch_size, -1)) # B x n_output
                    if val_lhs.is_cuda:
                        val_candidate_rhs = val_candidate_rhs.cuda()
                    val_lhs_repr, val_candidate_rhs_repr = model(val_lhs, val_candidate_rhs.view(val_batch.batch_size * n_rhs))  # B x dim, (B * n_output) x dim
                    val_candidate_rhs_repr = val_candidate_rhs_repr.view(val_batch.batch_size, n_rhs, -1)  # B x n_output x dim
                    similarity = model.similarity(val_lhs_repr, val_candidate_rhs_repr).squeeze(1)  # B x n_output
                    n_val_correct += (torch.max(similarity, dim=-1)[1].view(val_rhs.size()).data == val_rhs.data).sum()
                val_acc = 100. * n_val_correct / len(validation)

                # log progress, including validation metrics
                logger.log(time=time.time()-start, epoch=epoch, iterations=iterations,
                           loss=loss.data[0], accuracy=(100. * n_correct/n_total),
                           val_accuracy=val_acc)

                # found a better model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc

                    snapshot_prefix = os.path.join(save_path, 'best_snapshot')
                    path_prefix = snapshot_prefix + '_valacc_{}__iter_{}_'.format(val_acc, iterations)
                    snapshot_path = path_prefix + 'model.pt'

                    # save model, delete previous 'best_snapshot' files
                    torch.save(model, snapshot_path)
                    for f in glob.glob(snapshot_prefix + '*'):
                        if f != snapshot_path:
                            os.remove(f)

                    # save vocabulary for both entity fields
                    save_vocab(lhs_field, path_prefix + 'lhs_vocab.pkl')
                    save_vocab(rhs_field, path_prefix + 'rhs_vocab.pkl')

            elif iterations % log_every == 0:
                # log training progress
                logger.log(time=time.time()-start, epoch=epoch, iterations=iterations,
                           loss=loss.data[0], accuracy=(100. * n_correct/n_total))
