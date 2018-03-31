import os
import time
import glob
import torch

from pathlib import Path
from torchtext import data

from model import StarSpace, NegativeSampling, InnerProductSimilarity, MarginRankingLoss
from ag_news_corpus import AGNewsCorpus
from utils import TableLogger, get_args, train_validation_split, makedirs


TORCHTEXT_DIR = os.path.join(str(Path.home()), '.torchtext')
        

def main():
    args = get_args()

    print('Configuration:')
    print(vars(args))

    torch.cuda.set_device(args.gpu)

    TEXT = data.Field(batch_first=True, include_lengths=False,
                  tokenize='spacy')
    LABEL = data.Field(sequential=False, unk_token=None)

    train_validation, test = AGNewsCorpus.splits(
        root=os.path.join(TORCHTEXT_DIR, 'data'),
        text_field=TEXT, label_field=LABEL)

    train, validation = train_validation_split(train_validation, train_size=0.9)

    print('N samples train:', len(train))
    print('N samples validation:', len(validation))
    print('N samples test:', len(test))

    TEXT.build_vocab(train, max_size=args.max_vocab_size)
    LABEL.build_vocab(train)

    print('Vocab size TEXT:', len(TEXT.vocab))
    print('Vocab size LABEL:', len(LABEL.vocab))

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
            (train, validation, test), batch_size=args.batch_size, device=args.gpu)

    model = StarSpace(
        d_embed=args.d_embed,
        n_input=len(TEXT.vocab),
        n_output=len(LABEL.vocab),
        similarity=InnerProductSimilarity(),
        max_norm=20,
        aggregate=torch.sum)

    # TODO: implement loading from snapshot
    model.cuda()

    neg_sampling = NegativeSampling(n_output=len(LABEL.vocab), n_negative=args.n_negative)

    criterion = MarginRankingLoss(margin=1., aggregate=torch.mean)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    logger = TableLogger(headers=['time', 'epoch', 'iterations', 'loss', 'val_loss', 'accuracy', 'val_accuracy'])
    
    makedirs(args.save_path)
    
    iterations = 0
    start = time.time()
    train_iter.repeat = False
    best_val_acc = -1

    for epoch in range(args.epochs):
        n_correct, n_total = 0, 0
        for batch_idx, batch in enumerate(train_iter):

            model.train(); opt.zero_grad()

            iterations += 1

            # TODO: add input/output extractor function for different tasks
            # TODO: add prediction candidate generator function for different tasks
            # TODO: add correctness/accuracy function for different tasks
            # TODO: clean up accuracy computation
            
            # get similarity for positive entity pairs
            input_repr, pos_output_repr = model(batch.text, batch.label)  # B x dim, B x dim
            positive_similarity = model.similarity(input_repr, pos_output_repr).squeeze(1)  # B x 1

            # get similarity for negative entity pairs
            n_samples = batch.batch_size * args.n_negative
            neg_output = neg_sampling.sample(n_samples)
            if batch.text.is_cuda:
                neg_output = neg_output.cuda()
            input_repr, neg_output_repr = model(batch.text, neg_output)  # B x dim, (B * n_negative) x dim
            neg_output_repr = neg_output_repr.view(batch.batch_size, args.n_negative, -1)  # B x n_negative x dim
            negative_similarity = model.similarity(input_repr, neg_output_repr).squeeze(1)  # B x n_negative

            # calculate accuracy of predictions in the current batch
            output = torch.autograd.Variable(torch.arange(0, len(LABEL.vocab)).long().expand(batch.batch_size, -1)) # B x n_output
            if batch.text.is_cuda:
                output = output.cuda()
            input_repr, output_repr = model(batch.text, output.view(batch.batch_size * len(LABEL.vocab)))  # B x dim, (B * n_output) x dim
            output_repr = output_repr.view(batch.batch_size, len(LABEL.vocab), -1)  # B x n_output x dim
            similarity = model.similarity(input_repr, output_repr).squeeze(1)  # B x n_output
            n_correct += (torch.max(similarity, dim=-1)[1].view(batch.label.size()).data == batch.label.data).sum()
            n_total += batch.batch_size
            train_acc = 100. * n_correct/n_total

            # calculate loss
            loss = criterion(positive_similarity, negative_similarity)

            loss.backward(); opt.step()

            # checkpoint model periodically
            if iterations % args.save_every == 0:
                snapshot_prefix = os.path.join(args.save_path, 'snapshot')
                snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc, loss.data[0], iterations)
                torch.save(model, snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

            # evaluate performance on validation set
            if iterations % args.val_every == 0:
                model.eval()

                # calculate accuracy on validation set
                n_val_correct = 0
                for val_batch_idx, val_batch in enumerate(val_iter):
                    output = torch.autograd.Variable(torch.arange(0, len(LABEL.vocab)).long().expand(val_batch.batch_size, -1)) # B x n_output
                    if val_batch.text.is_cuda:
                        output = output.cuda()
                    input_repr, output_repr = model(val_batch.text, output.view(val_batch.batch_size * len(LABEL.vocab)))  # B x dim, (B * n_output) x dim
                    output_repr = output_repr.view(val_batch.batch_size, len(LABEL.vocab), -1)  # B x n_output x dim
                    similarity = model.similarity(input_repr, output_repr).squeeze(1)  # B x n_output
                    n_val_correct += (torch.max(similarity, dim=-1)[1].view(val_batch.label.size()).data == val_batch.label.data).sum()
                val_acc = 100. * n_val_correct / len(validation)

                # log progress, including validation metrics
                logger.log(('time', time.time()-start), ('epoch', epoch), ('iterations', iterations),
                           ('loss', loss.data[0]), ('accuracy', 100. * n_correct/n_total),
                           ('val_accuracy', val_acc))

                # update best valiation set accuracy
                if val_acc > best_val_acc:
                    # found a model with better validation set accuracy

                    best_val_acc = val_acc
                    snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
                    snapshot_path = snapshot_prefix + '_valacc_{}__iter_{}_model.pt'.format(val_acc, iterations)

                    # save model, delete previous 'best_snapshot' files
                    torch.save(model, snapshot_path)
                    for f in glob.glob(snapshot_prefix + '*'):
                        if f != snapshot_path:
                            os.remove(f)

            elif iterations % args.log_every == 0:
                # log training progress
                logger.log(('time', time.time()-start), ('epoch', epoch), ('iterations', iterations),
                           ('loss', loss.data[0]), ('accuracy', 100. * n_correct/n_total))

    model.eval()
    # calculate accuracy on test set
    n_test_correct = 0
    for test_batch_idx, test_batch in enumerate(test_iter):
        output = torch.autograd.Variable(torch.arange(0, len(LABEL.vocab)).long().expand(test_batch.batch_size, -1)) # B x n_output
        if test_batch.text.is_cuda:
            output = output.cuda()
        input_repr, output_repr = model(test_batch.text, output.view(test_batch.batch_size * len(LABEL.vocab)))  # B x dim, (B * n_output) x dim
        output_repr = output_repr.view(test_batch.batch_size, len(LABEL.vocab), -1)  # B x n_output x dim
        similarity = model.similarity(input_repr, output_repr).squeeze(1)  # B x n_output
        n_test_correct += (torch.max(similarity, dim=-1)[1].view(test_batch.label.size()).data == test_batch.label.data).sum()
    test_acc = 100. * n_test_correct / len(test)
    print('Accuracy on test set: {:12.4f}'.format(test_acc))

if __name__ == '__main__':
    main()
