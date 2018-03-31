import torch

from torchtext import data

from .model import StarSpace, NegativeSampling, InnerProductSimilarity, MarginRankingLoss
from .ag_news_corpus import AGNewsCorpus
from .utils import get_args, train_validation_split


TORCHTEXT_DIR = os.path.join(str(Path.home()), '.torchtext')
        

def main():
    args = get_args()
    torch.cuda.set_device(args.gpu)

    TEXT = data.Field(batch_first=True, include_lengths=False,
                  tokenize='spacy')
    LABEL = data.Field(sequential=False, unk_token=None)

    train_validation, test = AGNewsCorpus.splits(
        root=os.path.join(TORCHTEXT_DIR, 'data'),
        text_field=TEXT, label_field=LABEL)

    train, validation = train_validation_split(train_validation, train_size=0.9)

    print('N samples train:', len(train_dataset))
    print('N samples validation:', len(val_dataset))
    print('N samples test:', len(test_dataset))

    TEXT.build_vocab(train_dataset, max_size=args.max_vocab_size)
    LABEL.build_vocab(train_dataset)

    print('Vocab size TEXT:', len(TEXT.vocab))
    print('Vocab size LABEL:', len(LABEL.vocab))

    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
            (train, validation, test), batch_size=args.batch_size, device=args.gpu)

    model = StarSpace(
        dim=args.d_embed,
        n_input=len(TEXT.vocab),
        n_output=len(LABEL.vocab),
        similarity=InnerProductSimilarity(),
        max_norm=20,
        aggregate=torch.sum)

    neg_sampling = NegativeSampling(n_output=len(LABEL.vocab), n_negative=args.n_negative)

    criterion = MarginRankingLoss(margin=1., aggregate=torch.mean)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    iterations = 0
    start = time.time()
    best_dev_acc = -1
    train_iter.repeat = False
    header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Val/Loss     Accuracy  Val/Accuracy'
    dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:12.4f},{:12.4f}'.split(','))
    log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
    makedirs(args.save_path)
    print(header)

    for epoch in range(args.epochs):

        n_correct, n_total = 0, 0
        for batch_idx, batch in enumerate(train_iter):

            model.train(); opt.zero_grad()

            iterations += 1

            # extractor function for different tasks
            # get similarity for positive entity pairs
            input_repr, pos_output_repr = model(batch.text, batch.label)  # B x dim, B x dim

            positive_similarity = model.similarity(input_repr, pos_output_repr).squeeze(1)  # B x 1

            # get similarity for negative entity pairs
            batch_size = input.size(0)
            n_samples = batch_size * args.n_negative

            input_repr, neg_output_repr = model(batch.text, neg_sampling.sample(n_samples))  # B x dim, (B * n_negative) x dim
            neg_output_repr = neg_output_repr.view(batch_size, args.n_negative, -1)  # B x n_negative x dim

            negative_similarity = model.similarity(input_repr, neg_output_repr).squeeze(1)  # B x n_negative

            # calculate accuracy of predictions in the current batch
            n_correct += (torch.max(answer, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
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
            if iterations % args.dev_every == 0:
                model.eval()

                # calculate accuracy on validation set
                n_dev_correct = 0
                for dev_batch_idx, dev_batch in enumerate(dev_iter):
                    batch_size = dev_batch.batch_size
                    output = torch.autograd.Variable(torch.range(0, len(LABEL.vocab) - 1).long().expand(batch_size, -1)) # B x n_output
                    input_repr, output_repr = model(dev_batch.text, output)  # B x dim, B x n_output x dim
                    similarity = model.similarity(input_repr, output_repr).squeeze(1)  # B x n_output
                    n_dev_correct += (torch.max(similarity, dim=-1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                dev_acc = 100. * n_dev_correct / len(dev)

                print(dev_log_template.format(time.time()-start,
                    epoch, iterations, 1+batch_idx, len(train_iter),
                    100. * (1+batch_idx) / len(train_iter), loss.data[0], train_acc, dev_acc))

                # update best valiation set accuracy
                if dev_acc > best_dev_acc:

                    # found a model with better validation set accuracy

                    best_dev_acc = dev_acc
                    snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
                    snapshot_path = snapshot_prefix + '_devacc_{}__iter_{}_model.pt'.format(dev_acc, iterations)

                    # save model, delete previous 'best_snapshot' files
                    torch.save(model, snapshot_path)
                    for f in glob.glob(snapshot_prefix + '*'):
                        if f != snapshot_path:
                            os.remove(f)

            elif iterations % args.log_every == 0:

                # print progress message
                print(log_template.format(time.time()-start,
                    epoch, iterations, 1+batch_idx, len(train_iter),
                    100. * (1+batch_idx) / len(train_iter), loss.data[0], ' '*8, n_correct/n_total*100, ' '*12))

    model.eval()
    # calculate accuracy on test set
    n_test_correct = 0
    for test_batch_idx, test_batch in enumerate(test_iter):
        batch_size = test_batch.batch_size
        output = torch.autograd.Variable(torch.range(0, len(LABEL.vocab) - 1).long().expand(batch_size, -1)) # B x n_output
        input_repr, output_repr = model(test_batch.text, output)  # B x dim, B x n_output x dim
        similarity = model.similarity(input_repr, output_repr).squeeze(1)  # B x n_output
        n_test_correct += (torch.max(similarity, dim=-1)[1].view(test_batch.label.size()).data == test_batch.label.data).sum()
        #test_loss = criterion(answer, dev_batch.label)
    test_acc = 100. * n_test_correct / len(test)
    print('Accuracy on test set: {:12.4f}'.format(test_acc))

if __name__ == '__main__':
    main()
