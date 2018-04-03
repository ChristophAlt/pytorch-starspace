# PyTorch implementation of StarSpace

Based on the approach described in [StarSpace: Embed All The Things! by 
Ledell Wu, Adam Fisch, Sumit Chopra, Keith Adams, Antoine Bordes and Jason Weston](https://arxiv.org/abs/1709.03856)

**NOTE:** The current version is work in progress and doesn't yet match the functionality of the original implementation.

The C++ version maintained and developed by the authors can be found [here](https://github.com/facebookresearch/StarSpace).


## Installation
Tested on Pyton 3.6.1 and PyTorch 0.3.1


- Install PyTorch via [conda or pip](http://pytorch.org)
- Install dependencies  
`$ pip install -r requirements.txt`


## Usage
The commandline interface resembles the one used by the C++ version.

To train and validate a StarSpace model on the [AG news corpus](https://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html),
run the following commands:

### Train a model
```
$ ./starspace train \
    --train-file "<DATSET_PATH>/train.csv" \
    --model-path "<MODELDIR>/" \
    --file-format ag_news \
    --d-embed=10 \
    --lr=0.01 \
    --epochs=5
```

### Validate the trained model

```
$ ./starspace test \
    --test-file "<DATSET_PATH>/test.csv" \
    --model-path "<MODELDIR>/" \
    --file-format ag_news
```
