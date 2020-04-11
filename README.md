# lstm-crf-tagger
This is an implementation of LSTM-CRF tagger proposed in the paper [Neural Architectures for Named Entity Recognition]
(https://www.aclweb.org/anthology/N16-1030/).
The model is trained and tested on SIGHAN2005 bake-off for word segmentation. 

## Setup
Python3, Numpy, Tensorflow 2.0

## How to train the model
If you have downloaded the SIGHAN2005 bake-off(e.g. pku_training.utf8), you can run process_corpus.py to get the proper format for training:
```bash
python process_corpus /path/of/pku_training.utf8 
```
## Training
