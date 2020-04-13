# lstm-crf-tagger
This is an implementation of LSTM-CRF tagger proposed in the paper [Neural Architectures for Named Entity Recognition](https://www.aclweb.org/anthology/N16-1030/).
The model is trained and tested on SIGHAN2005 bake-off for word segmentation. 

## Setup
Python3, Numpy, Tensorflow 2.0

## Preparation
You can directly use the prepared corpus: ./data/pku_training.txt for training your own tagger. Here, pku_training.txt is created by running process_corpus.py on pku_training.utf8 provided by SIGHAN2005 bake-off.

If you have downloaded the SIGHAN2005 bake-off and want to train your model on other corpus(e.g. msr_training.utf8, cityu_training.utf8) included in the bake-off, you need to run process_corpus.py to get the training-read file. The corpus provided by SIGHAN2005 is in an original format:
```
中共中央  总书记  、  国家  主席  江  泽民  
（  一九九七年  十二月  三十一日  ）
```
You can run process_corpus.py like this:
```bash
python process_corpus.py /path/of/pku_training.utf8 SIGHAN2005 /path/of/pku_training_ready.txt
```
Now you've converted the original file to the training-ready file in this format(S: single, B: beginning, M: middle, E: end):
```
中      B
共      M
中      M
央      E
总      B
书      M
记      E
、      S
国      B
家      E
主      B
席      E
江      S
泽      B
民      E

（      S
一      B
九      M
九      M
七      M
年      E
十      B
二      M
月      E
三      B
十      M
一      M
日      E
）      S

...
```
## Training
1. Create TFRecord dataset.
```bash
python build_tfrecords.py /path/of/pku_training_ready.txt /path/of/dataset
```
This script converts your training-ready file to TFRecord format, shards them and write them to the directory: /path/of/dataset. In the meanwhile, it generates the vocabulary file: vocab.txt based on the training corpus.

2. Train the model.
```bash
python train_model.py '/path/of/dataset/train-?????-of-?????' '/path/of/dataset/valid-?????-of-?????' ./corpus/people_vec.txt /path/of/dataset/vocab.txt /path/to/save/checkpoints
```
Here, you can use the Chinese character vectors: /corpus/people_vec.txt for creating the embedding layer. These vectors are obtained by applying word2vec on the People's Daily dataset(199801-199806, 2014):
```bash
./word2vec -train /path/of/peoplesdaily.txt -output people_vec.txt -size 100 -window 5 -sample 1e-5 -negative 5 -hs 0 -binary 0 -cbow 0 -iter 5
```
## Experiments
I use **almost** the same hyperparameters (see configuration.py) with the training setup reported in [Neural Architectures for Named Entity Recognition](https://www.aclweb.org/anthology/N16-1030/), except I train the model only 30 epochs. In addition, I use L2-regularization with a lambda of 0.0001 to prevent the model from overfitting.

Notice the paper said "To prevent the learner from depending too heavily on one representation class, dropout is used." However, there's only one source representations in my implementation, so the dropout over the input embedding layer is not used by default(emb_dropout = 0). In fact, the macro F1 score could only reach around 0.79 if I set emb_dropout to be 0.5. 

The pre-trained embedding could be fine-tuned or not during training. However, fine-tuning Chinese character embeddings via backpropagation outperform keeping them static during training a large margin, as shown below:

| Method    | Macro F1|
| :-:       |:-:      |
|static     |81.34    |
|non-static |92.26    |

Furthermore, using static embedding throughout training leads overfitting around epoch 12, while overfitting is not ovserved around epoch 50 for non-static embedding.
