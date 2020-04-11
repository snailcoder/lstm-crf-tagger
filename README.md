# lstm-crf-tagger
This is an implementation of LSTM-CRF tagger proposed in the paper [Neural Architectures for Named Entity Recognition]
(https://www.aclweb.org/anthology/N16-1030/).
The model is trained and tested on SIGHAN2005 bake-off for word segmentation. 

## Setup
Python3, Numpy, Tensorflow 2.0

## How to train the model
You can directly use the prepared corpus: ./data/pku_training.txt for training your own tagger. Here, pku_training.txt is created by running process_corpus.py on pku_training.utf8 provided by SIGHAN2005 bake-off.

If you have downloaded the SIGHAN2005 bake-off and want to train your model on other corpus(e.g. msr_training.utf8, cityu_training.utf8) contained by the bake-off, you need to run process_corpus.py to get the training-read file. The corpus provided by SIGHAN2005 is in an original format:
```
中共中央  总书记  、  国家  主席  江  泽民  
（  一九九七年  十二月  三十一日  ）
```
You can run process_corpus.py like this:
```bash
python process_corpus /path/of/pku_training.utf8 SIGHAN2005 /path/of/processed_pku_training.txt
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
