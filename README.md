# bart-ChineseGEC

# Requirements
- transformers==4.15.0
- tokeniziers==0.10.3
- torch==1.7.0, 1.8.0, 1.8.1
- jieba
- rouge
- tqdm
- pandas

# How to use
## Download the data and the pre-trained model
The official NLPCC2018_GEC training and test data can be downloaded from [here](https://github.com/zhaoyyoo/NLPCC2018_GEC). In the official training data, one error sentences may have multiple corrections, we divide them into seperate parts. Since there are no official development data, we randomly extract 5,000 sentences from the training data as the development data. We also segment all sentences into characters. Here is our [preprocessed data](https://drive.google.com/file/d/1a9-DF90qY6heQXtNLaKzbVV8rWlVlRKQ/view?usp=sharing).

For pre-trained model, we use [t5-pegasus-base](https://huggingface.co/imxly/t5-pegasus/tree/main) and [bart-large-chinese](https://huggingface.co/fnlp/cpt-large/tree/main).
## Use fairseq command to turn the data into binary datasets
```
python preprocess.py  --user-dir ./user  \
--task bert_translation \
--srcdict $PATH_TO_PRE_TRAINED_MODEL/vocab.txt  \
-s src -t trg \
--bert-name $PATH_TO_PRE_TRAINED_MODEL \
--destdir $DATA_BIN_DIR  \
--trainpref $PATH_TO_DATA/train \
--validpref $PATH_TO_DATA/dev \
--testpref $PATH_TO_DATA/test
```
## Train the model
```
python train.py $DATA_BIN_DIR \
--seed 1 --user-dir ./user/ \
--optimizer adam --bert-name  $PATH_TO_PRETRAINED_MODEL \
--task bert_translation --batch-size 32 \
--arch bert_nmt --max-epoch 30 \
--save-dir  $SAVED_MODEL \
-s src -t trg --lr 0.00003
```
## Generate
```
python generate.py $DATA_BIN_DIR \
--task bert_translation  -s src -t trg  --user-dir ./user \
--path $SAVED_MODEL/checkpoint_best.pt --batch-size 32 \
--beam 12  > $OUTPUT_PATH/bert_encoder.txt
```
## Evaluate
Download the [PKUNLP word segmentation tools](http://59.108.48.37:9014/lcwm/pkunlp/downloads/libgrass-ui.tar.gz), and [m2scorer](https://github.com/nusnlp/m2scorer).
Use ``` fairseq_reformat.py``` to get the raw text output. 
```
python fairseq_reformat.py $OUTPUT_PATH/bert_encoder.txt
```
Delete the space and [UNK] tokens.
```
sed 's/<unk>//g' < $OUTPUT_PATH/bert_encoder.txt.reformat > bert_encoder.nounk.txt
sed 's/ //g' < bert_encoder.nounk.txt > bert_encoder.nospac.txt
```
Segment the sentence into words using [PKUNLP word segmentation tools](http://59.108.48.37:9014/lcwm/pkunlp/downloads/libgrass-ui.tar.gz).
```
cd libgrass-ui
python pkunlp_segment.py --corpus $PATH/bert_encoder.nospac.txt
```
Evaluate using [m2scorer](https://github.com/nusnlp/m2scorer). The gold edit file is in [official datasets](https://github.com/zhaoyyoo/NLPCC2018_GEC).
```
python $PATH/m2scorer.py $PATH/bert_encoder.nospac.txt.seg $PATH/gold/gold.01
```
