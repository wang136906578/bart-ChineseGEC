# bart-ChineseGEC
Finetune t5 and bart on Chinese Grammatical Error Correction data.
# Requirements
- transformers==4.4.3
- tokeniziers==0.10.3
- torch==1.8.1
- jieba
- tqdm
- pandas
# How to use
## Download the data and the pre-trained model
The official NLPCC2018_GEC training and test data can be downloaded from [here](https://github.com/zhaoyyoo/NLPCC2018_GEC). In the official training data, one error sentences may have multiple corrections, we divide them into seperate parts. Since there are no official development data, we randomly extract 5,000 sentences from the training data as the development data. We also segment all sentences into characters. Here is our [preprocessed data](https://drive.google.com/file/d/1a9-DF90qY6heQXtNLaKzbVV8rWlVlRKQ/view?usp=sharing).

For pre-trained model, we use [t5-pegasus-base](https://huggingface.co/imxly/t5-pegasus/tree/main) and [bart-large-chinese](https://huggingface.co/fnlp/cpt-large/tree/main).
## Finetune the model
```
python finetune.py --train_data data/train.tsv --dev_data data/dev.tsv \
--batch_size 32 --lr 0.00001 --seed 3 --save_dir $PATH_TO_SAVED_MODEL --num_epoch 20
```
## Generate
```
python predict.py --pretrain_model $PATH_TO_PRETRAINED_MODEL \
--model $PATH_TO_SAVED_MODEL --test_data data/test.tsv \
--result_file ./output.txt
```
## Evaluate
Download the [PKUNLP word segmentation tools](https://drive.google.com/file/d/17OKIqbS74_GATm65Yg-JowqXZsVk84-W/view?usp=sharing) (since the original url has been expired, we provide our saved one on Google drive for convenience), and [m2scorer](https://github.com/nusnlp/m2scorer).
```
cut -f 1 < output.txt > output.cut.txt
cd libgrass-ui
python pkunlp_segment.py --corpus output.cut.txt
python $PATH/m2scorer.py $PATH/output.cut.txt.seg $PATH/gold/gold.01
```
