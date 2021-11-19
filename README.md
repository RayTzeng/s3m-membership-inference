# Introduction
Official implementation of "Membership Inference Attacks Against Self-supervised Speech Models" [**arXiv**](https://arxiv.org/abs/2111.05113). 

In this work, we demonstrate that existing self-supervised speech model such as HuBERT, wav2vec 2.0, CPC and TERA are vulnerable to membership inference attack (MIA) and thus could reveal sensitive informations related to the training data.  
# Requirements
1. **Python** >= 3.6
2. Install **sox** on your OS
3. Install [**s3prl**](https://github.com/s3prl/s3prl) on your OS

```sh
git clone https://github.com/s3prl/s3prl
cd s3prl
pip install -e ./
```

4. Install the specific fairseq

```sh
pip install fairseq@git+https://github.com//pytorch/fairseq.git@f2146bdc7abf293186de9449bfa2272775e39e1d#egg=fairseq
```
# Preprocessing
First, extract the self-supervised feature of utterances in each corpus according to your needs. 

Currently, only **LibriSpeech** is available.

```sh
BASE_PATH=/path/of/the/corpus
OUTPUT_PATH=/path/to/save/feature
MODEL=wav2vec2
SPLIT=train-clean-100 # you should extract train-clean-100, dev-clean, dev-other, test-clean, test-other

python preprocess_feature_LibriSpeech.py \
    --base_path $BATH_PATH \
    --output_path $OUTPUT_PATH \
    --model $MODEL \
    --split $SPLIT

```


# Speaker-level MIA
After extracting the features, you can apply the attack against the models using either **basic attack** and **improved attack**. 

Noted that you should run the basic attack to generate the .csv file with similarity scores before performing improved attack.

### Basic Attack

```sh
SEEN_BASE_PATH=/path/you/save/feature/of/seen/corpus
UNSEEN_BASE_PATH=/path/you/save/feature/of/unseen/corpus
OUTPUT_PATH=/path/to/output/results
MODEL=wav2vec2

python predefined-speaker-level-MIA.py \
    --seen_base_path $SEEN_BATH_PATH \
    --unseen_base_path $UNSEEN_BATH_PATH \
    --output_path $OUTPUT_PATH \
    --model $MODEL \

```

### Improved Attack

```sh

python train-speaker-level-similarity-model.py \
    --seen_base_path $UNSEEN_BATH_PATH \
    --output_path $OUTPUT_PATH \
    --model $MODEL \
    --speaker_list "${OUTPUT_PATH}/${MODEL}-customized-speaker-level-attack-similarity.csv"

python customized-speaker-level-MIA.py \
    --seen_base_path $SEEN_BATH_PATH \
    --unseen_base_path $UNSEEN_BATH_PATH \
    --output_path $OUTPUT_PATH \
    --model $MODEL \
    --similarity_model_path "${OUTPUT_PATH}/customized-speaker-similarity-model-${MODEL}.pt"

```

# Utterance-level MIA
The process for utterance-level MIA is similar to that of speaker-level:
### Basic Attack

```sh
SEEN_BASE_PATH=/path/you/save/feature/of/seen/corpus
UNSEEN_BASE_PATH=/path/you/save/feature/of/unseen/corpus
OUTPUT_PATH=/path/to/output/results
MODEL=wav2vec2

python predefined-utterance-level-MIA.py \
    --seen_base_path $SEEN_BATH_PATH \
    --unseen_base_path $UNSEEN_BATH_PATH \
    --output_path $OUTPUT_PATH \
    --model $MODEL \

```

### Improved Attack

```sh

python train-utterance-level-similarity-model.py \
    --seen_base_path $UNSEEN_BATH_PATH \
    --output_path $OUTPUT_PATH \
    --model $MODEL \
    --speaker_list "${OUTPUT_PATH}/${MODEL}-customized-utterance-level-attack-similarity.csv"

python customized-utterance-level-MIA.py \
    --seen_base_path $SEEN_BATH_PATH \
    --unseen_base_path $UNSEEN_BATH_PATH \
    --output_path $OUTPUT_PATH \
    --model $MODEL \
    --similarity_model_path "${OUTPUT_PATH}/customized-utterance-similarity-model-${MODEL}.pt"

```

# Citation
If you find our work useful, please cite:

```sh
@article{tseng2021membership,
  title={Membership Inference Attacks Against Self-supervised Speech Models},
  author={Tseng, Wei-Cheng and Kao, Wei-Tsung and Lee, Hung-yi},
  journal={arXiv preprint arXiv:2111.05113},
  year={2021}
}
```
