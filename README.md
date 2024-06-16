# Jointly Pre-trained Template and Verbalizer

This project is based on OpenPrompt (https://github.com/thunlp/OpenPrompt). Our main codes locate at `openprompt/prompts/jptv_verbalizer.py`. 

## Prepare data

```
cd datasets/
bash download_text_classification.sh
```

## Pre-training

We provide the pre-trained weights in the `pretrained_weights` dir. If you want to pre-train template and verbalizer on your own, run the following command:

```
python cli.py --config_yaml experiment/pretrain_jptv.yaml
```

Please modify `jptv_verbalizer/mid_dim` in the yaml file to modify the dimension of masked semantic bases.

## Fine-tuning

```
python cli.py --config_yaml experiment/finetune_jptv_256.yaml
python cli.py --config_yaml experiment/finetune_jptv_64.yaml
```

The dataset in the yaml file can be `agnews`, `dbpedia`, `yahoo_answer_topics` or `guardiantopics`.

