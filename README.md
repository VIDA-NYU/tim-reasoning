# TIM Reasoning

Repository for the reasoning module of TIM (A Transparent, Interpretable, and Multimodal Personal Assistant).

This package works with Python 3.8 in Linux and Mac.

## Installation

1. Install the reasoning module from Github repository:
```
$ pip install git+https://github.com/VIDA-NYU/tim-reasoning.git
```

2. Download extra files:
```
$ python -m spacy download en_core_web_lg
$ git clone https://huggingface.co/edwardjross/xlm-roberta-base-finetuned-recipe-all
$ curl -L https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz  -o structured-prediction-srl-bert.2020.12.15.tar.gz
```

3. Run any example from the directory `examples`:
```
$ jupyter notebook
```