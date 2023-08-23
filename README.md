# TIM Reasoning

Repository for the reasoning module of TIM (A Transparent, Interpretable, and Multimodal Personal Assistant).

This package works with Python 3.8 in Linux and Mac.

## Installation

1. Install the reasoning module from Github repository:
```
$ pip install git+https://github.com/VIDA-NYU/tim-reasoning.git
```

2. Download models:

- Download the model for the recipe tagger [here](https://drive.google.com/file/d/1aYSlngadawRTKuIkd1FtMvrMBfenqPLH/view?usp=sharing) 
(then, uncompress it).
- Download the model for the BERT classifier [here](https://drive.google.com/file/d/1RsXbLrIubPTAbgP3NEAB73LMcAKDV3oE/view?usp=sharing) 
(then, uncompress it).

3. Download extra files:
```
$ python -m spacy download en_core_web_lg
$ python -c "import nltk;nltk.download('punkt')"
```

4. Run any example from the `examples` directory:
```
$ jupyter notebook
```

## Usage

1. For using the `Pddl2GraphConverter`:

```py
from tim_reasoning import Pddl2GraphConverter
# create the converter object
converter = Pddl2GraphConverter()
# convert the recipe pddl to graph
# provide the folder containing domain.pddl and steps
# make sure to name the step pddl files as step1.pddl, step2.pddl ...
graph = converter.convert(pddl_folder='./data/')
graph.print_dependencies()
```