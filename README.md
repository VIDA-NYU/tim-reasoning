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

For using the new `SessionManager`:

```py
from tim_reasoning import SessionManager

sm = SessionManager(
    unique_objects_file="data/step_goals/unique_objects.json",
    data_folder="data/step_goals/",
    patience=1,
)

message = [
    {
        "pos": [-1.2149151724097291, -0.4343880843796524, -0.6208099189217009],
        "xyxyn": [0.2, 0.2, 0.4, 0.4],
        "label": "tortilla",
        "id": 1,
        "status": "tracked",
        "last_seen": 133344374307946261,
        "state": {
            "in-package": 0.02,
            "plain": 0.45,
            "peanut-butter[initial]": 0.25,
            "peanut-butter[full]": 0.2,
            "pb+jelly[partial]": 0.02,
            "pb+jelly[full]": 0.02,
            "nutella[partial]": 0.02,
            "nutella[full]": 0.02,
            "nutella+banana[partial]": 0.02,
            "nutella+banana[full]": 0.02,
            "nutella+banana+cinnamon": 0.02,
            "folding": 0.02,
            "folded": 0.02,
            "cut": 0.02,
            "rolling": 0.02,
            "rolled": 0.02,
            "rolled+toothpicks": 0.02,
            "ends-cut": 0.02,
            "sliced[partial]": 0.02,
            "sliced[full]": 0.02,
            "on-plate[partial]": 0.02,
            "on-plate[full]": 0.02,
        },
        "hand_object_interaction": 0.89,
    },
    {
        "pos": [-1.2149151724097291, -0.4343880843796524, -0.6208099189217009],
        "xyxyn": [0.2, 0.2, 0.4, 0.4],
        "label": "spoon",
        "id": 10,
        "status": "tracked",
        "last_seen": 133344374307946261,
        "unexpected_object": True,
        "hand_object_interaction": 0.8,
    },
    {
        "pos": [-1.2149151724097291, -0.4343880843796524, -0.6208099189217009],
        "xyxyn": [0.0, 0.1, 0.1, 0.2],
        "label": "peanut butter jar",
        "id": 2,
        "status": "tracked",
        "last_seen": 133344374307946261,
        "hand_object_interaction": 0.0,
    },
]

sm.handle_message(message=message)
```