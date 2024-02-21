# hourglass-llama
A clean, modern, and generalized implementation of causal hierarchical transformers (wip)

## Install
```sh
# for training/development
pip install -e '.[train]'

# for inference
pip install .
```

## Usage (inference)
```py
from hourglass_llama import HourglassLlama

# load pretrained checkpoint
model = HourglassLlama.from_pretrained("some_hf_checkpoint")
```

## Usage (training)

Define a config file in `configs/`, called `demo_run` in this case:
```py
from hourglass_llama import (
    HourglassLlama,
    HourglassLlamaConfig,
    EnWiki8Dataset,
    Trainer,
    TrainConfig
)

model = HourglassLlama(HourglassLlama(xxx))
dataset = EnWiki8Dataset(xxx)

trainer = Trainer(
    model=model,
    dataset=dataset,    
    train_config=TrainConfig(xxx)
)
Run the training
```sh
python train.py demo_run
```
