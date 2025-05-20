# My SPBU diploma

python -m moshi.server --lora-weight=weights/finetune_4/checkpoints/checkpoint_001000/consolidated/lora.safetensors --config-path=weights/finetune_4/checkpoints/checkpoint_001000/consolidated/config.json --half
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 -m external.moshi-finetune.train configs/moshi_7B_finetune_6.yaml

This repository contains practical part of my SPBU diploma project: "Multimodal large language models with direct voice input and output"

Main goal of practical part is to apply MLLM with direct voice input and output to the task of chinese speech practice learning.

# Project structure
```
.
├── dataset      # dataset for finetuning
│   ├── audio
│   ├── text
│   ├── text_cleaned
│   └── voices
├── external      # external libraries
│   ├── moshi-finetune
│   └── poe-api-wrapper
├── LICENSE.md
├── notebooks
│   └── data      # notebooks for data processing
├── README.md
├── requirements.txt
├── scripts
│   └── data      # scripts for data processing and dataset creation
└── venv          # virtual environment
    ├── bin
    ├── include
    ├── lib
    ├── lib64 -> lib
    ├── pyvenv.cfg
    └── share
```

# Installation for development
1. clone the repository
```bash
git clone https://github.com/PE51K/spbu-diploma
```

2. create virtual environment
```bash
python3.12 -m venv venv
```

3. activate virtual environment
```bash
source venv/bin/activate
```

4. install requirements
```bash
pip install -r requirements.txt
```

5. install external libraries
```bash
pip install -e external/moshi-finetune
pip install -e external/poe-api-wrapper
```

# Model training
1. Prepare dataset
```bash
python scripts/data/create_sintetic_chinese_lessons_text_dataset.py
python scripts/data/tts.py --start ... --end ...
python scripts/data/create_annotations.py dataset/audio
python scripts/data/merge_conversation.py dataset/audio dataset/finetune
python scripts/data/prepare_moshi_jsonl.py dataset/finetune
```
1.1 In script args you can setup input/output dirs, parameters for tts, etc. Please refer to the code for more details.
1.2 It will take approximately 5 days to generate 10.000 pairs of audio/text files.

2. Prepare train config. Here is an example of config file:
```yaml
# data
data:
  eval_data: './dataset/finetune_1/finetune_data/val.jsonl'
  shuffle: true
  train_data: './dataset/finetune_1/finetune_data/train.jsonl'

# model
moshi_paths:
  hf_repo_id: "kyutai/moshiko-pytorch-bf16"

full_finetuning: false
lora:
  enable: true
  rank: 128
  scaling: 2.
  ft_embed: false

first_codebook_weight_multiplier: 100.
text_padding_weight: .5

# optim
duration_sec: 100
batch_size: 8
max_steps: 2000
gradient_checkpointing: true
optim:
  lr: 2e-6
  weight_decay: 0.1
  pct_start: 0.05

# other
seed: 0
log_freq: 1
eval_freq: 100
do_eval: true
do_ckpt: true
ckpt_freq: 100


save_adapters: true

run_dir: "./weights/finetune_1"

# wandb
wandb:
  project: "moshi-finetune"
  run_name: "finetune_1"
  key: ""
  offline: False
```
2.1 For more details please refer to the [https://github.com/kyutai-labs/moshi-finetune](https://github.com/kyutai-labs/moshi-finetune).

3. Train model. We will train LORA-adapter for Moshi model.
```bash
python -m external.moshi-finetune.train your_config.yaml
```
3.1 You should have GPU with at least 32GB of memory to train the 7b model.

# Run finetuned model
```bash
python -m moshi.server --lora-weight=weights/.../lora.safetensors --config-path=weights/.../config.json --half
```
* You should have GPU with at least 16GB of memory to run the model.
