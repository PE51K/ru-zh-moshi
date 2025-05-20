# RU-ZH-MOSHI

This repository contains source code and resources for experimental finetuning of opesource S2S-LLM [Moshi](https://github.com/kyutai-labs/moshi) on Russian-Chinese speech practice task.

# Repository structure

- `configs/` - contains configuration files for training and evaluation of the model using moshi-finetune
- `dataset/` - contains the dataset used for training and evaluation
- `external/` - contains git submodules and external libraries used in the project
- `notebooks/` - contains Jupyter notebooks for data exploration and visualization
- `scripts/` - contains scripts for data preparation
- `weights/` - contains weights of finetuned models
- `.gitignore` - gitignore file
- `.gitmodules` - git submodules file
- `LICENSE` - license file
- `README.md` - this file
- `requirements.txt` - requirements file for the project

# Local setup

1. Clone the repository:
```bash
git clone https://github.com/PE51K/ru-zh-moshi
cd ru-zh-moshi
```

2. Activate the virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```
3. Install the requirements:
```bash
pip install -r requirements.txt
```

4. Initialize git submodules:
```bash
git submodule init
git submodule update
```

5. Install the external libraries:
```bash
pip install -e external/moshi-finetune
```

# Dataset creation

1. Create syntetic text data using `notebooks/data/new/generate_text_data.ipynb`. Resuts will be saved in `dataset/new/text_generation_results/` folder.
2. Consolidate generated text data using `scripts/data/new/process_generated_text_data.py`. Results will be saved in `dataset/new/converted_generated_data.jsonl` file.
3. Prepare sample voices in `dataset/new/voices/` folder and generate audio files using `scripts/data/new/tts.py` script. Results will be saved in `dataset/new/audio/` folder.
4. Allighn generated audion with generated text using `scripts/data/new/align_segments.py` script.
5. Consolidate generated audio segments by conversations using `scripts/data/new/merge_conversations.py` script.
6. Create train/val split using `scripts/data/new/prepare_moshi_jsonl.py` script.

You can download datased from [yandex disk](https://disk.yandex.ru/d/Opp6-V_lowHl7g)

# Training and validation

1. Intall [moshi-finetune](https://github.com/kyutai-labs/moshi-finetune) if you haven't done it yet:
```bash
pip install -e external/moshi-finetune
```
2. Prepare config file for training folowing README.md from `external/moshi-finetune` repository. Example config file is provided in `configs/` folder.
3. Run training script:
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 -m external.moshi-finetune.train configs/your_config_file.yaml
```
- weights will be saved in `weights/` folder
- you can log training process using wandb
