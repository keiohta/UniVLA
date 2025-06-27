# LIBERO Benchmark

[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) is a benchmark for studying knowledge transfer in multitask and lifelong robot learning problems. 

![](imgs/libero.png)

## Environment Setup
```shell
apt-get install libgl1-mesa-dri

cd reference/RoboVLMs

bash scripts/setup_libero.sh
```

Dataset download from [huggingface](https://huggingface.co/datasets/openvla/modified_libero_rlds).

## Dataset Preparation
```shell
# 1. process the dataset
python tools/process/libero_process.py

# 2. extract the vq tokens, need to change the dataset & output path
bash scripts/tokenizer/extract_vq_emu3.sh 

# 3. pickle generation for training
python scripts/normalize_libero.py \
  --dataset_path ./datasets/processed_data \
  --output_path ./datasets/processed_data/meta \
  --normalizer_path ./configs/normalizer_libero \
  --output_filename libero_all_norm.pkl
```

## Model Training
```shell
# default is one node training, recommend multi-node training.
bash scripts/simulator/libero/train_libero_video.sh
```

## Model Evaluation
```shell
cd reference/RoboVLMs

# 1 GPU inference, modify the {task_suite_name} for different tasks
bash scripts/run_eval_libero_univla.sh ${CKPT_PATH} 
```
