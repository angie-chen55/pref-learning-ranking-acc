# Preference Learning Algorithms Do Not Learn Preference Rankings
This is the repository for the paper "[Preference Learning Algorithms Do Not Learn Preference Rankings](https://arxiv.org/abs/2405.19534)," by Angelica Chen, Sadhika Malladi, Lily H. Zhang, Xinyi Chen, Qiuyi Zhang, Rajesh Ranganath, and Kyunghyun Cho.

A draft of our paper can be found [in this repo](pref_learning_algs_do_not_learn_pref_rankings.pdf) and [on Arxiv](https://arxiv.org/abs/2405.19534).

## Set-Up
We have separate conda environments for each type of experiment.

### Ranking Accuracy Evaluations
To run the ranking accuracy evaluations (Fig. 1 and Tables 2-5), install the packages in [environment.yaml](environment.yaml):
```
conda env create -f environment.yaml -n ranking_acc
conda activate ranking_acc
```

### DPO and $DPO^\gamma$ Training
To train models using either DPO or $\mathcal{L}_\text{DPO}^\gamma$ (Eq. 4), install the packages in [environment_dpo.yaml](environment_dpo.yaml):
```
conda env create -f environment_dpo.yaml -n dpo
conda activate dpo
```

## Running Ranking Accuracy Evaluations
Our ranking accuracy evaluations should work with either local HF models or models hosted on the HF Hub.

To compute the *ranking accuracy* of a model on a variety of datasets, run:
```
conda activate ranking_acc
python calculate_calibration_metrics.py \
  --lm-name-or-path=<MODEL> \
  --output-dir=<OUTPUT_DIR> \
  --output-filename=<OUTPUT_FILENAME> \
  --sample-size=1000 \
  --batch-size=8
```

To compute the *idealized ranking accuracy* for any LLM trained with a particular reference model, run:
```
conda activate ranking_acc
python calculate_upper_bound_rlhf.py \
  --lm-name-or-path=<REFERENCE_MODEL> \
  --output-dir=<OUTPUT_DIR> \
  --output-filename=<OUTPUT_FILEPATH> \
  --sample-size=1000 \
  --batch-size=8
```

## Training a Model with $DPO^\gamma$
To train a model using the $\mathcal{L}_\text{DPO}^\gamma$ objective (Eq. 4), first run SFT on your model:
```
conda activate dpo
python sft.py \
  --model_name=<MODEL> \
  --dataset_name=Anthropic/hh-rlhf \
  --dataset_text_field=chosen \
  --instruction_template=\"Human:\" --response_template=\"Assistant:\" \
  --sanity_check=false \
  --output_dir=<OUTPUT_DIR> \
  --wandb_run_name=<RUN_NAME> \
  --gradient_accumulation_steps=<GRADIENT_ACCUMULATION_STEPS> \
  --per_device_train_batch_size=<PER_DEVICE_TRAIN_BATCH_SIZE> \
  --per_device_eval_batch_size=<PER_DEVICE_EVAL_BATCH_SIZE> \
  --learning_rate=<LR> \
  --warmup_steps=<WARMUP_STEPS> \
  --num_train_epochs=<NUM_TRAIN_EPOCHS> \
  --logging_steps=<LOGGING_STEPS> \
  --log_level=info \
  --evaluation_strategy=<EVALUATION_STRATEGY> \
  --eval_steps=<EVAL_STEPS> \
  --save_strategy=<SAVE_STRATEGY> \
  --save_steps=<SAVE_STEPS> \
  --load_best_model_at_end=true \
  --save_total_limit=<SAVE_TOTAL_LIMIT> \
  --seq_length=512
```

Then, run $DPO^\gamma$ on your SFT-ed model:
```
conda activate dpo
python dpo.py \
  --model_name_or_path=<PATH_TO_SFT_MODEL> \
  --beta=0.1 \
  --alpha=<GAMMA_VALUE> \
  --loss_type=alpha_scaling \
  --sanity_check=false \
  --logging_first_step=true --warmup_steps=150 --bf16=true \
  --output_dir=<OUTPUT_DIR> \
  --wandb_run_name=<RUN_NAME> \
  --gradient_accumulation_steps=<GRADIENT_ACCUMULATION_STEPS> \
  --per_device_train_batch_size=<PER_DEVICE_TRAIN_BATCH_SIZE> \
  --per_device_eval_batch_size=<PER_DEVICE_EVAL_BATCH_SIZE> \
  --learning_rate=<LR> \
  --num_train_epochs=<NUM_TRAIN_EPOCHS> \
  --log_level=info \
  --save_strategy=<SAVE_STRATEGY> \
  --save_steps=<SAVE_STEPS> \
  --load_best_model_at_end=true \
  --max_length=512 \
  --evaluation_strategy=<EVALUATION_STRATEGY> \
  --logging_steps=<LOGGING_STEPS> \
  --eval_steps=<EVAL_STEPS> \
  --optim=rmsprop
```
where the flag `--alpha` controls the value of $\gamma$.
