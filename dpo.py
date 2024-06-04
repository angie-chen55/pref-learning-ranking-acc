# Adapted from https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama_2/scripts/dpo_llama2.py
# 0. imports
from dataclasses import dataclass, field
from typing import Dict, Optional

import logging
import os
import sys
import torch
import wandb
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)

from dpo_trainer import DPOExperimentalTrainer
from process_rlhf_datasets import get_hh_for_dpo

logger = logging.getLogger(__name__)

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # training parameters
    alpha: Optional[float] = field(
        default=0.1, metadata={"help": "the beta parameter for DPO alpha-scaling loss"}
    )
    beta: Optional[float] = field(
        default=0.1, metadata={"help": "the beta parameter for DPO loss"}
    )
    gamma: Optional[float] = field(
        default=0.1, metadata={"help": "the gamma parameter for unlikelihood loss"}
    )
    loss_type: Optional[str] = field(
        default="sigmoid",
        metadata={
            "help": "The type of loss objective to use. Options: ['sigmoid', 'hinge', 'ipo', 'kto_pair', 'unlikelihood']"
        },
    )
    wandb_project: Optional[str] = field(
        default="dpo", metadata={"help": "wandb project name"}
    )
    wandb_run_name: Optional[str] = field(
        default="anthropic_hh_rlhf", metadata={"help": "wandb run name"}
    )
    wandb_entity: Optional[str] = field(
        default="llm-calibration", metadata={"help": "wandb username or team name"}
    )
    model_name_or_path: Optional[str] = field(
        default="gpt2", metadata={"help": "the model name"}
    )
    max_length: Optional[int] = field(
        default=512, metadata={"help": "max length of each sample"}
    )
    max_prompt_length: Optional[int] = field(
        default=256, metadata={"help": "max length of each sample's prompt"}
    )
    max_target_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "Only used for encoder decoder model. Max target of each sample's prompt"
        },
    )
    label_pad_token_id: Optional[int] = field(
        default=-100, metadata={"help": "label for non response tokens"}
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    # instrumentation
    sanity_check: Optional[bool] = field(
        default=True, metadata={"help": "only train on 1000 samples"}
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    # Torch dtype used for training
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Torch_dtype to use when loading and training the model (e.g. 'float16,' 'bfloat16', etc."
        },
    )
    generate_during_eval: Optional[bool] = field(
        default=True,
    )

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "the dataset for training"}
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.report_to = ["wandb"]
    training_args.run_name = script_args.wandb_run_name
    training_args.log_level = "info"
    training_args.logging_first_step = True
    os.environ["WANDB_PROJECT"] = script_args.wandb_project

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.setLevel(training_args.log_level.upper())
    logger.info(f"Training/evaluation parameters:\n{training_args}")
    logger.info(f"Script args:\n{script_args}")

    # 1. load a pretrained model
    device_map = {"": training_args.device.index}
    model_kwargs = {
        "device_map": device_map,
    }
    if script_args.torch_dtype is not None:
        assert hasattr(torch, script_args.torch_dtype)
        model_kwargs["torch_dtype"] = getattr(torch, script_args.torch_dtype)
    model_kwargs["attn_implementation"] = (
        "flash_attention_2" if script_args.use_flash_attn else "eager"
    )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path, **model_kwargs
    )

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    model_ref = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path, device_map=device_map
    )

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the Anthropic Helpful-Harmless dataset
    train_dataset = get_hh_for_dpo("train", sanity_check=script_args.sanity_check)
    eval_and_test = get_hh_for_dpo(
        "test", sanity_check=script_args.sanity_check
    ).train_test_split(test_size=0.5, seed=0)
    eval_dataset = eval_and_test["train"]
    test_dataset = eval_and_test["test"]

    if script_args.sanity_check:
        training_args.eval_steps = 10

    wandb.init(
        project=script_args.wandb_project,
        name=script_args.wandb_run_name,
        entity=script_args.wandb_entity,
    )

    # gradient ckpt
    model.config.use_cache = not training_args.gradient_checkpointing
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {
            "use_reentrant": script_args.use_reentrant
        }

    # 5. initialize the DPO trainer
    dpo_trainer = DPOExperimentalTrainer(
        model,
        model_ref,
        args=training_args,
        alpha=script_args.alpha,
        beta=script_args.beta,
        gamma=script_args.gamma,
        loss_type=script_args.loss_type,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=script_args.max_length,
        max_target_length=script_args.max_target_length,
        max_prompt_length=script_args.max_prompt_length,
        generate_during_eval=script_args.generate_during_eval,
    )

    # hack because DPOTrainer doesn't properly place both
    # models on the same device
    if dpo_trainer.is_fsdp_enabled:
        prepared_model = dpo_trainer._wrap_model(
            dpo_trainer.model, training=True, dataloader=None
        )
        if hasattr(dpo_trainer.lr_scheduler, "step"):
            prepared_model, dpo_trainer.optimizer = dpo_trainer.accelerator.prepare(
                prepared_model, dpo_trainer.optimizer
            )
        else:
            (
                prepared_model,
                dpo_trainer.optimizer,
                dpo_trainer.lr_scheduler,
            ) = dpo_trainer.accelerator.prepare(
                prepared_model, dpo_trainer.optimizer, dpo_trainer.lr_scheduler
            )
        dpo_trainer.model_wrapped = prepared_model
        dpo_trainer.model = prepared_model
        if dpo_trainer.ref_model is not None:
            dpo_trainer.ref_model = dpo_trainer.accelerator.prepare_model(
                dpo_trainer.ref_model
            )

        dpo_trainer.accelerator.prepare_model = (
            lambda model, *args, **kwargs: model
        )  # Monkey-patch prepare_model a no-op , since we have manually prepared the models

    # 6. train
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        if training_args.resume_from_checkpoint in ["true", "True"]:
            checkpoint = True
        elif training_args.resume_from_checkpoint in ["false", "False"]:
            checkpoint = False
        else:
            checkpoint = training_args.resume_from_checkpoint
    dpo_trainer.train(resume_from_checkpoint=checkpoint)

    # 7. save final model!
    if not script_args.sanity_check:
        dpo_trainer.save_model(training_args.output_dir)
