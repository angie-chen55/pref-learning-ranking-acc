from dataclasses import dataclass, field
from typing import List, Optional

import logging
import os
import sys
import torch
import wandb
from accelerate import Accelerator
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    set_seed,
    TrainingArguments,
)

from trl import SFTTrainer, is_xpu_available, DataCollatorForCompletionOnlyLM
from dataset_lib import PreferenceDataset
from data_collator_for_completion_only_lm import (
    DataCollatorForCompletionOnlyLMExperimental,
)

logger = logging.getLogger(__name__)


tqdm.pandas()


def tokenize_template(temp: str, tokenizer) -> List[int]:
    """Some tokenizers prepend '_' or other symbols in front of the template even when add_special_tokens=False. Fix this."""
    tokens = tokenizer.tokenize(
        f"\n{temp}"
    )  # Add the newline to avoid tokenizer pre-pending the '_'
    # Search through the tokens until we find one that starts the same way that the string does.
    fixed_tokens = None
    for i, t in enumerate(tokens):
        if temp.startswith(t):
            fixed_tokens = tokens[i:]
    if fixed_tokens is None:
        raise ValueError(
            f'Could not find any tokens in {tokens} that the string "{temp}" starts with.'
        )
    token_ids = tokenizer.convert_tokens_to_ids(fixed_tokens)
    return token_ids


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(
        default="facebook/opt-350m", metadata={"help": "the model name"}
    )

    # Dataset options
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "the dataset name"},
    )
    dataset_text_field: Optional[str] = field(
        default="sft_target", metadata={"help": "the text field of the dataset"}
    )
    instruction_template: Optional[str] = field(
        default=None,
        metadata={
            "help": "The start of the instruction portion of the chat logs, if applicable (e.g. '## Human:')"
        },
    )
    response_template: Optional[str] = field(
        default=None,
        metadata={
            "help": "The start of the response portion of the chat logs, if applicable (e.g. '## Assistant:')"
        },
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, appends `eos_token_id` at the end of each sample being packed."
        },
    )
    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, tokenizers adds special tokens to each sample being packed."
        },
    )
    eval_sample_size: Optional[int] = field(
        default=None, metadata={"help": "Size to sub-sample evaluation dataset to."}
    )
    preprocessed_data_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "If dataset has already been pre-processed, load from this filepath instead."
        },
    )

    # Instrumentation
    wandb_project: Optional[str] = field(
        default="sft", metadata={"help": "wandb project name"}
    )
    wandb_run_name: Optional[str] = field(
        default="anthropic_hh_rlhf", metadata={"help": "wandb run name"}
    )
    sanity_check: Optional[bool] = field(
        default=True, metadata={"help": "only train on 1000 samples"}
    )

    # Other training options
    seq_length: Optional[int] = field(
        default=512, metadata={"help": "Input sequence length"}
    )
    load_in_8bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 8 bits precision"}
    )
    load_in_4bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 4 bits precision"}
    )
    trust_remote_code: Optional[bool] = field(
        default=False, metadata={"help": "Enable `trust_remote_code`"}
    )
    peft_lora_r: Optional[int] = field(
        default=64, metadata={"help": "the r parameter of the LoRA adapters"}
    )
    peft_lora_alpha: Optional[int] = field(
        default=16, metadata={"help": "the alpha parameter of the LoRA adapters"}
    )
    use_auth_token: Optional[bool] = field(
        default=True, metadata={"help": "Use HF auth token to access the model"}
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    # Torch dtype used for training
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Torch_dtype to use when loading and training the model (e.g. 'float16,' 'bfloat16', etc."
        },
    )


parser = HfArgumentParser((ScriptArguments, TrainingArguments))
script_args, training_args = parser.parse_args_into_dataclasses()
training_args.report_to = ["wandb"]
training_args.run_name = script_args.wandb_run_name
os.environ["WANDB_PROJECT"] = script_args.wandb_project

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger.setLevel(training_args.log_level.upper())
logger.info(f"Training/evaluation parameters:\n{training_args}")
logger.info(f"Script args:\n{script_args}")

# Step 1: Load the model
if script_args.load_in_8bit and script_args.load_in_4bit:
    raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
elif script_args.load_in_8bit or script_args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
    )
    # Copy the model to each device
    device_map = (
        {"": f"xpu:{Accelerator().local_process_index}"}
        if is_xpu_available()
        else {"": Accelerator().local_process_index}
    )
    torch_dtype = torch.bfloat16
else:
    device_map = None
    quantization_config = None
    torch_dtype = None
    if script_args.torch_dtype is not None and hasattr(torch, script_args.torch_dtype):
        torch_dtype = getattr(torch, script_args.torch_dtype)

# Set seed before initializing model.
set_seed(training_args.seed)

model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
    use_auth_token=script_args.use_auth_token,
    attn_implementation="flash_attention_2" if script_args.use_flash_attn else "eager",
)
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
if tokenizer.pad_token is None:
    # Fix from https://github.com/huggingface/transformers/issues/4122#issuecomment-713433149
    tokenizer.pad_token = tokenizer.eos_token

if script_args.dataset_name == "Anthropic/hh-rlhf":
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template=tokenize_template(
            script_args.instruction_template, tokenizer
        ),
        response_template=tokenize_template(script_args.response_template, tokenizer),
        tokenizer=tokenizer,
        mlm=False,
    )
else:
    templates = {
        "gpt2": {
            "instruction_template": [
                20490
            ],  # [20490, 11097] or [20490, 25]. Tokenizer gives [1, 20490, *]
            "response_template": [15286, 25],
        },
        "EleutherAI/pythia-2.8b": {
            "instruction_template": [22705, 27],  # Tokenizer gives [3, 22705, 5136]
            "response_template": [17762, 27],
        },
        "meta-llama/Llama-2-7b-hf": {
            "instruction_template": [12968, 29901],
            "response_template": [4007, 22137, 29901],
        },
    }
    collator = DataCollatorForCompletionOnlyLMExperimental(
        instruction_template=templates[script_args.model_name]["instruction_template"],
        response_template=templates[script_args.model_name]["response_template"],
        tokenizer=tokenizer,
        mlm=False,
    )

if script_args.dataset_name == "Anthropic/hh-rlhf":
    dataset = load_dataset(script_args.dataset_name)
    if script_args.sanity_check:
        dataset = dataset.shuffle().select(range(1000))

else:

    def tokenize_str(temp: str, tokenizer):
        """For GPT-2 on SHP"""
        tokens = tokenizer.tokenize(temp, add_special_tokens=False)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        return token_ids

    dataset = {}
    for split in ["train", "validation", "test"]:
        dsname = script_args.dataset_name.split("/")[-1]
        ds_path = os.path.join(
            script_args.preprocessed_data_dir,
            f"{dsname}_{split}_sft.pt",
        )
        if os.path.exists(ds_path):
            logging.info(f"Found pre-processed data already at {ds_path}. Loading...")
            ds = load_from_disk(ds_path)
            if script_args.sanity_check:
                ds = ds.select(range(1000))
            dataset[split] = ds
        else:
            logging.info(
                f"Did not find already pre-processed data already at {ds_path}. Processing from scratch..."
            )
            try:
                ds = PreferenceDataset(
                    script_args.dataset_name,
                    dataset_config=None,
                    dataset_split=split,
                    sample_size=1000 if script_args.sanity_check else None,
                ).dataset_pairs
            except ValueError:
                continue

            def concat_instruction_and_response(sample):
                sample["full_text"] = (
                    "Human: "
                    + sample["instruction"]
                    + " Assistant: "
                    + sample[sample["preferred_output"]]
                )
                return sample

            ds = ds.map(concat_instruction_and_response)
            logging.info(f"{split} length before filter: ", len(ds))
            # eval loss will be nan if a full batch has no outputs for scoring, so we pre-emptively remove sequences that are too long
            ds = ds.filter(
                lambda s: len(tokenize_str("Human: " + s["instruction"], tokenizer))
                < script_args.seq_length - 3
            )
            logging.info(f"{split} length after filtering out long examples: ", len(ds))
            # also filter out examples that do not have a response template
            def check_for_response_template(full_text):
                tokenized = collator.tokenizer.encode(full_text)
                if collator.response_token_ids not in tokenized:
                    logging.info(
                        f"Filtering out example due to lacking response template."
                    )
                    return False
                return True

            ds = ds.filter(lambda s: check_for_response_template(s["full_text"]))

            logging.info(
                f"{split} length after filtering out examples without response template: ",
                len(ds),
            )
            ds.set_format(type="torch")
            ds.save_to_disk(f"{script_args.dataset_name}_{split}_sft.pt")
            dataset[split] = ds


training_dataset = dataset["train"]
if "test" in dataset:
    test_dataset = dataset["test"]
else:
    test_dataset = None
if "val" in dataset:
    val_dataset = dataset["val"]
elif "validation" in dataset:
    val_dataset = dataset["validation"]
else:
    val_dataset = None
if val_dataset is None and test_dataset is not None:
    val_test_split = test_dataset.train_test_split(test_size=0.5, seed=0)
    val_dataset = val_test_split["train"]
    test_dataset = val_test_split["test"]
if test_dataset is None and val_dataset is not None:
    val_test_split = val_dataset.train_test_split(test_size=0.5, seed=0)
    val_dataset = val_test_split["train"]
    test_dataset = val_test_split["test"]
if val_dataset is not None and script_args.eval_sample_size is not None:
    val_dataset = val_dataset.train_test_split(test_size=script_args.eval_sample_size)[
        "test"
    ]


# gradient ckpt
model.config.use_cache = not training_args.gradient_checkpointing
if training_args.gradient_checkpointing:
    training_args.gradient_checkpointing_kwargs = {
        "use_reentrant": script_args.use_reentrant
    }

# Step 5: Define the Trainer
wandb.init(
    project=script_args.wandb_project,
    name=script_args.wandb_run_name,
    entity="llm-calibration",
)
trainer = SFTTrainer(
    model=model,
    args=training_args,
    max_seq_length=script_args.seq_length,
    train_dataset=training_dataset,
    dataset_text_field=script_args.dataset_text_field,
    data_collator=collator,
    eval_dataset=val_dataset,
    packing=script_args.packing,
    dataset_kwargs={
        "append_concat_token": script_args.append_concat_token,
        "add_special_tokens": script_args.add_special_tokens,
    },
    # test_dataset=test_dataset,
)
trainer.accelerator.print(f"{trainer.model}")
checkpoint = None
if training_args.resume_from_checkpoint is not None:
    if training_args.resume_from_checkpoint in ["true", "True"]:
        checkpoint = True
    elif training_args.resume_from_checkpoint in ["false", "False"]:
        checkpoint = False
    else:
        checkpoint = training_args.resume_from_checkpoint
trainer.train(resume_from_checkpoint=checkpoint)

# Step 6: Save the model
if not script_args.sanity_check:
    # saving final model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(training_args.output_dir)
