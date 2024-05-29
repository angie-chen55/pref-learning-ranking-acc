import logging
import numpy as np
import os
import torch
import torch.distributed.checkpoint as dist_cp

from common import maybe_log, CHAT_TEMPLATE_MAP
from string import Template
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Any, Union


class ChatClient:
    def __init__(
        self,
        model_name_or_path: str = "gpt2",
        hf_model_name: str = None,  # The name of the model in the HF hub. Only necessary
        # if loading from a sharded FSDP checkpoint.
        logger: logging.Logger = None,
        max_generate_length: int = 500,
        device: str = "cuda",
        model_type: str = "hf",  # "hf" or "gpt"
    ):
        self.model_name_or_path = model_name_or_path
        self.hf_model_name = hf_model_name
        self.logger = logger
        self.model_type = model_type

        self.max_generate_length = max_generate_length

        self.device = device

        if (
            os.path.exists(model_name_or_path)
            and not os.path.exists(f"{model_name_or_path}/config.json")
            and os.path.exists(f"{model_name_or_path}/pytorch_model_fsdp_0")
        ):
            maybe_log(
                self.logger,
                f"{model_name_or_path} is a sharded FSDP model. Attempting to consolidate and convert.",
                level="info",
            )
            self.model, self.config, self.tokenizer = self._convert_checkpoint(
                hf_model_name=hf_model_name,
                fsdp_model_path=f"{model_name_or_path}/pytorch_model_fsdp_0",
                output_path=model_name_or_path,
            )
        else:
            self.config = AutoConfig.from_pretrained(
                self.model_name_or_path, trust_remote_code=True
            )
            extra_model_kwargs = {}
            if "gg-hf/gemma" in self.model_name_or_path:
                extra_model_kwargs["torch_dtype"] = torch.bfloat16
                extra_model_kwargs["device_map"] = "cuda"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path, trust_remote_code=True, **extra_model_kwargs
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            if "llama" in self.model_name_or_path or (
                self.hf_model_name is not None and "llama" in self.hf_model_name
            ):
                self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
            elif self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                raise ValueError(
                    f"Tokenizer for {self.model_name_or_path} does not have a PAD token."
                )
        if hasattr(self.config, "max_position_embeddings"):
            self.max_length = self.config.max_position_embeddings
        elif hasattr(self.config, "max_sequence_length"):
            self.max_length = self.config.max_sequence_length
        elif hasattr(self.config, "n_positions"):
            self.max_length = self.config.n_positions
        else:
            raise ValueError(
                f"Could not find max_position_embeddings, n_positions, or max_sequence_length in model config for {self.model_name_or_path}"
            )

    def _convert_checkpoint(
        self, hf_model_name: str, fsdp_model_path: str, output_path: str
    ):
        """
        hf_model_name: Name of model in HF Hub, e.g. "gpt2".
        fsdp_model_path: path to the fsdp checkpoint, for example `/x/checkpoint-xxx/pytorch_model_fsdp_x`
        output_path: output path to save the converted checkpoint
        """
        config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True).cuda()
        model = self._load_sharded_model_single_gpu(model, fsdp_model_path)
        model.save_pretrained(output_path, max_shard_size="10GB")
        tokenizer.save_pretrained(output_path)
        return model, config, tokenizer

    def _load_sharded_model_single_gpu(self, model, model_path):
        state_dict = {"model": model.state_dict()}

        dist_cp.load_state_dict(
            state_dict=state_dict,
            storage_reader=dist_cp.FileSystemReader(model_path),
            no_dist=True,
        )

        result = model.load_state_dict(state_dict["model"])
        maybe_log(
            self.logger,
            f"Sharded state checkpoint loaded from {model_path}. Result: {result}",
            level="info",
        )
        return model

    def _apply_chat_template(self, msgs) -> str:
        if self.model_name_or_path in CHAT_TEMPLATE_MAP:
            user_template, assistant_template, assistant_prompt = [
                Template(s) for s in CHAT_TEMPLATE_MAP[self.model_name_or_path]
            ]
            templated_str = ""
            for msg in msgs:
                if msg["role"] in ["user", "system"]:
                    templated_str += user_template.substitute(
                        user_message=msg["content"]
                    )
                elif msg["role"] == "assistant":
                    templated_str += assistant_template.substitute(
                        assistant_template.substitute(assistant_message=msg["content"])
                    )
                else:
                    raise ValueError(f"Unrecognized chat role: {msg['role']}")
            templated_str += assistant_prompt.substitute()
            chat_msgs = templated_str
        elif self.tokenizer.chat_template is not None:
            chat_msgs = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        else:
            raise NotImplementedError(
                f"Chat template does not exist for model {self.model_name_or_path}."
            )
        return chat_msgs

    def _tokenize_batch(
        self, text_batch: Union[List[List[Dict[str, str]]], List[str]]
    ) -> torch.tensor:
        """Tokenize either a batch of messages or a single text string."""
        if not text_batch:
            return None
        max_context_length = self.max_length - self.max_generate_length
        if isinstance(text_batch[0], list):
            # text batch is list of messages
            texts_to_tokenize = [self._apply_chat_template(text) for text in text_batch]
        else:
            texts_to_tokenize = text_batch
        tokenized = self.tokenizer(
            texts_to_tokenize,
            truncation=True,
            padding="longest",
            max_length=max_context_length,
            return_tensors="pt",
        ).input_ids
        return tokenized

    def _chat_hf_model(
        self, msgs: Union[List[Dict[str, str]], str], **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a completion for either a series of messages or a single text prompt.
        """
        tokenized = self._tokenize_batch([msgs])
        outputs = self.model.generate(tokenized.to(self.device), **kwargs)
        return {"output_strs": self.tokenizer.batch_decode(outputs)}

    def chat_single_turn(self, msgs: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        return self._chat_hf_model(msgs, **kwargs)

    def chat_raw_logits(
        self, input_ids: torch.tensor, attention_mask: torch.tensor
    ) -> torch.tensor:
        """Alternative API for getting all logits."""
        return self.model(input_ids, attention_mask=attention_mask).logits

    def chat_single_turn_text(self, text: str, **kwargs) -> Dict[str, Any]:
        msgs = [{"role": "user", "content": text}]
        return self._chat_hf_model(msgs, **kwargs)
