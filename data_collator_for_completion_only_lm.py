from transformers import DataCollatorForLanguageModeling
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import warnings
import numpy as np
import torch

class DataCollatorForCompletionOnlyLMExperimental(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, List[int]]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_template: Union[str, List[int]],
        instruction_template: Optional[Union[str, List[int]]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        if not self.mlm and self.instruction_template and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[0]:
                    # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                    if (
                        self.response_token_ids
                        == batch["labels"][i][idx : idx + len(self.response_token_ids)].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(self.response_token_ids)

                    # Make pytorch loss function ignore all tokens up through the end of the response key
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        else:
            for i in range(len(examples)):
                # look for instruction template
                try:
                    ins_start = np.where(batch["labels"][i] == self.instruction_token_ids[0])[0][0]
                except IndexError:
                    print(f"No instruction template start: {self.instruction_token_ids}")  #\n{batch['labels'][i]}")
                    batch["labels"][i, :] = self.ignore_index
                    continue 
                ins_end = ins_start + len(self.instruction_token_ids)
                try:
                    assert torch.equal(batch["labels"][i][ins_start: ins_end], torch.tensor(self.instruction_token_ids))
                except AssertionError:
                    print(f"No instruction template: {self.instruction_token_ids}")  #\n{batch['labels'][i]}")
                    batch["labels"][i, :] = self.ignore_index
                    continue 
                # look for response template
                try:
                    resp_start = np.where(batch["labels"][i] == self.response_token_ids[0])[0][0]
                except IndexError:
                    print(f"No response template start: {self.response_token_ids}")  #\n{batch['labels'][i]}")
                    print(self.tokenizer.decode(batch["input_ids"][i]))
                    batch["labels"][i, :] = self.ignore_index
                    continue 
                resp_end = resp_start + len(self.response_token_ids)
                try:
                    assert torch.equal(batch["labels"][i][resp_start: resp_end], torch.tensor(self.response_token_ids))
                except AssertionError:
                    print(f"No response template: {self.response_token_ids}")  #\n{batch['labels'][i]}")
                    batch["labels"][i, :] = self.ignore_index
                    continue
                # set relevant indices to be ignored
                batch["labels"][i][:resp_end] = self.ignore_index
        return batch