

from transformers.data.data_collator import PreTrainedTokenizerBase, PaddingStrategy
from typing import List, Optional, Tuple, Union, Any
import numpy as np
from dataclasses import dataclass
import torch

EncodedInput = List[int]

import torch


class CustomDataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`], *optional*):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    def __init__(
        self,
        training_type: str,
        tokenizer: PreTrainedTokenizerBase,
        model: Optional[Any] = None,
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        label_pad_token_id: int = -100,
        return_tensors: str = "pt",
    ):
        self.training_type = training_type
        self.tokenizer = tokenizer
        self.model = model
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors


        # Determine the label name (either 'label' or 'labels')
        label_name = "label" if "label" in features[0] else "labels"

        # Extract labels
        if self.training_type=="pre-training":
            labels = [feature[label_name][0] for feature in features] if label_name in features[0] else None
        else:
            labels = [feature[label_name] for feature in features] if label_name in features[0] else None


        # Convert list[None] to None if necessary
        if labels is not None and all(label is None for label in labels):
            labels = None

        # Check if decoder_input_ids are provided in features
        decoder_input_ids_present = any("decoder_input_ids" in feature for feature in features)

        # Remove labels from features
        inputs = [{k: v for k, v in feature.items() if k != label_name} for feature in features]

        # Pad the inputs (including decoder_input_ids if they are present)
        batch = self.tokenizer.pad(
            inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # Pad labels manually
        if labels is not None:
            no_padding = self.padding is False or self.padding == PaddingStrategy.DO_NOT_PAD
            if no_padding:
                batch["labels"] = labels
            else:
                max_label_length = max(len(l) for l in labels)
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                padding_side = self.tokenizer.padding_side
                padded_labels = []
                for label in labels:
                    remainder = [self.label_pad_token_id] * (max_label_length - len(label))
                    if padding_side == "right":
                        padded_label = label + remainder
                    else:
                        padded_label = remainder + label
                    padded_labels.append(padded_label)

                batch["labels"] = padded_labels

            # Convert labels to tensors
            if return_tensors == "pt":
                batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
            elif return_tensors == "tf":
                batch["labels"] = tf.constant(batch["labels"], dtype=tf.int64)
            else:
                batch["labels"] = np.array(batch["labels"], dtype=np.int64)
        else:
            batch["labels"] = None

        # Prepare decoder_input_ids if they are not provided
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
            and not decoder_input_ids_present
        ):
           
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
            batch["decoder_input_ids"] = decoder_input_ids


        return batch