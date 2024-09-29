"""
Processor class for Molmo.
"""

from typing import Optional

import PIL
from PIL import ImageOps
from PIL.Image import Image

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

import numpy as np
import torch

from transformers.image_utils import ImageInput
from transformers.processing_utils import (
    TextKwargs,
    ProcessingKwargs,
    ProcessorMixin,
)

from transformers.tokenization_utils_base import TextInput
from transformers.utils import logging

from transformers import AutoTokenizer
from .image_preprocessing_molmo import MolmoImagesKwargs, MolmoImageProcessor


logger = logging.get_logger(__name__)


DEFAULT_IMAGE_PATCH_TOKEN = f"<im_patch>"
DEFAULT_IM_START_TOKEN = f"<im_start>"
DEFAULT_IM_END_TOKEN = f"<im_end>"
DEFAULT_IM_COL_TOKEN = f"<im_col>"
IMAGE_PROMPT = "<|image|>"

EXTRA_TOKENS = (DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_COL_TOKEN, IMAGE_PROMPT)


def get_special_token_ids(tokenizer):
    ids = tokenizer.encode("".join(EXTRA_TOKENS), add_special_tokens=False)
    assert len(ids) == len(EXTRA_TOKENS)
    return {k: i for k, i in zip(EXTRA_TOKENS, ids)}


class MolmoTextKwargs(TextKwargs, total=False):
    style: Optional[str]
    system_prompt: Optional[str]
    message_format: Optional[str]
    always_start_with_space: Optional[bool]
    sequence_length: Optional[int]


class MolmoProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: MolmoTextKwargs
    images_kwargs: MolmoImagesKwargs
    _defaults = {
        "images_kwargs": {
            "max_crops": 12,
            "overlap_margins": [4, 4],
            "base_image_input_size": [336, 336],
            "image_token_length_w": 12,
            "image_token_length_h": 12,
            "image_patch_size": 14,
            "image_padding_mask": True,
        },
        "text_kwargs": {
            "style": "long_caption",
            "system_prompt": "none",
            "message_format": "role",
            "always_start_with_space": True,
            "sequence_length": 1536,
            "padding": False,
        },
    }


class MolmoProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, image_processor: MolmoImageProcessor = None, tokenizer : AutoTokenizer = None, **kwargs):
        # self.image_processor = image_processor
        # self.tokenizer = tokenizer
        super().__init__(image_processor, tokenizer)
        self._special_tokens = None

    @property
    def special_token_ids(self):
        if self._special_tokens is None:
            self._special_tokens = get_special_token_ids(self.tokenizer)
        return self._special_tokens

    def get_tokens_input(self, prompt, message_format, always_start_with_space):
        if message_format == "none" or message_format is None:
            pass
        elif message_format == "role":
            prompt = "User: " + prompt + " Assistant:"
        else:
            raise NotImplementedError(f"Message format {message_format} not implemented")

        if always_start_with_space:
            prompt = " " + prompt

        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)

        return tokens

    def process(
        self,
        text: TextInput = None,
        images: ImageInput = None,
        **kwargs: Unpack[MolmoProcessorKwargs],
    ):
        output_kwargs = self._merge_kwargs(
            MolmoProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        tokens = self.get_tokens_input(
            text,
            output_kwargs["text_kwargs"]["message_format"],
            output_kwargs["text_kwargs"]["always_start_with_space"],
        )

        image_token_id = self.special_token_ids[IMAGE_PROMPT]

        if images is not None:
            if not isinstance(images, (list, tuple)):
                images = [images]
            image_arrays = []
            for image in images:
                if isinstance(image, Image):
                    image = image.convert("RGB")
                    # Handle images with EXIF orientation tags, which PIL will ignore by default
                    # https://github.com/python-pillow/Pillow/issues/4703
                    img = ImageOps.exif_transpose(image)
                    image_arrays.append(np.array(image))
                else:
                    assert len(image.shape) == 3 and image.shape[-1] == 3
                    image_arrays.append(image.astype(np.uint8))
            images = image_arrays
            # For now only support inserting images at the start
            image_idx = [-1]*len(images)
        else:
            image_idx = None

        sequence_length = output_kwargs["text_kwargs"]["sequence_length"]

        image_patch_token_id = self.special_token_ids[DEFAULT_IMAGE_PATCH_TOKEN]
        image_col_token_id = self.special_token_ids[DEFAULT_IM_COL_TOKEN]
        image_start_token_id = self.special_token_ids[DEFAULT_IM_START_TOKEN]
        image_end_token_id = self.special_token_ids[DEFAULT_IM_END_TOKEN]
        out = self.image_processor.multimodal_preprocess(
            images=images,
            image_idx=image_idx,
            tokens=np.asarray(tokens).astype(np.int32),
            sequence_length=sequence_length,
            image_patch_token_id=image_patch_token_id,
            image_col_token_id=image_col_token_id,
            image_start_token_id=image_start_token_id,
            image_end_token_id=image_end_token_id,
            **output_kwargs["images_kwargs"]
        )

        # Prepend BOS
        # qwen2 and olmo do not have a BOS, and instead use EOS as a generic seperator token.
        bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        decoder_input_tokens = np.pad(out["input_ids"], [[1, 0]], constant_values=bos)
        out["input_ids"] = decoder_input_tokens
        if "image_input_idx" in out:
            # Shift patch mapping up by one since we added BOS
            image_input_idx = out["image_input_idx"]
            out["image_input_idx"] = np.where(image_input_idx < 0, image_input_idx, image_input_idx + 1)

        for k, v in out.items():
            out[k] = torch.from_numpy(v)

        return out


MolmoProcessor.register_for_auto_class()
