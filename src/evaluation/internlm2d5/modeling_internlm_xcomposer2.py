# Copyright (c) The InternLM team and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on transformers/src/transformers/models/llama/modeling_llama.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch InternLMXComposer2 model."""
import os
import re
import copy
import queue
import threading
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from PIL import Image
import numpy as np
import random
from torch import nn
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

try:
    from transformers.generation.streamers import BaseStreamer
except:  # noqa # pylint: disable=bare-except
    BaseStreamer = None

import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

from .build_mlp import build_vision_projector, build_vision_tower
from .ixc_utils import Image_transform, Video_transform, load_video, frame2img, get_font
from .configuration_internlm_xcomposer2 import InternLMXcomposer2Config
from .modeling_internlm2 import (
    InternLM2_INPUTS_DOCSTRING,
    InternLM2Model,
    InternLM2PreTrainedModel,
)

_CONFIG_FOR_DOC = "InternLMXcomposer2Config"

image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".wmv"}
from PIL.Image import Image as type_image


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True
        return False


def get_stopping_criteria(stop_words_ids):
    stop_words_ids = [torch.tensor([i]).cuda() for i in stop_words_ids]
    stopping_criteria = StoppingCriteriaList(
        [StoppingCriteriaSub(stops=stop_words_ids)]
    )
    return stopping_criteria


def set_random_seed(seed, set_cudnn=False):
    """Set the random seed for reproducibility.

    Parameters:
    seed (int): The seed to use for generating random numbers.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # For multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    if set_cudnn and torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class InternLMXComposer2ForCausalLM(InternLM2PreTrainedModel):
    _auto_class = "AutoModelForCausalLM"

    _tied_weights_keys = ["output.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = InternLM2Model(config)
        self.vocab_size = config.vocab_size
        self.output = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tokenizer = None
        self.hd_num = 25
        self.font = get_font()

        self.max_length = config.max_length
        print(f"Set max length to {self.max_length}")
        # Initialize weights and apply final processing
        self.post_init()
        self.plora_glb_GN = nn.Parameter(torch.zeros([1, 1, 4096]))
        self.plora_sub_GN = nn.Parameter(torch.zeros([1, 1, 1, 4096]))

        self.vit = build_vision_tower()
        self.vision_proj = build_vision_projector()

        self.vis_processor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, InternLM2Model):
            module.gradient_checkpointing = value
        if value:
            self.vit.vision_tower.vision_model.encoder.gradient_checkpointing = value

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.tok_embeddings = value

    def get_output_embeddings(self):
        return self.output

    def set_output_embeddings(self, new_embeddings):
        self.output = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def encode_text(self, text, add_special_tokens=False):
        token = self.tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids.to(self.device)
        embs = self.model.tok_embeddings(token)
        return embs

    def encode_img(self, image, hd_num=25):
        if image is None:
            return None
        if isinstance(image, str):
            _, ext = os.path.splitext(image)
            if ext.lower() in image_extensions:
                image = Image.open(image)
                image = Image_transform(image, hd_num=hd_num)
            elif ext.lower() in video_extensions:
                image = load_video(image)
                image = frame2img(image, self.font)
                image = Video_transform(image, hd_num=hd_num)
            else:
                print("Unknow input format", image)
                return None
            image = self.vis_processor(image).unsqueeze(0).to(self.device)
        elif isinstance(image, type_image):
            image = Image_transform(image, hd_num=hd_num)
            image = self.vis_processor(image).unsqueeze(0).to(self.device)
        else:
            assert isinstance(image, torch.Tensor)
        image = image.to(self.dtype)
        img_embeds, atts_img, img_target = self.img2emb(image)
        return img_embeds

    def img2emb(self, image):
        img_embeds, img_split = self.vit([image], self.plora_glb_GN, self.plora_sub_GN)
        if len(img_split) > 1:
            print("Batch Size >1 is not supported.")
            assert 0
        # print (img_embeds.shape)
        img_embeds = self.vision_proj(img_embeds)
        atts_img = torch.ones(img_embeds.size()[:-1], dtype=torch.long).to(
            img_embeds.device
        )

        img_target = (
            torch.ones(img_embeds.size()[:2], dtype=torch.long).to(img_embeds.device)
            * -100
        )

        return img_embeds, atts_img, img_target

    def prompt_wrap(self, img_embeds, prompt):
        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split("<ImageHere>")
        p_before_tokens = self.tokenizer(
            p_before, return_tensors="pt", add_special_tokens=True
        ).to(img_embeds.device)

        p_before_embeds = self.model.tok_embeddings(p_before_tokens.input_ids).expand(
            batch_size, -1, -1
        )
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds], dim=1)

        wrapped_atts_img = torch.ones(
            wrapped_img_embeds.size()[:-1], dtype=torch.long
        ).to(img_embeds.device)

        wrapped_target = (
            torch.ones(batch_size, wrapped_img_embeds.shape[1], dtype=torch.long).to(
                img_embeds.device
            )
            * -100
        )

        return wrapped_img_embeds, wrapped_atts_img, wrapped_target

    def text2emb(self, text, add_special_tokens=False):
        to_regress_tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=add_special_tokens,
        ).to(self.device)

        targets = self.mask_human_targets(to_regress_tokens.input_ids)
        targets = targets.to(self.device)
        return to_regress_tokens, targets

    def interleav_wrap_chat(
        self, query, image, history=[], meta_instruction="", max_length=16384, hd_num=24
    ):
        self.max_length = max_length
        prompt = ""
        if meta_instruction:
            prompt += (
                f"""[UNUSED_TOKEN_146]system\n{meta_instruction}[UNUSED_TOKEN_145]\n"""
            )
        for record in history:
            prompt += f"""[UNUSED_TOKEN_146]user\n{record[0]}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n{record[1]}[UNUSED_TOKEN_145]\n"""
        prompt += f"""[UNUSED_TOKEN_146]user\n{query}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"""

        image_nums = len(image)
        if image_nums == 1 and prompt.find("<ImageHere>") == -1:
            # print ('auto append image at the begining')
            prompt = "<ImageHere>" + prompt

        parts = prompt.split("<ImageHere>")
        wrap_embeds, wrap_im_mask = [], []
        temp_len = 0
        need_bos = True

        if len(parts) != image_nums + 1:
            # raise ValueError('Invalid <ImageHere> prompt format.')
            print("Waring! The image number != given position!")
        if image_nums > 1:
            hd_num = 6
        else:
            hu_num = hd_num
        for idx, part in enumerate(parts):
            if need_bos or len(part) > 0:
                part_tokens = self.tokenizer(
                    part,
                    return_tensors="pt",
                    padding="longest",
                    add_special_tokens=need_bos,
                ).to(self.device)
                if need_bos:
                    need_bos = False

                part_embeds = self.model.tok_embeddings(part_tokens.input_ids)
                wrap_embeds.append(part_embeds)
                wrap_im_mask.append(torch.zeros(part_embeds.shape[:2]))
                temp_len += part_embeds.shape[1]
            if idx < image_nums:
                img = self.encode_img(image[idx], hd_num)
                wrap_embeds.append(img)
                wrap_im_mask.append(torch.ones(img.shape[:2]))
                temp_len += img.shape[1]

            if temp_len > self.max_length:
                break

        wrap_embeds = torch.cat(wrap_embeds, dim=1)
        wrap_im_mask = torch.cat(wrap_im_mask, dim=1)
        wrap_embeds = wrap_embeds[:, : self.max_length].to(self.device)
        wrap_im_mask = wrap_im_mask[:, : self.max_length].to(self.device).bool()
        inputs = {"inputs_embeds": wrap_embeds}
        return inputs, wrap_im_mask, temp_len

    def interleav_wrap(self, img_list, text_list):
        wrap_embeds_list, wrap_atts_list = [], []
        wrap_target_list, wrap_im_mask_list = [], []

        for image, text in zip(img_list, text_list):
            img_embeds, atts_img, img_target = self.img2emb(image)
            text = text[0]
            parts = text.split("<ImageHere>")
            wrap_tokens, wrap_embeds, wrap_atts, wrap_im_mask = [], [], [], []
            temp_len = 0
            image_nums, im_len = img_embeds.shape[:2]
            need_bos = True
            for idx, part in enumerate(parts):
                if len(part) > 0:
                    part_tokens = self.tokenizer(
                        part,
                        return_tensors="pt",
                        padding="longest",
                        add_special_tokens=need_bos,
                    ).to(self.device)
                    if need_bos:
                        need_bos = False
                    wrap_tokens.append(part_tokens.input_ids)
                    part_embeds = self.model.tok_embeddings(part_tokens.input_ids)
                    wrap_embeds.append(part_embeds)
                    wrap_atts.append(part_tokens.attention_mask)
                    wrap_im_mask.append(
                        torch.zeros(part_embeds.shape[:2]).to(self.device)
                    )

                    temp_len += part_embeds.shape[1]
                if idx < image_nums:
                    wrap_tokens.append(img_target[idx].unsqueeze(0))
                    wrap_embeds.append(img_embeds[idx].unsqueeze(0))
                    wrap_atts.append(atts_img[idx].unsqueeze(0))
                    wrap_im_mask.append(torch.ones_like(atts_img[idx].unsqueeze(0)))

                    temp_len += im_len
                if temp_len > self.max_length:
                    break

            wrap_tokens = torch.cat(wrap_tokens, dim=1)
            wrap_embeds = torch.cat(wrap_embeds, dim=1)
            wrap_atts = torch.cat(wrap_atts, dim=1)
            wrap_im_mask = torch.cat(wrap_im_mask, dim=1)

            wrap_target = self.mask_human_targets(wrap_tokens).to(self.device)

            wrap_embeds = wrap_embeds[:, : self.max_length].to(self.device)
            wrap_atts = wrap_atts[:, : self.max_length].to(self.device)
            wrap_target = wrap_target[:, : self.max_length].to(self.device)
            wrap_im_mask = wrap_im_mask[:, : self.max_length].to(self.device)

            wrap_embeds_list.append(wrap_embeds)
            wrap_atts_list.append(wrap_atts)
            wrap_target_list.append(wrap_target)
            wrap_im_mask_list.append(wrap_im_mask)

        wrap_embeds = torch.cat(wrap_embeds_list)
        wrap_atts = torch.cat(wrap_atts_list)
        wrap_target = torch.cat(wrap_target_list)
        wrap_im_mask = torch.cat(wrap_im_mask_list)
        return wrap_embeds, wrap_atts, wrap_target, wrap_im_mask

    def mask_human_targets(self, input_ids, pure=False):
        target_batch = []
        for bs in range(input_ids.shape[0]):
            ids = input_ids[bs]
            targets = copy.deepcopy(ids)
            end_count = 0
            last_eoa = 0
            for i, temp_id in enumerate(ids):
                if temp_id == 92542:
                    if end_count % 2 == 0:
                        targets[last_eoa : i + 6] = -100
                    else:
                        last_eoa = i + 1
                    end_count += 1
                # # eos and following pad
                elif temp_id == 2:
                    # loss on eos, but not on pad
                    targets[i + 1 :] = -100
                    break
            # trunction, end at last question
            if temp_id != 2 and end_count % 2 == 0:
                # mask all after the last answer
                targets[last_eoa + 1 :] = -100
            target_batch.append(targets.unsqueeze(0))
        target_batch = torch.cat(target_batch, dim=0)
        return target_batch

    @add_start_docstrings_to_model_forward(InternLM2_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """

        samples = kwargs.get("samples", None)
        if samples:
            infer_mode = samples.get("infer_mode", "base")
            if samples["data_type"][0] == "text":
                has_img = False
            elif samples["data_type"][0] == "multi":
                has_img = True
            else:
                raise NotImplementedError

            # encode text
            text = samples["text_input"]
            # encode image
            if has_img:
                image = samples["image"]
                to_regress_embeds, attention_mask, targets, im_mask = (
                    self.interleav_wrap(image, text)
                )
            else:
                to_regress_tokens, targets = self.text2emb(text, add_special=True)
                to_regress_embeds = self.model.tok_embeddings(
                    to_regress_tokens.input_ids
                )
                attention_mask = to_regress_tokens.attention_mask
                im_mask = torch.zeros(to_regress_embeds.shape[:2]).cuda()

            inputs_embeds = to_regress_embeds[:, : self.max_length]
            attention_mask = attention_mask[:, : self.max_length]
            targets = targets[:, : self.max_length]
            im_mask = im_mask[:, : self.max_length].bool()
            labels = targets
        else:
            im_mask = kwargs.get("im_mask", None)
            infer_mode = kwargs.get("infer_mode", "base")
            if im_mask is None and inputs_embeds is not None:
                im_mask = torch.zeros(inputs_embeds.shape[:2]).to(inputs_embeds.device)
                im_mask = im_mask.bool()

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            im_mask=im_mask,
            infer_mode=infer_mode,
        )

        hidden_states = outputs[0]
        logits = self.output(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        im_mask=None,
        infer_mode="base",
        **kwargs,
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        im_mask = im_mask

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "im_mask": im_mask,
                "infer_mode": infer_mode,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past

    def build_inputs(
        self,
        tokenizer,
        query: str,
        history: List[Tuple[str, str]] = [],
        meta_instruction="",
    ):
        prompt = ""
        if meta_instruction:
            prompt += f"""<s>[UNUSED_TOKEN_146]system\n{meta_instruction}[UNUSED_TOKEN_145]\n"""
        else:
            prompt += "<s>"
        for record in history:
            prompt += f"""[UNUSED_TOKEN_146]user\n{record[0]}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n{record[1]}[UNUSED_TOKEN_145]\n"""
        prompt += f"""[UNUSED_TOKEN_146]user\n{query}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"""
        return tokenizer([prompt], return_tensors="pt")

    @torch.no_grad()
    def chat(
        self,
        tokenizer,
        query: str,
        image: List[Tuple[str, str]] = [],
        hd_num: int = 24,
        history: List[Tuple[str, str]] = [],
        streamer: Optional[BaseStreamer] = None,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_p: float = 0.8,
        repetition_penalty: float = 1.005,
        infer_mode: str = "base",
        use_meta: bool = False,
        meta_instruction: str = "You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n"
        "- InternLM-XComposer (浦语·灵笔) is a multi-modality conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
        "- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.\n"
        "- InternLM-XComposer (浦语·灵笔) is capable of comprehending and articulating responses effectively based on the provided image.",
        **kwargs,
    ):

        if not use_meta:
            meta_instruction = ""
        if image is None:
            inputs = self.build_inputs(tokenizer, query, history, meta_instruction)
            im_mask = torch.zeros(inputs["input_ids"].shape[:2]).cuda().bool()
        else:
            inputs, im_mask, _ = self.interleav_wrap_chat(
                query,
                image,
                history=history,
                meta_instruction=meta_instruction,
                hd_num=hd_num,
            )
        inputs = {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}
        # also add end-of-assistant token in eos token id to avoid unnecessary generation
        eos_token_id = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids(["[UNUSED_TOKEN_145]"])[0],
        ]
        outputs = self.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            im_mask=im_mask,
            infer_mode=infer_mode,
            **kwargs,
        )
        if image is None:
            outputs = outputs[0].cpu().tolist()[len(inputs["input_ids"][0]) :]
        else:
            outputs = outputs[0].cpu().tolist()
        response = tokenizer.decode(outputs, skip_special_tokens=True)
        response = response.split("[UNUSED_TOKEN_145]")[0]
        history = history + [(query, response)]
        return response, history

    @torch.no_grad()
    def write_artical(
        self,
        inst: str,
        image: List[Tuple[str, str]] = [],
        hd_num: int = 25,
        history: List[Tuple[str, str]] = [],
        streamer: Optional[BaseStreamer] = None,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_p: float = 0.8,
        repetition_penalty: float = 1.005,
        max_length: int = 8192,
        seed: int = -1,
        use_meta: bool = False,
        **kwargs,
    ):
        meta_instruction = """You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).
- InternLM-XComposer (浦语·灵笔) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""
        if seed != -1:
            set_seed(seed)
        if len(history):
            print(
                "Only chat function support multi round now, history will be ignored in the artical mode"
            )
        stop_words_ids = [92542]
        stopping_criteria = get_stopping_criteria(stop_words_ids)

        if not use_meta:
            meta_instruction = ""
        with torch.no_grad():
            inputs, im_mask, len_input_tokens = self.interleav_wrap_chat(
                inst, image, meta_instruction=meta_instruction, max_length=max_length
            )
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                generate = self.generate(
                    inputs_embeds=inputs["inputs_embeds"],
                    do_sample=do_sample,
                    num_beams=num_beams,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    stopping_criteria=stopping_criteria,
                    max_new_tokens=max_length - len_input_tokens,
                    top_p=0.8,
                    top_k=40,
                    length_penalty=1.0,
                    im_mask=im_mask,
                    infer_mode="write",
                )

        response = generate[0].tolist()
        response = self.tokenizer.decode(response, skip_special_tokens=True)
        # remove eoa
        response = response.replace("[UNUSED_TOKEN_145]", "")
        response = response.replace("[UNUSED_TOKEN_146]", "")

        return response

    @torch.no_grad()
    def write_webpage(
        self,
        inst: str,
        image: List[Tuple[str, str]] = [],
        max_new_tokens: int = 4800,
        do_sample: bool = True,
        num_beams: int = 2,
        temperature: float = 1.0,
        repetition_penalty: float = 3.0,
        seed: int = -1,
        use_meta: bool = False,
        task: str = "Instruction-aware Webpage Generation",
        **kwargs,
    ):

        if seed != -1:
            set_random_seed(seed, set_cudnn=True)
        with torch.no_grad():
            inputs, im_mask, len_input_tokens = self.interleav_wrap_chat(inst, image)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                generate = self.generate(
                    inputs_embeds=inputs["inputs_embeds"],
                    do_sample=do_sample,
                    temperature=temperature,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens,
                    im_mask=im_mask,
                    infer_mode="web",
                )
        response = generate[0].tolist()
        response = self.tokenizer.decode(response, skip_special_tokens=True)
        # remove eoa
        response = response.replace("[UNUSED_TOKEN_145]", "")
        out = response.replace("[UNUSED_TOKEN_146]", "")
        image_type = "random"
        pattern = r"""https://source\.unsplash\.com/random/(\d+)x(\d+)/\?([^'"]+)"""
        if image_type == "placeholder":
            out = re.sub(pattern, r"https://placehold.co/\1x\2", out)
        elif image_type == "random":
            out = re.sub(pattern, r"https://picsum.photos/\1/\2", out)

        with open(task.replace(" ", "_") + ".html", "w") as f:
            f.write(out)
        return out

    @torch.no_grad()
    def resume_2_webpage(
        self,
        inst: str,
        image: List[Tuple[str, str]] = [],
        max_new_tokens: int = 4800,
        do_sample: bool = True,
        num_beams: int = 2,
        temperature: float = 1.0,
        repetition_penalty: float = 3.0,
        seed: int = -1,
        use_meta: bool = False,
        task: str = "Resume-to-Personal Page",
        **kwargs,
    ):

        if seed != -1:
            set_random_seed(seed, set_cudnn=True)
        try:
            with open(inst) as fd:
                resume = fd.read()
        except:
            print("The input should be a resume with markdown format.")
        inst = " Generate a personal page using the content in the resume:" + resume
        with torch.no_grad():
            inputs, im_mask, len_input_tokens = self.interleav_wrap_chat(inst, image)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                generate = self.generate(
                    inputs_embeds=inputs["inputs_embeds"],
                    do_sample=do_sample,
                    temperature=temperature,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens,
                    im_mask=im_mask,
                    infer_mode="web",
                )
        response = generate[0].tolist()
        response = self.tokenizer.decode(response, skip_special_tokens=True)
        # remove eoa
        response = response.replace("[UNUSED_TOKEN_145]", "")
        html = response.replace("[UNUSED_TOKEN_146]", "")

        if seed != -1:
            set_random_seed(seed, set_cudnn=True)
        js_inst = " Generate JavaScript events for the html code:" + html
        with torch.no_grad():
            inputs, im_mask, len_input_tokens = self.interleav_wrap_chat(js_inst, image)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                generate = self.generate(
                    inputs_embeds=inputs["inputs_embeds"],
                    do_sample=do_sample,
                    temperature=temperature,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens,
                    im_mask=im_mask,
                    infer_mode="web",
                )
        response = generate[0].tolist()
        response = self.tokenizer.decode(response, skip_special_tokens=True)
        # remove eoa
        response = response.replace("[UNUSED_TOKEN_145]", "")
        js = response.replace("[UNUSED_TOKEN_146]", "")

        if re.search(r"</script>", html):
            js = re.findall(r"<script>([\s\S]*?)<\/script>", js)
            html = re.sub(r"(</script>)", f"\n{js}\n" + r"\1", html)
        elif re.search(r"</html>", html):
            html = re.sub(r"(</html>)", f"\n{js}\n" + r"\1", html)
        out = html

        image_type = "random"
        pattern = r"""https://source\.unsplash\.com/random/(\d+)x(\d+)/\?([^'"]+)"""
        if image_type == "placeholder":
            out = re.sub(pattern, r"https://placehold.co/\1x\2", out)
        elif image_type == "random":
            out = re.sub(pattern, r"https://picsum.photos/\1/\2", out)

        with open(task.replace(" ", "_") + ".html", "w") as f:
            f.write(out)
        return out

    @torch.no_grad()
    def screen_2_webpage(
        self,
        inst: str,
        image: List[Tuple[str, str]] = [],
        max_new_tokens: int = 4800,
        do_sample: bool = True,
        num_beams: int = 2,
        temperature: float = 1.0,
        repetition_penalty: float = 3.0,
        seed: int = -1,
        use_meta: bool = False,
        task: str = "Screenshot-to-Webpage",
        **kwargs,
    ):

        if seed != -1:
            set_random_seed(seed, set_cudnn=True)
        if len(image) == 0:
            print("No image is provided, skip")
            return ""
        inst = " Generate the HTML code of this web image with Tailwind CSS."
        with torch.no_grad():
            inputs, im_mask, len_input_tokens = self.interleav_wrap_chat(inst, image)

        with torch.autocast(device_type="cuda"):
            with torch.no_grad():
                generate = self.generate(
                    inputs_embeds=inputs["inputs_embeds"],
                    do_sample=do_sample,
                    temperature=temperature,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens,
                    im_mask=im_mask,
                    infer_mode="web",
                )
        response = generate[0].tolist()
        response = self.tokenizer.decode(response, skip_special_tokens=True)
        # remove eoa
        response = response.replace("[UNUSED_TOKEN_145]", "")
        out = response.replace("[UNUSED_TOKEN_146]", "")
        image_type = "random"
        pattern = r"""https://source\.unsplash\.com/random/(\d+)x(\d+)/\?([^'"]+)"""
        if image_type == "placeholder":
            out = re.sub(pattern, r"https://placehold.co/\1x\2", out)
        elif image_type == "random":
            out = re.sub(pattern, r"https://picsum.photos/\1/\2", out)

        with open(task.replace(" ", "_") + ".html", "w") as f:
            f.write(out)
        return out
