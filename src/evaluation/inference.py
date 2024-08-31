import json
import math
import os

import torch
from PIL import Image
from PIL.Image import Image as type_image
from autopeftmodel import AutoPeftModelForCausalLMWithResizedWTE
from datasets import load_dataset
from fire import Fire
from tqdm import tqdm
from transformers import AutoModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoProcessor, AutoModelForVision2Seq
from utils import load_image as load_image_ext


def cover_substrings(sentence, substrings, language="en"):
    """
    Cover the substrings in the sentence with a placeholder.
    Not used in VCR paper but useful for debugging.

    Parameters:
    sentence (str): The sentence to be covered.
    substrings (list): The substrings to be covered.
    language (str): The language of the sentence. Default is "en".

    Returns:
    str: The covered sentence.
    """
    if language == "en":
        cover_prompt = "[covered]"
        starting_prompt = "Text context in image starts with:"
        ending_prompt = " Text context in image ends here. Question: "
    elif language == "zh":
        cover_prompt = "[被覆盖]"
        starting_prompt = "图片中的文字："
        ending_prompt = "图片中的文字结束。问题："
    for substring in substrings:
        sentence = sentence.replace(substring, cover_prompt)
    return starting_prompt + sentence + ending_prompt


def get_question(language, caption=None, crossed_texts=None):
    """
    Get the question for the the given language.

    Parameters:
    language (str): The language of the question.
    caption (str): The caption of the image. Default is None.
    crossed_texts (list): The crossed texts in the image. Default is None.

    Returns:
    str: The question.
    """
    if caption is not None:
        context = cover_substrings(caption, crossed_texts, language)
    else:
        context = ""
    if language == "en":
        return (
            context
            + "What is the covered texts in the image? Please restore the covered texts without outputting the explanations."
        )
    elif language == "zh":
        return (
            context
            + "图像中被覆盖的文本是什么？请在不输出解释的情况下还原被覆盖的文本。"
        )
    else:
        raise ValueError("Unsupported language")


def get_model(model_id, dtype, device=None, finetune_peft_path=None):
    """
    Get the model, tokenizer, and processor for the given model id.

    Parameters:
    model_id (str): The model id of HF.
    dtype (torch.dtype): The dtype of the model. Recommended to use torch.bfloat16.
    device (str): The device to use. Default is "cuda".
    finetune_peft_path (str): The path of the finetuned model if any. Default is None.

    Returns:
    tuple: The model, tokenizer, and processor.
    """
    if device is not None:
        device_map = None
    else:
        device_map = "auto"
    is_finetune = finetune_peft_path is not None
    if model_id in [
        "openbmb/MiniCPM-Llama3-V-2_5",
        "OpenGVLab/InternVL-Chat-V1-5",
    ]:
        if is_finetune:
            raise ValueError(f"Fine-tuning is not supported for {model_id}")
        model = AutoModel.from_pretrained(
            model_id, device_map=device_map, trust_remote_code=True, torch_dtype=dtype
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, device_map=device_map, trust_remote_code=True
        )
        model.eval()
        processor = None
    elif model_id in [
        "OpenGVLab/InternVL2-26B",
        "OpenGVLab/InternVL2-40B",
        "OpenGVLab/InternVL2-Llama3-76B",
    ]:

        def split_model(model_name):
            device_map = {}
            world_size = torch.cuda.device_count()
            num_layers = {
                "InternVL2-8B": 32,
                "InternVL2-26B": 48,
                "InternVL2-40B": 60,
                "InternVL2-Llama3-76B": 80,
            }[model_name]
            # Since the first GPU will be used for ViT, treat it as half a GPU.
            num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
            num_layers_per_gpu = [num_layers_per_gpu] * world_size
            num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
            layer_cnt = 0
            for i, num_layer in enumerate(num_layers_per_gpu):
                for j in range(num_layer):
                    device_map[f"language_model.model.layers.{layer_cnt}"] = i
                    layer_cnt += 1
            device_map["vision_model"] = 0
            device_map["mlp1"] = 0
            device_map["language_model.model.tok_embeddings"] = 0
            device_map["language_model.model.embed_tokens"] = 0
            device_map["language_model.output"] = 0
            device_map["language_model.model.norm"] = 0
            device_map["language_model.lm_head"] = 0
            device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

            return device_map

        device_map = split_model(model_id.split("/")[-1])
        if is_finetune:
            raise ValueError(f"Fine-tuning is not supported for {model_id}")
        model = AutoModel.from_pretrained(
            model_id, device_map=device_map, trust_remote_code=True, torch_dtype=dtype
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, device_map=device_map, trust_remote_code=True
        )
        model.eval()
        processor = None
    elif model_id in [
        "internlm/internlm-xcomposer2-vl-7b",
        "internlm/internlm-xcomposer2-4khd-7b",
        "internlm/internlm-xcomposer2d5-7b",
    ]:
        if is_finetune:
            raise ValueError(f"Fine-tuning is not supported for {model_id}")
        if model_id == "internlm/internlm-xcomposer2-vl-7b":
            from internlm2.modeling_internlm_xcomposer2 import (
                InternLMXComposer2ForCausalLM,
            )

            model = InternLMXComposer2ForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
        elif model_id == "internlm/internlm-xcomposer2-4khd-7b":
            from internlm2r4k.modeling_internlm_xcomposer2 import (
                InternLMXComposer2ForCausalLM,
            )

            model = InternLMXComposer2ForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
        elif model_id == "internlm/internlm-xcomposer2d5-7b":
            from internlm2d5.modeling_internlm_xcomposer2 import (
                InternLMXComposer2ForCausalLM,
            )

            model = InternLMXComposer2ForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                torch_dtype=dtype,
                trust_remote_code=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, device_map=device_map, trust_remote_code=True
        )
        model.eval()
        processor = None
    elif model_id == "HuggingFaceM4/idefics2-8b":
        if is_finetune:
            raise ValueError(f"Fine-tuning is not supported for {model_id}")
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, device_map=device_map, trust_remote_code=True, torch_dtype=dtype
        )
        tokenizer = None

    elif model_id in [
        "Qwen/Qwen-VL-Chat",
        "THUDM/cogvlm2-llama3-chinese-chat-19B",
        "THUDM/cogvlm2-llama3-chat-19B",
        "THUDM/glm-4v-9b",
    ]:
        if is_finetune:
            model = AutoPeftModelForCausalLMWithResizedWTE.from_pretrained(
                finetune_peft_path,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).eval()
        else:
            if "Qwen" in model_id:
                from QWen.modeling_qwen import QWenLMHeadModel

                model = QWenLMHeadModel.from_pretrained(
                    model_id,
                    device_map=device_map,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                ).eval()
            else:
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map=device_map,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                ).eval()
        # AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model.eval()
        processor = None
    elif model_id in ["THUDM/cogvlm-chat-hf"]:
        from transformers import AutoModelForCausalLM, LlamaTokenizer

        processor = None
        tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()

    elif model_id in ["echo840/Monkey-Chat"]:
        if is_finetune:
            raise ValueError(f"Fine-tuning is not supported for {model_id}")
        from text_monkey.modeling_monkey import MonkeyLMHeadModel

        model = MonkeyLMHeadModel.from_pretrained(
            model_id, device_map=device_map, trust_remote_code=True
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eod_id
        processor = None
    elif model_id in [
        "nyu-visionx/cambrian-34b",
        "nyu-visionx/cambrian-phi3-3b",
        "nyu-visionx/cambrian-8b",
        "nyu-visionx/cambrian-13b",
    ]:
        model_path = os.path.expanduser(model_id)
        from cambrian.mm_utils import (
            get_model_name_from_path,
        )
        from cambrian.model.builder import load_pretrained_model

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, processor, _ = load_pretrained_model(
            model_path, None, model_name, device_map=device_map
        )
    elif model_id in ["Qwen/Qwen2-VL-7B-Instruct"]:
        from transformers import (
            Qwen2VLForConditionalGeneration,
            AutoTokenizer,
            AutoProcessor,
        )

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = None
    elif model_id in ["microsoft/Phi-3.5-vision-instruct"]:
        from transformers import AutoModelForCausalLM
        from transformers import AutoProcessor

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=dtype,
            _attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, num_crops=4
        )
        tokenizer = None
    else:
        raise ValueError(f"Unsupported model {model_id}")
    return model, tokenizer, processor


def inference_with_image_path(
    model_id, image_paths, question, dtype, finetune_peft_path, max_tokens_len, device
):
    """
    Inference the model on the given image paths.

    Parameters:
    model_id (str): The model id of HF.
    image_paths (list): The paths of the images.
    question (str): The question to ask the model.
    dtype (torch.dtype): The dtype of the model. Recommended to use torch.bfloat16.
    finetune_peft_path (str): The path of the finetuned model if any. Default is None.
    max_tokens_len (int): The maximum tokens length. Default is None.

    Returns:
    dict: The inference results.
    """
    model, tokenizer, processor = get_model(model_id, dtype, finetune_peft_path)
    res = {}
    for image_id, image_path in enumerate(image_paths):
        res.update(
            inference_single(
                model_id,
                model,
                tokenizer,
                processor,
                image_path,
                image_id,
                question,
                dtype,
                max_tokens_len,
                device,
            )
        )
    return res


def inference_single(
    model_id,
    model,
    tokenizer,
    processor,
    image,
    image_id,
    question,
    dtype=torch.bfloat16,
    max_tokens_len=None,
    device=None,
):
    # todo: this script is different from the one on the github since I deleted all the device = "cuda" logic.
    # What we want to achieve: we need to load model in multiple (device_map = "auto") while making the data able to inference over it.
    # What I have now: RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:2 and cuda:0!
    # Clearly, the model and the data is not on the same device due to the device_map = "auto".
    """
    Inference the model on the given a single image.

    Parameters:
    model_id (str): The model id of HF.
    model (torch.nn.Module): The model.
    tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizer): The tokenizer.
    processor (transformers.processors.Processor): The processor.
    image (str): The path of the image.
    image_id (int): The id of the image.
    question (str): The question to ask the model.
    dtype (torch.dtype): The dtype of the model. Recommended to use torch.bfloat16.
    max_tokens_len (int): The maximum tokens length. Default is None.

    Returns:
    dict: The inference results only with the single image.
    """
    if device is not None:
        model = model.to(device)
    res = {}
    if max_tokens_len is None:
        max_tokens_len = 150
    if isinstance(image, str):  # image is a path
        image_id = image
        image = Image.open(image).convert("RGB")
    elif isinstance(image, type_image):  # image is a PIL image
        pass
    else:
        raise ValueError(f"Unsupported image type {type(image)}")

    if model_id == "openbmb/MiniCPM-Llama3-V-2_5":
        msgs = [{"role": "user", "content": question}]
        with torch.no_grad():
            res[image_id] = model.chat(
                image=image,
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=True,
                temperature=0.7,
                max_new_tokens=max_tokens_len,
            )
    elif model_id in [
        "OpenGVLab/InternVL-Chat-V1-5",
        "OpenGVLab/InternVL2-26B",
        "OpenGVLab/InternVL2-40B",
        "OpenGVLab/InternVL2-Llama3-76B",
    ]:
        pixel_values = load_image_ext(image, max_num=6).to(dtype).to(model.device)
        generation_config = dict(
            num_beams=1,
            max_new_tokens=max_tokens_len,
            do_sample=False,
        )
        with torch.no_grad():
            res[image_id] = model.chat(
                tokenizer, pixel_values, question, generation_config
            )
    elif model_id in [
        "internlm/internlm-xcomposer2-vl-7b",
        "internlm/internlm-xcomposer2-4khd-7b",
    ]:
        query = f"<ImageHere>{question}"
        with torch.no_grad():
            res[image_id], _ = model.chat(
                tokenizer,
                query=query,
                image=image,
                history=[],
                do_sample=False,
                max_new_tokens=max_tokens_len,
            )
    elif model_id in ["internlm/internlm-xcomposer2d5-7b"]:
        model.tokenizer = tokenizer
        res[image_id], _ = model.chat(
            tokenizer,
            question,
            [image],
            do_sample=False,
            num_beams=3,
            use_meta=True,
            max_new_tokens=max_tokens_len,
        )
    elif model_id == "HuggingFaceM4/idefics2-8b":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{question}"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_tokens_len)
            res[image_id] = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
    elif model_id in [
        "THUDM/cogvlm2-llama3-chat-19B",
        "THUDM/cogvlm2-llama3-chinese-chat-19B",
    ]:
        query = f"Human: {question}"
        history = []
        input_by_model = model.build_conversation_input_ids(
            tokenizer,
            query=query,
            history=history,
            images=[image],
            template_version="chat",
        )
        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(model.device),
            "token_type_ids": input_by_model["token_type_ids"]
            .unsqueeze(0)
            .to(model.device),
            "attention_mask": input_by_model["attention_mask"]
            .unsqueeze(0)
            .to(model.device),
            "images": (
                [[input_by_model["images"][0].to(dtype).to(model.device)]]
                if image is not None
                else None
            ),
        }
        gen_kwargs = {
            "max_new_tokens": max_tokens_len,
            "pad_token_id": 128002,
        }
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            response = tokenizer.decode(outputs[0])
            res[image_id] = response.split("<|end_of_text|>")[0]
    elif model_id == "THUDM/cogvlm-chat-hf":
        input_by_model = model.build_conversation_input_ids(
            tokenizer, query=question, history=[], images=[image]
        )  # chat mode
        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(model.device),
            "token_type_ids": input_by_model["token_type_ids"]
            .unsqueeze(0)
            .to(model.device),
            "attention_mask": input_by_model["attention_mask"]
            .unsqueeze(0)
            .to(model.device),
            "images": (
                [[input_by_model["images"][0].to(dtype).to(model.device)]]
                if image is not None
                else None
            ),
        }
        gen_kwargs = {"max_length": 2048, "do_sample": False}

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            res[image_id] = tokenizer.decode(outputs[0])
    elif model_id == "Qwen/Qwen-VL-Chat":
        query = tokenizer.from_list_format(
            [
                {"image": "tmp.png"},
                {"text": f"{question}"},
            ]
        )
        with torch.no_grad():
            response, history = model.chat(
                tokenizer,
                query=query,
                history=None,
                images=[image],
                max_new_tokens=max_tokens_len,
            )
            res[image_id] = response
    elif model_id == "echo840/Monkey-Chat":
        query = f"<img>tmp.jpg</img> {question} Answer: "

        input_ids = tokenizer(query, return_tensors="pt", padding="longest")
        attention_mask = input_ids.attention_mask.to(model.device)
        input_ids = input_ids.input_ids.to(model.device)

        pred = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_tokens_len,
            min_new_tokens=1,
            length_penalty=1,
            num_return_sequences=1,
            output_hidden_states=True,
            use_cache=True,
            pad_token_id=tokenizer.eod_id,
            eos_token_id=tokenizer.eod_id,
            images=[image],
        )
        response = tokenizer.decode(
            pred[0][input_ids.size(1) :].cpu(), skip_special_tokens=True
        ).strip()
        res[image_id] = response
    elif model_id == "THUDM/glm-4v-9b":
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "image": image, "content": question}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        gen_kwargs = {"max_length": max_tokens_len, "do_sample": True, "top_k": 1}
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            res[image_id] = tokenizer.decode(outputs[0])
    elif model_id in [
        "nyu-visionx/cambrian-34b",
        "nyu-visionx/cambrian-phi3-3b",
        "nyu-visionx/cambrian-8b",
        "nyu-visionx/cambrian-13b",
    ]:
        from utils import cambrian_process

        input_ids, image_tensor, image_sizes, prompt = cambrian_process(
            image,
            question,
            tokenizer,
            processor,
            model.config,
            model_name=model_id.split("/")[-1],
        )
        input_ids = input_ids.to(non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True if 0 > 0 else False,
                temperature=0,
                num_beams=1,
                max_new_tokens=max_tokens_len,
                use_cache=True,
            )
        res[image_id] = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()
    elif model_id in [
        "Qwen/Qwen2-VL-7B-Instruct",
    ]:
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        text_prompt = processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = processor(
            text=[text_prompt], images=[image], padding=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_tokens_len)
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            res[image_id] = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )[0]
    elif model_id in ["microsoft/Phi-3.5-vision-instruct"]:
        messages = [
            {"role": "user", "content": "<|image_1|>\n" + question},
        ]
        prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(prompt, [image], return_tensors="pt").to(model.device)
        generation_args = {
            "max_new_tokens": max_tokens_len,
            "temperature": 0.0,
            "do_sample": False,
        }
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                eos_token_id=processor.tokenizer.eos_token_id,
                **generation_args,
            )
            generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
            res[image_id] = processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
    else:
        raise ValueError(f"Unsupported model {model_id}")
    return res


def inference_single_pipeline(
    model_id="echo840/Monkey-Chat",
    image_paths=["main_pic_output.png_stacked_image.jpg"],
    language="en",
    dtype=torch.bfloat16,
    finetune_peft_path=None,
    max_tokens_len=None,
    device=None,
):
    """
    Inference the model on the given image paths and language type.

    Parameters:
    model_id (str): The model id of HF.
    image_paths (list): The paths of the images.
    language (str): The language of the question. Default is "en".
    dtype (torch.dtype): The dtype of the model. Recommended to use torch.bfloat16.
    finetune_peft_path (str): The path of the finetuned model if any. Default is None.
    max_tokens_len (int): The maximum tokens length. Default is None.

    Returns:
    dict: The inference results.
    """
    if language not in ["en", "zh"]:
        raise ValueError("Unsupported language")
    if model_id == "HuggingFaceM4/idefics2-8b":
        assert language == "en", "Only support English for this model"
    question = get_question(language)
    res = inference_with_image_path(
        model_id,
        image_paths,
        question,
        dtype,
        finetune_peft_path,
        max_tokens_len=max_tokens_len,
        device=device,
    )
    print(res)
    return res


def main(
    dataset_handler="vcr-org/VCR-wiki-en-hard-test",
    model_id="internlm/internlm-xcomposer2d5-7b",
    device="cuda",
    dtype="bf16",
    save_interval=5,  # Save progress every 100 images
    resume=True,  # Whether to resume from the last saved state
    finetune_peft_path=None,
    end_index=5000,
):
    """
    Inference the model on a given dataset.
    Note that the max tokens length is set to 2 times the length of the tokenize caption.

    Parameters:
    dataset_handler (str): The path of the dataset if local or HF dataset handler.
    model_id (str): The model id of HF.
    device (str): The device to use. Default is "cuda". If None, the device_map of the model will be set to "auto".
    dtype (torch.dtype): The dtype of the model. Recommended to use torch.bfloat16.
    save_interval (int): Save progress every save_interval images. Default is 50.
    resume (bool): Whether to resume from the last saved state. Default is True.
    finetune_peft_path (str): The path of the finetuned model if any. Default is None.
    end_index (int): The end index of the dataset. Default is None.

    Output:
    json: The inference results.
    """
    print(locals())
    if dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp32":
        dtype = torch.float32
    elif dtype == "fp16":
        dtype = torch.float16
    else:
        raise ValueError("Unsupported dtype")

    print("eval dataset: ", dataset_handler)
    if "en" in dataset_handler:
        language = "en"
    elif "zh" in dataset_handler:
        language = "zh"
    else:
        raise ValueError(f"Unsupported language")
    if "easy" in dataset_handler:
        difficulty = "easy"
    elif "hard" in dataset_handler:
        difficulty = "hard"
    else:
        raise ValueError(f"Unsupported difficulty")
    dataset = load_dataset(dataset_handler)["test"]
    if finetune_peft_path is not None:
        model_id_name = model_id.replace("/", "-")
        finetune_peft_path_name = finetune_peft_path.split("/")[-1]
        model_id_name = f"{model_id_name}_{finetune_peft_path_name}"
        print(f"Eval {finetune_peft_path}")
    else:
        model_id_name = model_id.replace("/", "-")
    if end_index is not None:
        output_file = f"{model_id_name}_{difficulty}_{language}_{end_index}.json"
        end_index_ = min(end_index, len(dataset))

    else:
        output_file = f"{model_id_name}_{language}_{difficulty}.json"
        end_index_ = len(dataset)
    print(f"Output file: {output_file}")

    if resume:
        try:
            with open(output_file, "r") as json_file:
                merged_dict = json.load(json_file)
            print(f"Resuming from {output_file}")
        except FileNotFoundError:
            print(f"No existing progress found, starting from scratch.")
            merged_dict = {}
    else:
        merged_dict = {}

    model, tokenizer, processor = get_model(model_id, dtype, device, finetune_peft_path)

    question = get_question(language)
    res_stacked_image = {}
    res_only_it_image = {}
    res_only_it_image_small = {}
    failed_image_ids = []

    start_index = len(merged_dict)

    for image_id in tqdm(range(start_index, end_index_)):
        stacked_image = dataset[image_id]["stacked_image"]
        only_it_image = dataset[image_id]["only_it_image"]
        only_it_image_small = dataset[image_id]["only_it_image_small"]
        try:
            toke = tokenizer.encode(dataset[image_id]["caption"])
            max_tokens_len = int(len(toke) * 2)
        except:
            max_tokens_len = 200  # default choice, change if needed
        try:
            res_stacked_image.update(
                inference_single(
                    model_id,
                    model,
                    tokenizer,
                    processor,
                    stacked_image,
                    str(image_id),
                    question,
                    dtype,
                    max_tokens_len,
                    device,
                )
            )
            res_stacked_image_success = True
        except Exception as e:
            print(f"Failed at image_id, res_stacked_image: {image_id}")
            failed_image_ids.append(image_id)
            print(e)
            res_stacked_image_success = False
        try:
            res_only_it_image.update(
                inference_single(
                    model_id,
                    model,
                    tokenizer,
                    processor,
                    only_it_image,
                    str(image_id),
                    question,
                    dtype,
                    max_tokens_len,
                    device,
                )
            )
            res_only_it_image_success = True
        except Exception as e:
            print(f"Failed at image_id, res_only_it_image: {image_id}")
            failed_image_ids.append(image_id)
            print(e)
            res_only_it_image_success = False
        try:
            res_only_it_image_small.update(
                inference_single(
                    model_id,
                    model,
                    tokenizer,
                    processor,
                    only_it_image_small,
                    str(image_id),
                    question,
                    dtype,
                    max_tokens_len,
                    device,
                )
            )
            res_only_it_image_small_success = True
        except Exception as e:
            print(f"Failed at image_id, res_only_it_image_small: {image_id}")
            failed_image_ids.append(image_id)
            print(e)
            res_only_it_image_small_success = False
        merged_dict[str(image_id)] = {
            "res_stacked_image": (
                res_stacked_image[str(image_id)] if res_stacked_image_success else ""
            ),
            "res_only_it_image": (
                res_only_it_image[str(image_id)] if res_only_it_image_success else ""
            ),
            "res_only_it_image_small": (
                res_only_it_image_small[str(image_id)]
                if res_only_it_image_small_success
                else ""
            ),
        }

        if (image_id + 1) % save_interval == 0:
            with open(output_file, "w", encoding="utf-8") as json_file:
                json.dump(merged_dict, json_file, indent=4, ensure_ascii=False)
            print(f"Progress saved at image_id: {image_id}")

    # Final save after the loop
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(merged_dict, json_file, indent=4, ensure_ascii=False)

    print(f"Failed image ids: {failed_image_ids}")
    return merged_dict, output_file


if __name__ == "__main__":
    Fire(main)
