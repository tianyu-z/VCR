# test.py
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from utils import load_image as load_image_ext
import torch
from PIL import Image
import os
from transformers import AutoProcessor, AutoModelForVision2Seq

from transformers import AutoModelForCausalLM, AutoTokenizer

from fire import Fire
from datasets import load_from_disk, load_dataset
from PIL.Image import Image as type_image
import json
from tqdm import tqdm
from autopeftmodel import AutoPeftModelForCausalLMWithResizedWTE


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


def get_model(model_id, device, dtype, finetune_peft_path=None):
    """
    Get the model, tokenizer, and processor for the given model id.

    Parameters:
    model_id (str): The model id of HF.
    device (str): The device to run the model. Recommended to use "cuda".
    dtype (torch.dtype): The dtype of the model. Recommended to use torch.bfloat16.
    finetune_peft_path (str): The path of the finetuned model if any. Default is None.

    Returns:
    tuple: The model, tokenizer, and processor.
    """
    is_finetune = finetune_peft_path is not None
    if model_id in [
        "openbmb/MiniCPM-Llama3-V-2_5",
        "OpenGVLab/InternVL-Chat-V1-5",
        "OpenGVLab/InternVL2-26B"
    ]:

        if is_finetune:
            raise ValueError(f"Fine-tuning is not supported for {model_id}")
        model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=dtype
        )
        model = model.to(device=device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model.eval()
        processor = None
    elif model_id in ["internlm/internlm-xcomposer2-vl-7b", "internlm/internlm-xcomposer2-4khd-7b", "internlm/internlm-xcomposer2d5-7b"]:
        if is_finetune:
            raise ValueError(f"Fine-tuning is not supported for {model_id}")
        from internlm.modeling_internlm_xcomposer2 import (
            InternLMXComposer2ForCausalLM,
        )

        model = InternLMXComposer2ForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=dtype
        )
        model = model.to(device=device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model.eval()
        processor = None
    elif model_id == "HuggingFaceM4/idefics2-8b":
        if is_finetune:
            raise ValueError(f"Fine-tuning is not supported for {model_id}")
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=dtype
        ).to(device=device)
        tokenizer = None

    elif model_id in [
        "Qwen/Qwen-VL-Chat",
        "THUDM/cogvlm2-llama3-chinese-chat-19B",
        "THUDM/cogvlm2-llama3-chat-19B",
        "THUDM/glm-4v-9b",
    ]:
        if is_finetune:
            model = AutoPeftModelForCausalLMWithResizedWTE.from_pretrained(
                finetune_peft_path, trust_remote_code=True, torch_dtype=dtype
            ).eval()
        else:
            if "Qwen" in model_id:
                from QWen.modeling_qwen import QWenLMHeadModel

                model = QWenLMHeadModel.from_pretrained(
                    model_id, trust_remote_code=True, torch_dtype=dtype
                ).eval()
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map=device,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                ).eval()
        # AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = model.to(device=device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model.eval()
        processor = None
    elif model_id in ["echo840/Monkey-Chat"]:
        if is_finetune:
            raise ValueError(f"Fine-tuning is not supported for {model_id}")
        from text_monkey.modeling_monkey import MonkeyLMHeadModel

        model = MonkeyLMHeadModel.from_pretrained(
            model_id, device_map="cuda", trust_remote_code=True
        ).eval()
        # model = model.to(device=device)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eod_id
        processor = None
    elif model_id == "nyu-visionx/cambrian-34b":
        model_path = os.path.expanduser("nyu-visionx/cambrian-34b")
        # model_path = "src/evaluation/cambrian-34b"
        from cambrian.mm_utils import (
            get_model_name_from_path,
        )
        from cambrian.model.builder import load_pretrained_model

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, processor, _ = load_pretrained_model(
            model_path, None, model_name
        )
    else:
        raise ValueError(f"Unsupported model {model_id}")
    return model, tokenizer, processor


def inference_with_image_path(
    model_id, image_paths, question, device, dtype, finetune_peft_path, max_tokens_len
):
    """
    Inference the model on the given image paths.

    Parameters:
    model_id (str): The model id of HF.
    image_paths (list): The paths of the images.
    question (str): The question to ask the model.
    device (str): The device to run the model. Recommended to use "cuda".
    dtype (torch.dtype): The dtype of the model. Recommended to use torch.bfloat16.
    finetune_peft_path (str): The path of the finetuned model if any. Default is None.
    max_tokens_len (int): The maximum tokens length. Default is None.

    Returns:
    dict: The inference results.
    """
    model, tokenizer, processor = get_model(model_id, device, dtype, finetune_peft_path)
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
                device,
                dtype,
                max_tokens_len,
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
    device,
    dtype,
    max_tokens_len,
):
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
    device (str): The device to run the model. Recommended to use "cuda".
    dtype (torch.dtype): The dtype of the model. Recommended to use torch.bfloat16.
    max_tokens_len (int): The maximum tokens length. Default is None.

    Returns:
    dict: The inference results only with the single image.
    """
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
    elif model_id in ["OpenGVLab/InternVL-Chat-V1-5", "OpenGVLab/InternVL2-26B"]:
        pixel_values = load_image_ext(image, max_num=6).to(dtype).cuda()
        generation_config = dict(
            num_beams=1,
            max_new_tokens=max_tokens_len,
            do_sample=False,
        )
        with torch.no_grad():
            res[image_id] = model.chat(
                tokenizer, pixel_values, question, generation_config
            )
    elif model_id in ["internlm/internlm-xcomposer2-vl-7b", "internlm/internlm-xcomposer2-4khd-7b"]:
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
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            res[image_id], _ = model.chat(tokenizer, question, [image], do_sample=False, num_beams=3, use_meta=True)
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
        inputs = {k: v.to(device) for k, v in inputs.items()}
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
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(device),
            "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to(device),
            "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to(device),
            "images": (
                [[input_by_model["images"][0].to(device).to(dtype)]]
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
        attention_mask = input_ids.attention_mask
        input_ids = input_ids.input_ids

        pred = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
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
        ).to(
            device
        )  # chat mode
        gen_kwargs = {"max_length": max_tokens_len, "do_sample": True, "top_k": 1}
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            res[image_id] = tokenizer.decode(outputs[0])
    elif model_id == "nyu-visionx/cambrian-34b":
        from utils import cambrian_process

        input_ids, image_tensor, image_sizes, prompt = cambrian_process(
            image,
            question,
            tokenizer,
            processor,
            model.config,
            model_name=model_id.split("/")[-1],
        )
        input_ids = input_ids.to(device=device, non_blocking=True)
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
    else:
        raise ValueError(f"Unsupported model {model_id}")
    return res


def inference_single_pipeline(
    model_id="echo840/Monkey-Chat",
    image_paths=["main_pic_output.png_stacked_image.jpg"],
    device="cuda",
    language="en",
    dtype=torch.bfloat16,
    finetune_peft_path=None,
    max_tokens_len=None,
):
    """
    Inference the model on the given image paths and language type.

    Parameters:
    model_id (str): The model id of HF.
    image_paths (list): The paths of the images.
    device (str): The device to run the model. Recommended to use "cuda".
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
        device,
        dtype,
        finetune_peft_path,
        max_tokens_len=max_tokens_len,
    )
    print(res)
    return res


def main(
    dataset_handler="vcr-org/VCR-wiki-en-hard-test",
    model_id="THUDM/cogvlm2-llama3-chat-19B",
    device="cuda",
    dtype="bf16",
    save_interval=5,  # Save progress every 100 images
    resume=True,  # Whether to resume from the last saved state
    finetune_peft_path=None,
    end_index=None,
):
    """
    Inference the model on a given dataset.
    Note that the max tokens length is set to 2 times the length of the tokenize caption.

    Parameters:
    dataset_handler (str): The path of the dataset if local or HF dataset handler.
    model_id (str): The model id of HF.
    device (str): The device to run the model. Recommended to use "cuda".
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

    if finetune_peft_path is not None:
        model_id_name = model_id.replace("/", "-")
        finetune_peft_path_name = finetune_peft_path.split("/")[-1]
        model_id_name = f"{model_id_name}_{finetune_peft_path_name}"
        print(f"Eval {finetune_peft_path}")
    else:
        model_id_name = model_id.replace("/", "-")
    if end_index is not None:
        output_file = f"{model_id_name}_{difficulty}_{language}_{end_index}.json"
    else:
        output_file = f"{model_id_name}_{difficulty}_{language}.json"
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

    dataset = load_dataset(dataset_handler)["test"]

    model, tokenizer, processor = get_model(model_id, device, dtype, finetune_peft_path)

    question = get_question(language)
    res_stacked_image = {}
    res_only_it_image = {}
    res_only_it_image_small = {}
    failed_image_ids = []

    start_index = len(merged_dict)

    for image_id in tqdm(range(start_index, min(end_index, len(dataset)))):
        stacked_image = dataset[image_id]["stacked_image"]
        only_it_image = dataset[image_id]["only_it_image"]
        only_it_image_small = dataset[image_id]["only_it_image_small"]
        toke = tokenizer.encode(dataset[image_id]["caption"])
        max_tokens_len = int(len(toke) * 2)
        res_stacked_image.update(
            inference_single(
                model_id,
                model,
                tokenizer,
                processor,
                stacked_image,
                str(image_id),
                question,
                device,
                dtype,
                max_tokens_len,
            )
        )
        res_only_it_image.update(
            inference_single(
                model_id,
                model,
                tokenizer,
                processor,
                only_it_image,
                str(image_id),
                question,
                device,
                dtype,
                max_tokens_len,
            )
        )
        res_only_it_image_small.update(
            inference_single(
                model_id,
                model,
                tokenizer,
                processor,
                only_it_image_small,
                str(image_id),
                question,
                device,
                dtype,
                max_tokens_len,
            )
        )

        merged_dict[str(image_id)] = {
            "res_stacked_image": res_stacked_image[str(image_id)],
            "res_only_it_image": res_only_it_image[str(image_id)],
            "res_only_it_image_small": res_only_it_image_small[str(image_id)],
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

