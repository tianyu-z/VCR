import json
import os
from transformers import AutoTokenizer, AutoModel
import torch
import torchvision.transforms as T
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from torchvision.transforms.functional import InterpolationMode
from PIL.Image import Image as type_image
from cambrian.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from cambrian.mm_utils import tokenizer_image_token, process_images
from cambrian.conversation import conv_templates

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def cambrian_process(
    image, question, tokenizer, image_processor, model_config, model_name
):
    if model_name == "cambrian-phi3-3b":
        conv_mode = "phi3"
    elif model_name == "cambrian-8b":
        conv_mode = "llama_3"
    elif model_name == "cambrian-34b":
        conv_mode = "chatml_direct"
    elif model_name == "cambrian-13b":
        conv_mode = "vicuna_v1"
    else:
        raise ValueError(f"Unsupported model name {model_name}")

    qs = question

    if model_config.mm_use_im_start_end:
        qs = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + qs
        )
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_size = [image.size]
    image_tensor = process_images([image], image_processor, model_config)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    return input_ids, image_tensor, image_size, prompt


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=6, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    if isinstance(image_file, str):  # image is a path
        image = Image.open(image_file).convert("RGB")
    elif isinstance(image_file, type_image):  # image is a PIL image
        image = image_file
    else:
        raise ValueError(f"Unsupported image type {type(image_file)}")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_all_json_files(folder_path):
    """
    Get all the JSON files under a specified folder.
    """
    json_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files


def find_json_filename_includes(folder_path, search_strings):
    """
    Find a JSON filename that includes all the specified strings.
    """
    json_files = get_all_json_files(folder_path)
    for file in json_files:
        if all(search_string in file for search_string in search_strings):
            return file
    return None


def read_json_into_dict(file_path):
    """
    Read the JSON file into a dictionary.
    """
    with open(file_path, "r") as f:
        return json.load(f)


def get_string_from_dict(data_dict, outer_key, inner_key):
    """
    Return the string given the keys like "5", "res_only_it_image".
    """
    try:
        return data_dict[outer_key][inner_key]
    except KeyError:
        return None


def rough_filter(model, language, answer_text):
    if model == "HuggingFaceM4/idefics2-8b" and "en" in language:
        return True
    elif model == "openbmb/MiniCPM-Llama3-V-2_5" and "en" in language:
        if "I can't" in answer_text:
            return False
        elif "I cannot" in answer_text:
            return False
        elif "sorry" in answer_text.lower():
            return False
        else:
            return True
    elif model == "openbmb/MiniCPM-Llama3-V-2_5" and "zh" in language:
        if "我无法" in answer_text:
            return False
        elif "抱歉" in answer_text:
            return False
        else:
            return True
    elif model == "internlm/internlm-xcomposer2-vl-7b" and "zh" in language:
        if "我无法" in answer_text:
            return False
        elif "抱歉" in answer_text:
            return False
        return True
    elif model == "internlm/internlm-xcomposer2-vl-7b" and "en" in language:
        if "I can't" in answer_text:
            return False
        elif "I cannot" in answer_text:
            return False
        elif "sorry" in answer_text.lower():
            return False
        else:
            return True
    elif model == "OpenGVLab/InternVL-Chat-V1-5" and "en" in language:
        if "I can't" in answer_text:
            return False
        elif "I cannot" in answer_text:
            return False
        elif "sorry" in answer_text.lower():
            return False
        else:
            return True
    elif model == "OpenGVLab/InternVL-Chat-V1-5" and "zh" in language:
        if "我无法" in answer_text:
            return False
        elif "抱歉" in answer_text:
            return False
        else:
            return True
    elif model.startswith("Qwen/Qwen-VL-Chat") and "en" in language:
        if "I can't" in answer_text:
            return False
        elif "I cannot" in answer_text:
            return False
        elif "sorry" in answer_text.lower():
            return False
        else:
            return True
    elif model.startswith("Qwen/Qwen-VL-Chat") and "zh" in language:
        if "抱歉" in answer_text:
            return False
    elif model.startswith("THUDM/cogvlm2-llama3-chat-19B") and "en" in language:
        if "I can't" in answer_text:
            return False
        elif "I cannot" in answer_text:
            return False
        else:
            return True
    elif model.startswith("THUDM/cogvlm2-llama3-chinese-chat-19B") and "zh" in language:
        if "无法" in answer_text:
            return False
        else:
            return True
    elif model == "Claude" and "en" in language:
        if "I can't" in answer_text:
            return False
        elif "I cannot" in answer_text:
            return False
        elif "sorry" in answer_text.lower():
            return False
        else:
            return True
    elif (
        model.lower() in ["gpt4o", "gpt", "gpt4", "gpt4v", "gpt-4-turbo"]
        or model == "Qwen-VL-Max"
        and "en" in language
    ):
        if "I can't" in answer_text:
            return False
        elif "I cannot" in answer_text:
            return False
        elif "sorry" in answer_text.lower():
            return False
        else:
            return True
    elif (
        model
        in [
            "01-ai_Yi-VL-6B",
            "01-ai_Yi-VL-34B",
            "deepseek-ai_deepseek-vl-1.3b-chat",
            "deepseek-ai_deepseek-vl-7b-chat",
        ]
        and "en" in language
    ):
        if "I can't" in answer_text:
            return False
        elif "I cannot" in answer_text:
            return False
        elif "sorry" in answer_text.lower():
            return False
        else:
            return True
    elif (
        model
        in [
            "01-ai_Yi-VL-6B",
            "01-ai_Yi-VL-34B",
            "deepseek-ai_deepseek-vl-1.3b-chat",
            "deepseek-ai_deepseek-vl-7b-chat",
            "Qwen-VL-Max",
        ]
        and "zh" in language
    ):
        if "无法" in answer_text:
            return False
        elif "抱歉" in answer_text:
            return False
        else:
            return True
    else:
        if "I can't" in answer_text:
            return False
        elif "I cannot" in answer_text:
            return False
        elif "sorry" in answer_text.lower():
            return False
        if "无法" in answer_text:
            return False
        elif "抱歉" in answer_text:
            return False
        else:
            return True


def zero_template(crossed_text):
    return {
        "crossed_text": crossed_text,
        "max_sim_val": 0,
        "max_sim_string": "",
        "precision": 0,
        "recall": 0,
        "f1": 0,
        "jaccard": 0,
        "rouge1": 0,
        "exact_match": 0,
    }


def matcher(string):
    return "_" + string + "_"
