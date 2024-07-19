import base64
import json
import os
from tqdm import trange
import argparse
import requests
import httpx
from time import sleep


def encode_image_to_base64(image_path):
    # if image_path is not an url
    if not image_path.startswith("http"):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    else:
        encoded_string = base64.b64encode(httpx.get(image_path).content).decode("utf-8")

    return encoded_string


def resume_init(args):
    name_data = args.dataset_handler.replace("/", "_").replace("-", "_")
    file_name = f"{args.model_id}_{name_data}_inference.json"
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            output = json.load(f)
        # reverse the for loop of dictionary
        keys = list(output.keys())
        values = list(output.values())
        start = 0
    else:
        output = {
            str(i): {
                "res_stacked_image": "",
                "res_only_it_image": "",
            }
            for i in range(args.end)
        }
        start = 0
    return output, start, file_name


def get_question(language):
    if language == "en":
        question = "What is the covered texts in the image? Please restore the covered texts without outputting the explanations."
    elif language == "zh":
        question = "图像中被覆盖的文本是什么？请在不输出解释的情况下还原被覆盖的文本。"
    else:
        raise ValueError(f"Language {language} not supported.")
    return question


def claude(output, start, question, file_name, image_path, model_id, end):
    try:
        import anthropic
    except ImportError:
        print("Please install anthropic first. Run `pip install anthropic`.")

    media_type = "image/png"
    client = anthropic.Anthropic(api_key=YOUR_API_KEY)
    for i in trange(start, end):
        image1_path = os.path.join(image_path, f"stacked_image_{i}.png")
        image1_data = encode_image_to_base64(image1_path)
        image2_path = os.path.join(image_path, f"only_it_image_{i}.png")
        image2_data = encode_image_to_base64(image2_path)
        try:
            if output[str(i)]["res_stacked_image"] == "":
                message = client.messages.create(
                    model=model_id,
                    max_tokens=256,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": image1_data,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": question,
                                },
                            ],
                        }
                    ],
                )
                output[str(i)]["res_stacked_image"] = message.dict()["content"][0][
                    "text"
                ]
            if output[str(i)]["res_only_it_image"] == "":
                message = client.messages.create(
                    model=model_id,
                    max_tokens=256,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": image2_data,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": question,
                                },
                            ],
                        }
                    ],
                )
                output[str(i)]["res_only_it_image"] = message.dict()["content"][0][
                    "text"
                ]

        except Exception as e:
            print(e)
            print(f"Error at {i}")
            continue
        if i % 10 == 0:
            with open(file_name, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=4)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    return output


def gpt(output, start, question, file_name, image_path, model_id, end):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {YOUR_API_KEY}",
    }
    for i in trange(start, end):
        image1_path = os.path.join(image_path, f"stacked_image_{i}.png")
        image1_data = encode_image_to_base64(image1_path)
        image2_path = os.path.join(image_path, f"only_it_image_{i}.png")
        image2_data = encode_image_to_base64(image2_path)
        try:
            if output[str(i)]["res_stacked_image"] == "":
                payload = {
                    "model": model_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": question,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image1_data}",
                                    },
                                },
                            ],
                        }
                    ],
                    "max_tokens": 256,
                }
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                output[str(i)]["res_stacked_image"] = response.json()["choices"][0][
                    "message"
                ]["content"]
                sleep(1)
            if output[str(i)]["res_only_it_image"] == "":
                payload = {
                    "model": model_id,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": question,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image2_data}",
                                    },
                                },
                            ],
                        }
                    ],
                    "max_tokens": 256,
                }
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                output[str(i)]["res_only_it_image"] = response.json()["choices"][0][
                    "message"
                ]["content"]
                sleep(1)
        except Exception as e:
            print(e)
            print(f"Error at {i}")
            continue
        if i % 10 == 0:
            with open(file_name, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=4)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    return output


def gemini(output, start, question, file_name, image_path, model_id, end):
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Please install python-dotenv first. Run `pip install python-dotenv`.")
    try:
        import google.generativeai as genai
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
    except ImportError:
        print(
            "Please install google-generativeai first. Run `pip install google-generativeai`."
        )
    try:
        import PIL.Image
    except ImportError:
        print("Please install pillow first. Run `pip install pillow`.")

    load_dotenv()
    genai.configure(api_key=os.getenv("GENAI_API_KEY"))
    NO_SAFETY = {
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    model = genai.GenerativeModel(model_id, safety_settings=NO_SAFETY)
    for i in trange(start, end):
        image1_path = os.path.join(image_path, f"stacked_image_{i}.png")
        image2_path = os.path.join(image_path, f"only_it_image_{i}.png")
        if image1_path.startswith("http"):
            image1_data = PIL.Image.open(httpx.get(image1_path))
        else:
            image1_data = PIL.Image.open(image1_path)
        if image2_path.startswith("http"):
            image2_data = PIL.Image.open(httpx.get(image2_path))
        else:
            image2_data = PIL.Image.open(image2_path)
        try:
            if output[str(i)]["res_stacked_image"] == "":
                response_1 = model.generate_content(
                    [question, image1_data], stream=False
                )
                output[str(i)]["res_stacked_image"] = response_1.text
            if output[str(i)]["res_only_it_image"] == "":
                response_2 = model.generate_content(
                    [question, image2_data], stream=False
                )
                output[str(i)]["res_only_it_image"] = response_2.text
        except Exception as e:
            print(e)
            print(f"Error at {i}")
            continue
        if i % 10 == 0:
            with open(file_name, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=4)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    return output


def reka(output, start, question, file_name, image_path, model_id, end):
    try:
        from reka.client import Reka
        from reka import ChatMessage
    except ImportError:
        print("Please install reka first. Run `pip install reka-api>=2.0.0`.")

    client = Reka(
        api_key=YOUR_API_KEY,
    )
    if not image_path.startswith("http"):
        raise ValueError("image_path must be a url for Reka.")
    for i in trange(start, end):
        try:
            if output[str(i)]["res_stacked_image"] == "":
                response = client.chat.create(
                    messages=[
                        ChatMessage(
                            content=[
                                {
                                    "type": "image_url",
                                    "image_url": f"{image_path}/stacked_image_{i}.png",
                                },
                                {"type": "text", "text": question},
                            ],
                            role="user",
                        )
                    ],
                    model=model_id,
                )
                output[str(i)]["res_stacked_image"] = response.responses[
                    0
                ].message.content
            if output[str(i)]["res_only_it_image"] == "":
                response = client.chat.create(
                    messages=[
                        ChatMessage(
                            content=[
                                {
                                    "type": "image_url",
                                    "image_url": f"{image_path}/only_it_image_{i}.png",
                                },
                                {"type": "text", "text": question},
                            ],
                            role="user",
                        )
                    ],
                    model=model_id,
                )
                output[str(i)]["res_only_it_image"] = response.responses[
                    0
                ].message.content
        except Exception as e:
            print(e)
            print(f"Error at {i}")
            continue
        if i % 10 == 0:
            with open(file_name, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=4)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    return output


def Qwen(output, start, question, file_name, image_path, model_id, end):
    from http import HTTPStatus

    try:
        import dashscope
    except ImportError:
        print("Please install dashscope first. Run `pip install dashscope`.")

    dashscope.api_key = YOUR_API_KEY
    if not image_path.startswith("http"):
        raise ValueError("image_path must be a url for Qwen.")
    for i in trange(start, end):
        try:
            if output[str(i)]["res_stacked_image"] == "":
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": f"{image_path}/stacked_image_{i}.png",
                            },
                            {"text": question},
                        ],
                    }
                ]
                response = dashscope.MultiModalConversation.call(
                    model=model_id, messages=messages, top_k=1
                )
                if response.status_code == HTTPStatus.OK:
                    output[str(i)]["res_stacked_image"] = response.output.choices[
                        0
                    ].message.content[0]["text"]
                else:
                    print(response.code)
                    print(response.message)
            if output[str(i)]["res_only_it_image"] == "":
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "image": f"{image_path}/only_it_image_{i}.png",
                            },
                            {"text": question},
                        ],
                    }
                ]
                response = dashscope.MultiModalConversation.call(
                    model=model_id, messages=messages, top_k=1
                )
                if response.status_code == HTTPStatus.OK:
                    output[str(i)]["res_only_it_image"] = response.output.choices[
                        0
                    ].message.content[0]["text"]
                else:
                    print(response.code)
                    print(response.message)
        except Exception as e:
            print(e)
            print(f"Error at {i}")
            continue
        if i % 10 == 0:
            with open(file_name, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=4)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    return output


def main(args):
    output, start, file_name = resume_init(args)
    language = "en" if "en" in args.dataset_handler else "zh"
    question = get_question(language)
    if args.image_path.startswith("https://raw.githubusercontent.com/tianyu-z/"):
        image_path = args.image_path + args.dataset_handler + "/main/raw/"
    elif args.image_path.startswith("http"):
        raise NotImplementedError(
            "Please Implement your own logic for non-default image url."
        )
    else:
        # image_path = args.image_path + args.dataset_handler
        image_path = args.image_path
    if "claude" in args.model_id:
        output = claude(
            output, start, question, file_name, image_path, args.model_id, args.end
        )
    elif "gpt" in args.model_id:
        output = gpt(
            output, start, question, file_name, image_path, args.model_id, args.end
        )
    elif "qwen-vl" in args.model_id:
        output = Qwen(
            output, start, question, file_name, image_path, args.model_id, args.end
        )
    elif "reka" in args.model_id:
        output = reka(
            output, start, question, file_name, image_path, args.model_id, args.end
        )
    elif "gemini" in args.model_id:
        output = gemini(
            output, start, question, file_name, image_path, args.model_id, args.end
        )
    else:
        print(f"model_id {args.model_id} not supported.")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation pipeline of VCR.")
    parser.add_argument(
        "--model_id",
        type=str,
        help="model name of close_source_model",
        required=True,
    )

    parser.add_argument(
        "--dataset_handler",
        type=str,
        help="dataset handler",
        required=True,
    )

    parser.add_argument(
        "--image_path",
        type=str,
        help="image path, can be a local path or a url",
        default="https://raw.githubusercontent.com/tianyu-z/",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="api key for the model",
        required=True,
    )
    parser.add_argument(
        "--end",
        type=int,
        help="end index of the dataset",
        default=500,
    )
    supported_model_ids = [
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "gpt-4o",
        "gpt-4-turbo",
        "qwen-vl-max",
        "reka-core-20240501",
        "gemini-1.5-pro-latest",
        "gpt-4o-mini-2024-07-18",  # gpt-4o-mini
    ]
    supported_dataset_handlers = [
        "VCR-wiki-en-easy-test-500",
        "VCR-wiki-en-hard-test-500",
        "VCR-wiki-zh-easy-test-500",
        "VCR-wiki-zh-hard-test-500",
    ]
    args = parser.parse_args()
    if args.model_id not in supported_model_ids:
        print(
            f"model_id not in {supported_model_ids}, please double check the code to make sure the model_id fits the code before you run it."
        )

    if args.dataset_handler not in supported_dataset_handlers:
        print(
            f"dataset_handler not in {supported_dataset_handlers}, please double check the code to make sure the dataset_handler fits the code before you run it."
        )
    YOUR_API_KEY = args.api_key
    main(args)

    # # test case
    # YOUR_API_KEY = "test_api_key"
    # output = {
    #     str(i): {
    #         "res_stacked_image": "",
    #         "res_only_it_image": "",
    #     }
    #     for i in range(500)
    # }
    # start = 499
    # file_name = "test.json"
    # gemini(
    #     output,
    #     start,
    #     "What is the covered texts in the image? Please restore the covered texts without outputting the explanations.",
    #     file_name,
    #     "https://raw.githubusercontent.com/tianyu-z/VCR-wiki-zh-easy-test-500/main/raw/",
    #     "gemini-1.5-pro-latest",
    # )
