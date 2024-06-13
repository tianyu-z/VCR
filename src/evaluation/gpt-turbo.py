import base64
import requests
import json
import os

YOUR_API_KEY = ""
PATH_OF_IMAGE = ""
# OpenAI API Key

output = {
    str(i): {
        "res_stacked_image": "",
        "res_only_it_image": "",
        "res_only_it_image_small": "",
    }
    for i in range(500)
}


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


for i in [0, 500]:
    # Path to your image
    image1_path = os.path.join(PATH_OF_IMAGE, f"stacked_image_{i}.png")

    # Getting the base64 string
    base64_image1 = encode_image(image1_path)
    image2_path = os.path.join(PATH_OF_IMAGE, f"only_it_image_{i}.png")
    base64_image2 = encode_image(image2_path)
    image3_path = os.path.join(PATH_OF_IMAGE, f"only_it_image_small_{i}.png")
    base64_image3 = encode_image(image3_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {YOUR_API_KEY}",
    }

    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the covered texts in the image? Please restore the covered texts without outputting the explanations.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image1}"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )

    output[str(i)]["res_stacked_image"] = response.json()["choices"][0]["message"][
        "content"
    ]

    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the covered texts in the image? Please restore the covered texts without outputting the explanations.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image2}"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    output[str(i)]["res_only_it_image"] = response.json()["choices"][0]["message"][
        "content"
    ]

    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is the covered texts in the image? Please restore the covered texts without outputting the explanations.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image3}"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    output[str(i)]["res_only_it_image_small"] = response.json()["choices"][0][
        "message"
    ]["content"]

with open("gpt-4-turbo_en_500_evaluation_result.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)
