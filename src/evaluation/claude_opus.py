import anthropic
import base64
import json
import os

YOUR_API_KEY = ""
PATH_OF_IMAGE = ""
client = anthropic.Anthropic(api_key=YOUR_API_KEY)
media_type = "image/png"  # Update this to match your image type (e.g., image/jpeg)

output = {
    str(i): {
        "res_stacked_image": "",
        "res_only_it_image": "",
        "res_only_it_image_small": "",
    }
    for i in range(500)
}


# end of using the example
# Function to read a local image and encode it to base64
def encode_image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string


for i in range(0, 500):

    # Path to the local image
    image1_path = os.path.join(PATH_OF_IMAGE, f"stacked_image_{i}.png")

    image1_data = encode_image_to_base64(image1_path)
    image2_path = os.path.join(PATH_OF_IMAGE, f"only_it_image_{i}.png")

    image2_data = encode_image_to_base64(image2_path)
    image3_path = os.path.join(PATH_OF_IMAGE, f"only_it_image_small_{i}.png")

    image3_data = encode_image_to_base64(image3_path)
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
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
                        "text": "What is the covered texts in the image? Please restore the covered texts without outputting the explanations.",
                    },
                ],
            }
        ],
    )

    output[str(i)]["res_stacked_image"] = message.dict()["content"][0]["text"]
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
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
                        "text": "What is the covered texts in the image? Please restore the covered texts without outputting the explanations.",
                    },
                ],
            }
        ],
    )
    output[str(i)]["res_only_it_image"] = message.dict()["content"][0]["text"]
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image3_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": "What is the covered texts in the image? Please restore the covered texts without outputting the explanations.",
                    },
                ],
            }
        ],
    )
    output[str(i)]["res_only_it_image_small"] = message.dict()["content"][0]["text"]

with open("Claude_en_500_evaluation_result.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)
