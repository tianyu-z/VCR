from torchvision import transforms
from PIL import Image, ImageFont
from functools import partial
from utils import (
    mask_nouns,
    mask_sentence,
    mask_percentage,
    mask_ngram,
    split_and_cross_out_image,
)


class VCRTransform:
    def __init__(
        self,
        mode="easy",
        mask_mode="ngram",
        mask_p=0.5,
        n_gram=5,
        n_lines=5,
        language="en",
        font_path="arial.ttf",
        font_size=20,
        background_color="white",
        output_tensor=False,
    ):
        """
        Args:
            mode (str): 'easy' or 'hard' or None. If 'easy', the text will be crossed out in the middle of the image.
                        If 'hard', the text will be crossed out in the upper part of the image.
                        If None, the parameters mask_mode, mask_p, n_gram, n_lines, language, font_path,
                        font_size, background_color, output_tensor will be used.
            mask_mode (str): 'nouns' or 'sentence' or 'percentage' or 'ngram'.
            mask_p (float): The percentage of words to mask out.
            n_gram (int): The number of subwords to mask out.
            n_lines (int): The number of lines at most to split the text into.
            language (str): 'en' or 'zh'.
            font_path (str): The path to the font file. e.g. 'arial.ttf' for English or 'simsun.ttc' for 'Chinese'.
            font_size (int): The font size.
            background_color (str): The background color of the text image.
            output_tensor (bool): Whether to output tensor or PIL image.

        """
        self.language = language
        self.output_tensor = output_tensor
        if mode.lower() not in ["easy", "hard"]:
            self.mode = None
            self.mask_mode = mask_mode
            self.mask_p = mask_p
            self.n_gram = n_gram
            self.n_lines = n_lines
            self.font_path = font_path
            self.font_size = font_size
            self.background_color = background_color

            print("Warning: mode is not configured.")
        else:
            print(
                f"Using mode {mode} for language {self.language}. Note that configuration will be overwritten by mode {mode}."
            )
            self.mask_mode = "ngram"
            self.mask_p = 0.5
            self.n_gram = 5
            self.n_lines = 5
            self.font_size = 20
            self.background_color = "white"

            if self.language == "en":
                if mode.lower() == "easy":
                    self.lower_cross_height = 0.46
                    self.upper_cross_height = 0.77
                    self.font_path = "arial.ttf"

                elif mode.lower() == "hard":
                    self.lower_cross_height = 0.41
                    self.upper_cross_height = 0.80
                    self.font_path = "arial.ttf"

            elif self.language == "zh":
                if mode.lower() == "easy":
                    self.lower_cross_height = 0.38
                    self.upper_cross_height = 0.62
                    self.font_path = "simsun.ttc"

                elif mode.lower() == "hard":
                    self.lower_cross_height = 0.26
                    self.upper_cross_height = 0.75
                    self.font_path = "simsun.ttc"

            else:
                raise ValueError(
                    f"Language {self.language} is not currently supported."
                )

        assert self.mask_mode in [
            "nouns",
            "sentence",
            "percentage",
            "ngram",
        ], "Only 'nouns', 'sentence', 'percentage', ngram' are supported."
        config = {
            "mask_mode": self.mask_mode,
            "mask_p": self.mask_p,
            "n_gram": self.n_gram,
            "n_lines": self.n_lines,
            "language": self.language,
            "font_path": self.font_path,
            "font_size": self.font_size,
            "background_color": self.background_color,
            "output_tensor": self.output_tensor,
        }
        print(f"Using config: {config}")

        if self.mask_mode == "nouns":
            self.mask_func = partial(mask_nouns, language=self.language)
        elif self.mask_mode == "sentence":
            self.mask_func = partial(mask_sentence, language=self.language)
        elif self.mask_mode == "percentage":
            self.mask_func = partial(
                mask_percentage, mask_p=self.mask_p, language=self.language
            )
        else:  # mask_mode == 'ngram'
            self.mask_func = partial(
                mask_ngram, mask_p=self.mask_p, n=self.n_gram, language=self.language
            )

    def __call__(self, example):
        image = example["image"]
        text_in_image = example["caption"]
        if "crossed_text" in example:
            if example["crossed_text"] is not None:
                crossed_text = example["crossed_text"]
            else:
                crossed_text = None
        else:
            crossed_text = None

        width, height = image.size
        font = ImageFont.truetype(self.font_path, self.font_size)

        # Generate text_to_cross in case we only provided cross_text_func
        text_image, truncated_text, texts_to_cross = split_and_cross_out_image(
            text=text_in_image,
            font=font,
            max_width=width,
            lower_cross_height=self.lower_cross_height,
            upper_cross_height=self.upper_cross_height,
            texts_to_cross=crossed_text,
            cross_text_func=self.mask_func,
            background_color=self.background_color,
            language=self.language,
            n_lines=self.n_lines,
        )

        # Create stacked image (VI+TEI)
        stacked_image = Image.new(
            "RGB", (width, height + text_image.height + 1), "white"
        )
        stacked_image.paste(image, (0, 0))
        stacked_image.paste(text_image, (0, height))

        # Create only text image, with the same resolution as the stacked image
        stacked_width, stacked_height = stacked_image.size
        text_image_width, text_image_height = text_image.size
        text_only_image = Image.new("RGB", (stacked_width, stacked_height), "white")
        top_left_y = (stacked_height - text_image_height) // 2
        text_only_image.paste(text_image, (0, top_left_y))

        if self.output_tensor:
            transform = transforms.ToTensor()
            stacked_image = transform(stacked_image)
            text_only_image = transform(text_only_image)

        example["stacked_image"] = stacked_image
        example["only_it_image"] = text_only_image
        if crossed_text is None:
            example["crossed_text"] = texts_to_cross
        return example


if __name__ == "__main__":
    # English example
    example = {
        "image": Image.open("assets/main_pic.png"),
        "caption": "Machine learning researchers from around the globe are excited by the new GPU. Even if it is as large as a stovetop, its cutting-edge capabilities enable more efficient and cheaper large-scale experiments.",
        "crossed_text": [
            "learning researchers from around the",
            "cutting-edge capabilities enable more",
        ],
    }
    transform = VCRTransform(mode="easy", language="en")
    transformed_example = transform(example)
    transformed_example["stacked_image"].save("en_example_image.png")

    # Chinese example
    example = {
        "image": Image.open("assets/main_pic.png"),
        "caption": "来自全球各地的机器学习研究人员都对新型 GPU 感到兴奋。即使它只有炉灶那么大，其尖端功能也能让大规模实验更高效、更便宜。",
        "crossed_text": [
            "研究人员都对新型 GPU 感到",
            "即使它只有炉灶那么大",
            "尖端功能也能让大规模",
        ],
    }
    transform = VCRTransform(mode="easy", language="zh")
    transformed_example = transform(example)
    transformed_example["stacked_image"].save("zh_example_image.png")
