import random
from typing import Optional, List, Callable, Set, Union

import PIL
import datasets
import numpy as np
import spacy
import torch
from PIL import ImageDraw, ImageFont, PngImagePlugin

LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

try:
    nlp_en = spacy.load("en_core_web_sm")
except Exception as e:
    spacy.cli.download("en_core_web_sm")
    nlp_en = spacy.load("en_core_web_sm")

try:
    nlp_zh = spacy.load("zh_core_web_sm")
except Exception as e:
    spacy.cli.download("zh_core_web_sm")
    nlp_zh = spacy.load("zh_core_web_sm")

nlp = {"en": nlp_en, "zh": nlp_zh}


def mask_nouns(caption, language="en"):
    """Mask all the nouns in the caption."""
    doc = nlp[language](caption)
    caption_to_mask = [token.text for token in doc if token.pos_ == "NOUN"]
    return caption_to_mask


def mask_sentence(caption, language="en"):
    """Mask one sentence in the caption."""
    doc = nlp[language](caption)
    sentences = [sent.text for sent in doc.sents]
    return random.sample(sentences, 1)


def mask_percentage(caption, mask_p=0.5, language="en"):
    """Mask a percentage of the caption."""
    doc = nlp[language](caption)
    tokens = [token.text for token in doc]
    mask_num = int(len(tokens) * mask_p)
    mask_indices = random.sample(range(len(tokens)), mask_num)
    return [token for i, token in enumerate(tokens) if i not in mask_indices]


def mask_ngram(caption, mask_p=0.5, n=5, language="en"):
    """Mask n-grams in the caption that doesn't contain any punctuation, numbers, or named entities."""
    processed_doc = nlp[language](caption)

    ngrams = []
    splitter = "" if language == "zh" else " "
    for sentence in processed_doc.sents:
        words = [
            token.text
            for token in sentence
            if not token.is_punct
            and not token.like_num
            and not any(char.isdigit() for char in token.text)
            and not token.ent_type_
            in [
                "PERSON",
                "NORP",
                "FAC",
                "ORG",
                "LOC",
                "DATE",
                "TIME",
                "PERCENT",
                "MONEY",
                "QUANTITY",
                "CARDINAL",
            ]
        ]
        if len(words) < n:
            continue
        # Generate all possible ngrams in the sentence
        sentence_ngrams = [
            (i, splitter.join(words[i : i + n])) for i in range(len(words) - n + 1)
        ]

        # Shuffle ngrams to randomize selection
        random.shuffle(sentence_ngrams)

        selected_ngrams = []
        covered_indices = set()
        for ngram in sentence_ngrams:
            start_idx = ngram[0]
            ngram_text = ngram[1]
            if (
                start_idx in covered_indices
                or start_idx + n in covered_indices
                or ngram_text not in caption
            ):
                continue
            # Mark the indices of the current ngram as covered
            covered_indices.update(list(range(start_idx - 1, start_idx + n + 1)))
            selected_ngrams.append(ngram_text)
            if len(covered_indices) >= len(words) * mask_p:
                break
        ngrams.extend(selected_ngrams)
    return ngrams


def set_seed(seed: int):
    """Set seed for everything for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return


def filter_censor(example, censor: Union[Set[str], List[str], None] = None):
    """Filter out examples that contain any of the censor words."""
    if censor is None:
        return True
    for c in censor:
        if c in example["caption"]:
            return False
    return True


def filter_invalid_image(example):
    """Filter out examples with invalid images."""
    try:
        image = example["image"]
        image_ = datasets.Image().decode_example(
            image
        )  # Decode the image data to check if it's valid
        # Perform a simple operation that would fail if the image is corrupt
        _ = image_.getdata()[0]  # Access the first pixel
    except Exception as e:
        # Handle exceptions that indicate problems with the image
        print(f"Error processing image: {e}")
        return False
    return True


def create_text_image(
    text: str,
    font_path: str,
    font_size: int,
    text_color: str = "black",
    background_color: str = "white",
):
    """
    Create an image with the given text using the specified font.
    :param text: str. The text to render.
    :param font_path: str. The path to the font file.
    :param font_size: int. The size of the font.
    :param text_color: str. The color of the text.
    :param background_color: str. The background color of the text image (TEI).
    :return: image (PIL.Image, the TEI), font (ImageFont), text (str).
    """
    font = ImageFont.truetype(font_path, font_size)

    # Create an initial image to measure text dimensions
    temp_image = PIL.Image.new("RGB", (800, 200), background_color)
    draw = ImageDraw.Draw(temp_image)

    # Calculate text dimensions using the textbbox method
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Create the final image with the correct dimensions
    image = PIL.Image.new("RGB", (text_width, text_height), background_color)
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, font=font, fill=text_color)
    return image, font, text  # Now returning all three values


def split_and_cross_out_image(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    lower_cross_height: float,
    upper_cross_height: float,
    texts_to_cross: Optional[List[str]] = None,
    cross_text_func: Optional[Callable] = None,
    background_color: str = "white",
    language: str = "en",
    n_lines: int = 5,
):
    """
    Split text into lines and cross out specified texts. Return the final image with text and crossed-out texts.
    :param text: str. The caption text to split and cross out.
    :param font: ImageFont.FreeTypeFont. The font to use for the text.
    :param max_width: int. The maximum width of the text image.
    :param lower_cross_height: float. The lower bound of the cross-out line.
    :param upper_cross_height: float. The upper bound of the cross-out line.
    :param texts_to_cross: Optional[List[str]]. A list of texts to cross out. If None, use cross_text_func to generate.
    :param cross_text_func: Optional[Callable]. A function that takes the text and returns a list of texts to cross out.
    :param background_color: str. The background color of the text image.
    :param language: str. The language of the text. Currently only support ["en", "zh"].
    :param n_lines: int. The number of lines of captions to keep.
    :return: stacked_image (VI+TEI image), text_in_image (TEI image), texts_to_cross (List[str], crossed-out texts)
    """
    assert (
        texts_to_cross is not None or cross_text_func is not None
    ), "Either texts_to_cross or cross_text_func must be provided."

    if language == "en":
        sample_text = "hg"
        words = text.split()
        spliter = " "
        height_buffer = 0.2
    elif language == "zh":
        sample_text = "汉"
        words = list(text)
        spliter = ""
        height_buffer = 0.3
    else:
        raise ValueError(f"Language {language} is not supported.")

    current_line = ""
    lines = []
    temp_image = PIL.Image.new("RGB", (100, 100), background_color)
    temp_draw = ImageDraw.Draw(temp_image)

    # Split text into lines
    for word in words:
        test_line = f"{current_line}{spliter}{word}".strip()
        text_bbox = temp_draw.textbbox((0, 0), test_line, font=font)
        test_width = text_bbox[2] - text_bbox[0]
        if test_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    # Keep only the first n_lines
    if n_lines > 0:
        lines = lines[:n_lines]
    text_in_image = spliter.join(lines)

    # Generate texts to cross out
    if texts_to_cross is None:
        texts_to_cross = cross_text_func(text_in_image)

    # Calculate total height and prepare final image
    tmp = PIL.Image.new("RGB", (100, 100))
    drawtmp = ImageDraw.Draw(tmp)

    bboxtmp = drawtmp.textbbox((0, 0), sample_text, font=font)
    line_height = int((bboxtmp[3] - bboxtmp[1]) * (1 + height_buffer))

    stacked_height = line_height * len(lines)
    stacked_image = PIL.Image.new(
        "RGB", (max_width, stacked_height + 1), background_color
    )
    draw = ImageDraw.Draw(stacked_image)

    # Draw each line of text initially
    y = 0
    for line in lines:
        draw.text((0, y), line, font=font, fill="black")
        y += line_height

    # Cross out multiple texts
    for text_to_cross in texts_to_cross:
        if text_to_cross not in text_in_image:
            continue
        # Accumulate text to identify cross-out positions
        # full_text = spliter.join(lines)
        start_index = text_in_image.find(text_to_cross)
        end_index = start_index + len(text_to_cross)

        current_length = 0
        y = 0
        for i, line in enumerate(lines):
            line_start = current_length
            line_end = current_length + len(line)

            # Check if any part of the text_to_cross falls into the current line
            if (
                (line_start <= start_index < line_end)
                or (line_start < end_index <= line_end)
                or (start_index <= line_start and end_index >= line_end)
            ):
                cross_start = max(start_index, line_start) - line_start
                cross_end = min(end_index, line_end) - line_start
                cross_start_pos = draw.textbbox((0, 0), line[:cross_start], font=font)[
                    2
                ]
                cross_end_pos = draw.textbbox((0, 0), line[:cross_end], font=font)[2]
                draw.rectangle(
                    (
                        cross_start_pos,
                        y + lower_cross_height * line_height,
                        cross_end_pos,
                        y + upper_cross_height * line_height,
                    ),
                    fill=background_color,
                )
            current_length += len(line) + len(spliter)
            y += line_height

    return stacked_image, text_in_image, texts_to_cross
