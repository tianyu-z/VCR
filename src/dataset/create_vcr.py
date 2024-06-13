import os
from functools import partial
from random import sample
from typing import Optional, List

import fire
import torchvision.transforms as transforms
from PIL import Image, ImageFont
from datasets import load_dataset, load_from_disk, Image, Dataset

from src.dataset.utils import split_and_cross_out_image
from utils import set_seed, filter_censor, filter_invalid_image, mask_nouns, mask_sentence, mask_percentage, mask_ngram


def generate_vcr_single(
        example,
        mask_mode: str = "ngram",
        mask_p: float = 0.5,
        n_gram: int = 5,
        n_lines: int = 5,
        language: str = "en",
        easy_mode: bool = False,
        font_size: int = 20,
        texts_to_cross: Optional[List[str]] = None,
        background_color: str = "white",
        save_image: bool = False,
        save_image_name: Optional[str] = None,
        output_tensor: bool = False,
):
    """
    Map function for the dataset. Add the masked text to the example.
    :param example: a single example from the dataset, with keys 'image', 'caption'
    :param mask_mode: str. The mode of masking. One of ["nouns", "sentence", "percentage", "ngram"].
    :param mask_p: float. The maximum proportion of spaCy tokens to mask in the text.
    :param n_gram: int. The n-gram size for the n-gram masking mode.
    :param n_lines: int. The number of lines to stack the text below the image.
    :param language: float. The language of the text. Currently only support ["en", "zh"].
    :param easy_mode: bool. If True, toggle easy mode generation.
    :param font_size: int. The font size of the text.
    :param texts_to_cross: Optional[List[str]]. A list of texts to cross out. If None, use cross_text_func to generate.
    :param background_color: str. The background color of the text image.
    :param save_image: bool. If True, save the generated images.
    :param save_image_name: str. The name of the saved images.
    :param output_tensor: bool. If True, convert the output images to tensors.
    :return: example with additional keys 'stacked_image' (VI+TEI), 'only_it_image' (TEI), 'crossed_text' (List[str]).
    """
    assert mask_mode in ["nouns", "sentence", "percentage", "ngram"]

    if mask_mode == "nouns":
        mask_func = partial(mask_nouns, language=language)
    elif mask_mode == "sentence":
        mask_func = partial(mask_sentence, language=language)
    elif mask_mode == "percentage":
        mask_func = partial(mask_percentage, mask_p=mask_p, language=language)
    else:  # mask_mode == 'ngram'
        mask_func = partial(mask_ngram, mask_p=mask_p, n=n_gram, language=language)

    if language == "en":
        font_path = "arial.ttf"
        if easy_mode:
            lower_cross_height = 0.46
            upper_cross_height = 0.77
        else:
            lower_cross_height = 0.41
            upper_cross_height = 0.80
    elif language == "zh":
        font_path = "simsun.ttc"
        if easy_mode:
            lower_cross_height = 0.38
            upper_cross_height = 0.62
        else:
            lower_cross_height = 0.26
            upper_cross_height = 0.75
    else:
        raise ValueError(f"Language {language} is not currently supported.")

    image = example["image"]
    text_in_image = example["caption"]

    width, height = image.size
    font = ImageFont.truetype(font_path, font_size)

    # below, we generate text_to_cross in case we only provided cross_text_func
    text_image, truncated_text, texts_to_cross = split_and_cross_out_image(
        text=text_in_image,
        font=font,
        max_width=width,
        lower_cross_height=lower_cross_height,
        upper_cross_height=upper_cross_height,
        texts_to_cross=texts_to_cross,
        cross_text_func=mask_func,
        background_color=background_color,
        language=language,
        n_lines=n_lines,
    )

    # Create stacked image (VI+TEI)
    stacked_image = Image.new("RGB", (width, height + text_image.height + 1), "white")
    stacked_image.paste(image, (0, 0))
    stacked_image.paste(text_image, (0, height))

    # Create only text image, with the same resolution as the stacked image
    stacked_width, stacked_height = stacked_image.size
    text_image_width, text_image_height = text_image.size
    text_only_image = Image.new("RGB", (stacked_width, stacked_height), "white")
    top_left_y = (stacked_height - text_image_height) // 2
    text_only_image.paste(text_image, (0, top_left_y))

    if save_image:
        if save_image_name is None:
            stacked_image.save("stacked_image.jpg")
            text_only_image.save("only_it_center.jpg")
            text_image.save("text_image.jpg")
        else:
            stacked_image.save(save_image_name + "_stacked_image.jpg")
            text_only_image.save(save_image_name + "_only_it_center.jpg")
            text_image.save(save_image_name + "_text_image.jpg")
    if output_tensor:
        transform = transforms.Compose([transforms.ToTensor(), ])
        stacked_image = transform(stacked_image)
        text_only_image = transform(text_only_image)

    example["stacked_image"] = stacked_image
    example["only_it_image"] = text_only_image
    example["caption"] = truncated_text
    example["crossed_text"] = texts_to_cross
    return example


def generate_vcr(dataset_name_or_path: str = "wit_en",
                 local_dataset: bool = False,
                 mask_mode: str = "ngram",
                 mask_p: float = 0.5,
                 n_gram: int = 5,
                 n_lines: int = 5,
                 easy_mode: bool = False,
                 num_examples: int = 0,
                 language: str = "en",
                 censor_path: str = None,
                 random_seed: int = 42,
                 output_path: str = './data'
                 ):
    """
    Create a new VCR dataset from a given dataset.
    :param dataset_name_or_path: str. The name or path of the original image-text pair dataset. Need to have "image" and "caption" columns.
    :param local_dataset: bool. If True, load the dataset from local disk. Otherwise, load the dataset from the Hugging Face dataset hub.
    :param mask_mode: str. The mode of masking. One of ["nouns", "sentence", "percentage", "ngram"].
    :param mask_p: float. The maximum proportion of spaCy tokens to mask in the text.
    :param n_gram: int. The n-gram size for the n-gram masking mode.
    :param n_lines: int. The number of lines to stack the text below the image.
    :param easy_mode: bool. If True, toggle easy mode generation.
    :param num_examples: int. The number of examples to generate. 0 for all examples.
    :param language: str. The language of the text. Currently only support ["en", "zh"].
    :param censor_path: str. The path to a censor list. None for no censor list used.
    :param random_seed: int. The random seed.
    :param output_path: str. The path to save the new dataset.
    :return: None. Save the new dataset to disk.
    """
    set_seed(random_seed)

    censor = None
    if censor_path is not None:
        with open(censor_path) as f:
            censor = set([line.strip() for line in f.readlines()])

    assert language in ["en", "zh"]

    if local_dataset:
        dataset = load_from_disk(dataset_name_or_path)
    else:
        dataset = load_dataset(dataset_name_or_path)

    assert isinstance(dataset, Dataset), "The dataset must be a Dataset object."

    if num_examples > 0:
        dataset = dataset.select(sample(range(len(dataset)), num_examples))

    # Initial filtering.
    dataset = dataset.cast_column("image", Image(decode=False))
    dataset = dataset.filter(
        lambda x: x["caption"] is not None
                  and len(x["caption"].strip()) > 0
                  and filter_censor(x, censor))
    dataset = dataset.filter(filter_invalid_image)
    dataset = dataset.cast_column("image", Image(decode=True))

    # Generate the columns for VCR.
    dataset = dataset.map(
        generate_vcr_single,
        fn_kwargs={
            "mask_mode": mask_mode,
            "mask_p": mask_p,
            "language": language,
            "n_gram": n_gram,
            "n_lines": n_lines,
            "easy_mode": easy_mode,
        },
    )

    # Add "question_id" column, which is the index of the example
    def add_question_id(example, idx):
        example["question_id"] = idx
        return example

    dataset = dataset.map(add_question_id, with_indices=True)

    # Remove unnecessary columns
    keep_columns = ["question_id", "image", "caption", "stacked_image", "only_it_image", "crossed_text"]
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in keep_columns])

    # Filter out examples with no crossed text.
    dataset = dataset.filter(lambda x: len(x["crossed_text"]) > 0)
    diff_mode = "easy" if easy_mode else "hard"
    dataset.save_to_disk(os.path.join(output_path,
                                      f"vcr_{dataset_name_or_path.replace('/', '-')}_{language}_{diff_mode}_{mask_mode}_{mask_p}_{n_gram}_{n_lines}"))


if __name__ == "__main__":
    fire.Fire(generate_vcr)
