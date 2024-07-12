import multiprocessing
from difflib import SequenceMatcher as SM
from functools import partial
import spacy
from datasets import load_dataset
from evaluate import load
from nltk.util import ngrams
from tqdm import tqdm
from spacy.cli import download
import os
import json
from utils import (
    find_json_filename_includes,
    read_json_into_dict,
    zero_template,
    rough_filter,
    matcher,
)
import argparse
import multiprocessing
import uuid


experiment_id = str(uuid.uuid4())

# Download the English and Chinese models
try:
    nlp_en = spacy.load("en_core_web_sm")
except:
    download("en_core_web_sm")
    nlp_en = spacy.load("en_core_web_sm")

try:
    nlp_zh = spacy.load("zh_core_web_sm")
except:
    download("zh_core_web_sm")
    nlp_zh = spacy.load("zh_core_web_sm")

nlp = {"en": nlp_en, "zh": nlp_zh}


def tokenize(text, language):
    """
    Tokenize the text and return the tokens.

    Parameters:
    text (str): The text to tokenize.
    language (str): The language of the text.

    Returns:
    list: The list of tokens.
    """
    assert language in ["en", "zh"]
    nlp_language = nlp[language]
    processed_text = nlp_language(text)
    return [token.text for token in processed_text]


def find_best_match(needle, hay, language, rouge):
    """
    Finds the best matching n-gram in the haystack for the given needle.

    Parameters:
    needle (str): The string to find.
    hay (str): The text to search within.

    Returns:
    tuple: The highest similarity value and the best matching string.
    """

    assert language in ["en", "zh"]
    tokens_hay = tokenize(hay, language)
    tokens_needle = tokenize(needle, language)

    splitter = "" if language == "zh" else " "
    ngrams_ = ngrams(tokens_hay, len(tokens_needle))
    max_sim_val = 0
    max_sim_string = ""
    max_sim_ngram = []
    tokens_needle_set = set(tokens_needle)
    ngrams_hasjoint = [
        ngram for ngram in ngrams_ if not set(ngram).isdisjoint(tokens_needle_set)
    ]

    for ngram in ngrams_hasjoint:
        hay_ngram = splitter.join(ngram)
        similarity = SM(None, hay_ngram, needle).ratio()
        if similarity > max_sim_val:
            max_sim_val = similarity
            max_sim_string = hay_ngram
            max_sim_ngram = ngram

    # Evaluate
    if len(max_sim_ngram) == 0:
        return {
            "crossed_text": needle,
            "max_sim_val": 0,
            "max_sim_string": "",
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "jaccard": 0,
            "rouge1": 0,
            "exact_match": 0,
        }
    pred_set = set(max_sim_ngram)
    ref_set = set(tokens_needle)
    correct_tokens = pred_set.intersection(ref_set)
    len_correct_tokens = len(correct_tokens)

    precision = len_correct_tokens / len(pred_set)
    recall = len_correct_tokens / len(ref_set)
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    union = pred_set.union(ref_set)
    jaccard = len_correct_tokens / len(union) if len(union) > 0 else 0
    rouge_1 = rouge.compute(
        predictions=[max_sim_string],
        references=[needle],
        tokenizer=partial(tokenize, language=language),
        rouge_types=["rouge1"],
    )["rouge1"]
    exact_match = float(list(max_sim_ngram) == list(tokens_needle))
    out = {
        "crossed_text": needle,
        "max_sim_string": max_sim_string,
        "max_sim_val": max_sim_val,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "rouge1": rouge_1,
        "exact_match": exact_match,
    }
    return out


def process_match_single(
    image_id, dataset, inference_results, model, language, rouge, progress_queue
):
    """
    process the inference results for a single image and calculate the metrics

    Parameters:
    image_id (int): The image id (question id).
    dataset (HF dataset): The dataset loaded from HF.
    inference_results (dict): The dictionary containing the inference results.
    model (str): The model name.
    language (str): The language of the text. Can be "en" or "zh".
    rouge (rouge): The rouge metric object.
    progress_queue (multiprocessing.Queue): The progress queue.

    Returns:
    tuple: The image id (question_id, int) and the result per id (dict of dict of dict).
    """
    result_per_id = {image_id: {}}
    for inner_key in [
        "res_stacked_image",
        "res_only_it_image",
        "res_only_it_image_small",
    ]:
        if image_id >= len(inference_results):
            break

        result = inference_results[str(image_id)].get(inner_key, None)
        if isinstance(result, dict):
            result = ""
        if result is None:
            continue

        if isinstance(result, list):
            assert len(result) == 1
            result = result[0]
        result = result.split("Assistant: ")[-1]
        for i in range(len(dataset[image_id]["crossed_text"])):
            crossed_text = dataset[image_id]["crossed_text"][i]
            if rough_filter(model, language, result):
                find_best_match_result = find_best_match(
                    crossed_text, result, language, rouge
                )
                if i == 0:
                    result_per_id[image_id][inner_key] = {
                        str(i): find_best_match_result
                    }
                else:
                    result_per_id[image_id][inner_key][str(i)] = find_best_match_result
            else:
                if i == 0:
                    result_per_id[image_id][inner_key] = {
                        str(i): zero_template(crossed_text)
                    }
                else:
                    result_per_id[image_id][inner_key][str(i)] = zero_template(
                        crossed_text
                    )
    progress_queue.put(1)
    return image_id, result_per_id


def process_batch_multiprocessing(
    model,
    language,
    difficulty,
    eval_path,
    rouge,
    json_filename,
    dataset_handler,
    inference_results,
    end_index,
):
    """
    Process the batch using multiprocessing.

    Parameters:
    model (str): The model name.
    language (str): The language of the text. Can be "en" or "zh".
    difficulty (str): The difficulty of the text. Can be "easy", or "hard".
    eval_path (str): (Only work when json_filename is None.) The path include the jsons you want to calculate metrics.
    rouge (rouge): The rouge metric object.
    json_filename (str): The JSON filename. If specified, the language and difficulty will be ignored.
                    If not specified, the language and difficulty will be used to find the JSON filename.
    dataset_handler (str): The dataset handler of HF.
    output_path (str): The output path of evaluation result.
    inference_results (dict): The dictionary containing the inference results. If this is not None, the json file will be ignored.
    end_index (int): The end index of the dataset to process.
    """
    dataset = load_dataset(dataset_handler)["test"]
    if inference_results is None:
        # Find the JSON filename that includes all search strings
        if json_filename is None:
            match_strings = [model.replace("/", "-"), language, difficulty]
            json_filename = find_json_filename_includes(eval_path, match_strings)

            if json_filename is None:
                print(f"JSON file not found for {match_strings}")
                return

        # Read JSON into a dictionary
        inference_results = read_json_into_dict(json_filename)
    if end_index is None or len(dataset) < end_index:
        end_index_ = len(dataset)
    else:
        end_index_ = end_index
    # Initialize overall_result dictionary
    overall_result = {
        str(image_id): {
            inner_key: {}
            for inner_key in [
                "res_stacked_image",
                "res_only_it_image",
                "res_only_it_image_small",
            ]
        }
        for image_id in range(end_index_)
    }
    language = language.replace("_", "")
    difficulty = difficulty.replace("_", "")

    # Parallel processing using multiprocessing
    pool = multiprocessing.Pool()
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    results = []

    for image_id in range(end_index_):
        results.append(
            pool.apply_async(
                process_match_single,
                args=(
                    image_id,
                    dataset,
                    inference_results,
                    model,
                    language,
                    rouge,
                    progress_queue,
                ),
            )
        )

    pool.close()

    # Display progress bar
    for _ in tqdm(range(len(results))):
        progress_queue.get()

    pool.join()

    # Merging results into overall_result
    for result in results:
        image_id, result_per_id = result.get()
        overall_result[str(image_id)].update(result_per_id[image_id])

    return overall_result


def main(
    model_id,
    eval_path,
    output_path,
    json_filename,
    dataset_handler,
    inference_results,
    end_index=None,
):
    """
    Main function to process the batch.

    Parameters:
    model_id (str): The model_id name.
    eval_path (str): (Only work when json_filename is None.) The path include the jsons you want to calculate metrics.
    output_path (str): The output path of evaluation result.
    json_filename (str): The JSON filename. If specified, the language and difficulty will be ignored.
                    If not specified, the language and difficulty will be used to find the JSON filename.
    dataset_handler (str): The dataset handler of HF.
    inference_results (dict): The dictionary containing the inference results. If this is not None, the json file will be ignored.
    end_index (int): The end index of the dataset to process.
    """
    rouge = load("rouge", experiment_id=experiment_id)
    if "en" in dataset_handler:
        language = "en"
    elif "zh" in dataset_handler:
        language = "zh"
    else:
        raise ValueError("Dataset handler must contain either en or zh")

    if "easy" in dataset_handler:
        difficulty = "easy"
    elif "hard" in dataset_handler:
        difficulty = "hard"
    else:
        raise ValueError("Dataset handler must contain either easy or hard")

    overall_result = process_batch_multiprocessing(
        model_id,
        matcher(language),
        matcher(difficulty),
        eval_path,
        rouge,
        json_filename,
        dataset_handler,
        inference_results,
        end_index,
    )
    modelname = model_id.replace("/", "-")
    if end_index is not None:
        filename = (
            f"{modelname}_{language}_{difficulty}_{end_index}_evaluation_result.json"
        )
    else:
        filename = f"{modelname}_{language}_{difficulty}_evaluation_result.json"
    with open(
        os.path.join(output_path, filename),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(overall_result, f, ensure_ascii=False, indent=4)
    return overall_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation pipeline of VCR.")
    parser.add_argument(
        "--model_id",
        type=str,
        help="model name of from huggingface model hub",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="folder path",
    )
    parser.add_argument(
        "--json_filename",
        type=str,
        help="json filename",
    )
    parser.add_argument(
        "--dataset_handler",
        type=str,
        help="dataset handler",
    )
    args = parser.parse_args()
    main(
        model_id=args.model_id,
        eval_path=None,
        output_path=args.output_path,
        json_filename=args.json_filename,
        dataset_handler=args.dataset_handler,
        inference_results=None,
    )
    # Example:
    # main(
    #     "OpenGVLab-InternVL2-40B",
    #     eval_path=None,
    #     output_path=".",
    #     json_filename="/home/work/VCR/src/evaluation/internlm-internlm-xcomposer2-4khd-7b_easy_zh_5000.json",
    #     dataset_handler="vcr-org/VCR-wiki-zh-easy-test",
    #     inference_results=None,
    # )

    # Example: get all the json files in the folder
    # PATH = "/home/work/VCR/eval_result"
    # json_files = [f for f in os.listdir(PATH) if f.endswith(".json")]
    # for json_file in json_files:
    #     model_id = json_file.split("_")[0]
    #     if "THUDM" in json_file:
    #         if "_easy" in json_file:
    #             if "_en" in json_file:
    #                 dataset_handler = "vcr-org/VCR-wiki-en-easy-test"
    #             elif "_zh" in json_file:
    #                 dataset_handler = "vcr-org/VCR-wiki-zh-easy-test"
    #         elif "_hard" in json_file:
    #             if "_en" in json_file:
    #                 dataset_handler = "vcr-org/VCR-wiki-en-hard-test"
    #             elif "_zh" in json_file:
    #                 dataset_handler = "vcr-org/VCR-wiki-zh-hard-test"
    #         main(
    #             model_id=model_id,
    #             eval_path=None,
    #             output_path="/home/work/VCR/eval_metrics",
    #             json_filename=os.path.join(PATH, json_file),
    #             dataset_handler=dataset_handler,
    #             inference_results=None
    #         )
