# VCR: Visual Caption Restoration

[Tianyu Zhang†](https://ai.t-zhang.com), [Suyuchen Wang†](https://github.com/sheryc), [Lu Li](https://sites.google.com/view/meetluli/home), [Ge Zhang](https://scholar.google.com/citations?user=qyTrq4kAAAAJ), [Perouz Taslakian](https://perouz.github.io/), [Sai Rajeswar](https://sairajeswar.com/), [Jie Fu](https://bigaidream.github.io/), [Bang Liu](https://www-labs.iro.umontreal.ca/~liubang/), [Yoshua Bengio](https://yoshuabengio.org/)

† Equal contribution

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2406.06462-blue.svg)](https://arxiv.org/abs/2406.06462)
[![Static Badge](https://img.shields.io/badge/%F0%9F%A4%97-VCR_wiki_full-red)](https://huggingface.co/collections/vcr-org/vcr-visual-caption-recognition-6661393b1761e2aff7b967b9)
[![Static Badge](https://img.shields.io/badge/%F0%9F%A4%97-VCR_wiki_small_test-red)](https://huggingface.co/collections/vcr-org/vcr-visual-caption-restoration-smaller-test-subsets-6667b591329b67db9408b493)
</div>

<div align="center">
  <img src="assets/icon_vcr.jpg" alt="VCR-Wiki Logo" width="475"/>
</div>

# News
- 🔥🔥🔥 **[2024-06-13]** We release the evaluation codes for open-source models, closed-source models and the pipeline of creating the dataset.
- 🔥🔥🔥 **[2024-06-12]** We have incorperated the VCR-wiki evaluation process in [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) framework. Now, users can use one line command to run the evaluation of models on the VCR-wiki test datasets.
- 🔥🔥🔥 **[2024-06-11]** Our paper has been released on the [arXiv](https://arxiv.org/abs/2406.06462), including the evaluation results of a series of models.
- 🔥🔥🔥 **[2024-06-10]** We have released the [VCR-wiki dataset](https://huggingface.co/vcr-org), which contains 2.11M English and 346K Chinese entities sourced from Wikipedia, offered in both easy and hard variants. The dataset is available in the Hugging Face Datasets library.

# Quick Start
```bash
pip install datasets
```
```python
from datasets import load_dataset
# load the English easy mode dataset
dataset = load_dataset("vcr-org/VCR-wiki-en-easy")
# load the English hard mode dataset
dataset = load_dataset("vcr-org/VCR-wiki-en-hard")
# load the Chinese easy mode dataset
dataset = load_dataset("vcr-org/VCR-wiki-zh-easy")
# load the Chinese hard mode dataset
dataset = load_dataset("vcr-org/VCR-wiki-zh-hard")

for obs in dataset['train']: # or 'validation' or 'test'
    # your own code here
```
## Dataset List
|  Dataset   | Download Link |
|  ----  | ----  | 
|  100 Test Subset  | <li>[🤗vcr-org/VCR-wiki-en-easy-test-100](https://huggingface.co/datasets/vcr-org/VCR-wiki-en-easy-test-100) <li> [🤗vcr-org/VCR-wiki-en-hard-test-100](https://huggingface.co/datasets/vcr-org/VCR-wiki-en-hard-test-100) <li> [🤗vcr-org/VCR-wiki-zh-easy-test-100](https://huggingface.co/datasets/vcr-org/VCR-wiki-zh-easy-test-100) <li> [🤗vcr-org/VCR-wiki-zh-hard-test-100](https://huggingface.co/datasets/vcr-org/VCR-wiki-zh-hard-test-100)|
|  500 Test Subset  | <li>[🤗vcr-org/VCR-wiki-en-easy-test-500](https://huggingface.co/datasets/vcr-org/VCR-wiki-en-easy-test-500) <li> [🤗vcr-org/VCR-wiki-en-hard-test-500](https://huggingface.co/datasets/vcr-org/VCR-wiki-en-hard-test-500) <li> [🤗vcr-org/VCR-wiki-zh-easy-test-500](https://huggingface.co/datasets/vcr-org/VCR-wiki-zh-easy-test-500) <li> [🤗vcr-org/VCR-wiki-zh-hard-test-500](https://huggingface.co/datasets/vcr-org/VCR-wiki-zh-hard-test-500)|
|  5000 (Full) Test Set   | <li>[🤗vcr-org/VCR-wiki-en-easy-test](https://huggingface.co/datasets/vcr-org/VCR-wiki-en-easy-test) <li> [🤗vcr-org/VCR-wiki-en-hard-test](https://huggingface.co/datasets/vcr-org/VCR-wiki-en-hard-test) <li> [🤗vcr-org/VCR-wiki-zh-easy-test](https://huggingface.co/datasets/vcr-org/VCR-wiki-zh-easy-test) <li> [🤗vcr-org/VCR-wiki-zh-hard-test](https://huggingface.co/datasets/vcr-org/VCR-wiki-zh-hard-test)|
|  Train Validation Test (Full) Set    | <li>[🤗vcr-org/VCR-wiki-en-easy](https://huggingface.co/datasets/vcr-org/VCR-wiki-en-easy) <li> [🤗vcr-org/VCR-wiki-en-hard](https://huggingface.co/datasets/vcr-org/VCR-wiki-en-hard) <li> [🤗vcr-org/VCR-wiki-zh-easy](https://huggingface.co/datasets/vcr-org/VCR-wiki-zh-easy) <li> [🤗vcr-org/VCR-wiki-zh-hard](https://huggingface.co/datasets/vcr-org/VCR-wiki-zh-hard)|## Training and Evaluation Datasets

# Introduction
We present VCR-wiki, a dataset designed for the Visual Caption Restoration (VCR) task. 

Please refer to our main figure below for an overview of the VCR task.

<div align="center">
  <img src="assets/main_pic_en_easy.jpg" alt="VCR-Wiki Logo" width="450"/>
</div>


VCR challenges models to restore partially obscured text within images, leveraging pixel-level hints and contextual cues. Unlike traditional text-based tasks, VCR necessitates a synergistic understanding of **visual image (VI)**, **string text (ST)**, and **text embedded in image (TEI)**. Our dataset is crafted using a pipeline that generates synthetic images from image-caption pairs with adjustable caption visibility, allowing for varied difficulty levels. The show the pipeline of creating the dataset here and we will release the code of creating the dataset soon.

<div align="center">
  <img src="assets/vcr_pipeline.png" alt="VCR-pipeline" width="900"/>
</div>


VCR-wiki comprises **2.11M** English and **346K** Chinese entities sourced from Wikipedia, offered in both easy and hard variants. Initial results indicate that current vision-language models fall short compared to human performance on this task.


# Model Evaluation

## Method 1 (recommended): use the evaluation script
### Open-source evaluation
We support open-source model_id: 
```python
["openbmb/MiniCPM-Llama3-V-2_5",
"OpenGVLab/InternVL-Chat-V1-5",
"internlm/internlm-xcomposer2-vl-7b",
"HuggingFaceM4/idefics2-8b",
"Qwen/Qwen-VL-Chat",
"THUDM/cogvlm2-llama3-chinese-chat-19B",
"THUDM/cogvlm2-llama3-chat-19B",
"echo840/Monkey-Chat",]
```
For the models not on list, they are not intergated with huggingface, please refer to their github repo to create the evaluation pipeline.

```bash
# We use HuggingFaceM4/idefics2-8b and vcr_wiki_en_easy as an example
# Inference from the VLMs and save the results to {model_id}_{difficulty}_{language}.json
cd src/evaluation
python3 inference.py --dataset_handler "vcr-org/VCR-wiki-en-easy-test" --model_id "HuggingFaceM4/idefics2-8b" --device "cuda" --dtype "bf16" --save_interval 50 --resume True

# Evaluate the results and save the evaluation metrics to {model_id}_{difficulty}_{language}_evaluation_result.json
python3 evaluation_metrics.py --model_id HuggingFaceM4/idefics2-8b --output_path . --json_filename "HuggingFaceM4_idefics2-8b_en_easy.json" --dataset_handler "vcr-org/VCR-wiki-en-easy-test"

# To get the mean score of all the `{model_id}_{difficulty}_{language}_evaluation_result.json` in `jsons_path` (and the std, confidence interval if `--bootstrap`) of the evaluation metrics
python3 gather_results.py --jsons_path .
```

### Close-source evaluation
We provide the evaluation script for the close-source model: `GPT-4o`, `GPT-4-Turbo`, `Claude-3-Opus` in the `evaluation` folder.

You need an API Key, a pre-saved testing dataset and specify the path of the data saving the paper
```bash
cd src/evaluation
# save the testing dataset to the path
python3 save_image_from_dataset.py --output_path .

# Inference Put your API key and Image Path in the evaluation script (e.g. gpt-4o.py)
python3 gpt-4o.py

# Evaluate the results and save the evaluation metrics to {model_id}_{difficulty}_{language}_evaluation_result.json
python3 evaluation_metrics.py --model_id gpt4o --output_path . --json_filename "gpt4o_en_easy.json" --dataset_handler "vcr-org/VCR-wiki-en-easy-test"

# To get the mean score of all the `{model_id}_{difficulty}_{language}_evaluation_result.json` in `jsons_path` (and the std, confidence interval if `--bootstrap`) of the evaluation metrics
python3 gather_results.py --jsons_path .
```

## Method 2: use lmms-eval framework
You may need to incorporate the inference method of your model if the lmms-eval framework does not support it. For details, please refer to [here](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/model_guide.md)
```bash
pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
# We use HuggingFaceM4/idefics2-8b and vcr_wiki_en_easy as an example
python3 -m accelerate.commands.launch --num_processes=8 -m lmms_eval --model idefics2 --model_args pretrained="HuggingFaceM4/idefics2-8b" --tasks vcr_wiki_en_easy --batch_size 1 --log_samples --log_samples_suffix HuggingFaceM4_idefics2-8b_vcr_wiki_en_easy --output_path ./logs/
```

# Dataset Generation

The code for generating a VCR dataset is in `src/dataset`. Before you start, you need:

1. A dataset containing two columns: `image` and `caption`, where `image` contains PIL.Image objects and `caption` is the corresponding caption.
2. A font file for rendering text on images. We used Arial for English and SimSum for Chinese in our experiments.
3. (Optional) A censor word list for initial filtering of harmful entries.

To generate a VCR dataset, you can run the following command:

```bash
cd src/build_dataset
python generate_vcr_dataset.py \
    --dataset_path /path/to/dataset \
    --is_local_dataset True \
    --mask_mode "ngram" \
    --language "en" \
    --font_path /path/to/font \
    --censor_path /path/to/censor \
    --output_dir /path/to/output
```

The full list of arguments for `generate_vcr_dataset.py` is as follows:
* `--dataset_path`: The name or path of the original image-text pair dataset. Need to have "image" and "caption" columns.
* `--is_local_dataset`: Whether the dataset is stored locally. If True, the script will call `datasets.load_from_disk()` to load the dataset.
* `--mask_mode`: The masking mode for generating VCR dataset. One of "nouns", "sentence", "percentage", "ngram". Default is "ngram".
* `--mask_p`: The percentage of words to mask when `mask_mode` is "percentage". Default is 0.5.
* `--n_gram`: The n-gram length when `mask_mode` is "ngram". Default is 5.
* `--n_lines`: The total number of lines of caption to keep in the image. Default is 5.
* `--language`: The language of the dataset. Currently, has to be one of "en" or "zh".
* `--easy_mode`: Whether to generate the easy mode dataset. Default is False.
* `--font_path`: The path to the font file for rendering text on images. You will need to download the font file yourself.
* `--font_size`: The font size for rendering text on images. Default is 20.
* `--background_color`: The background color for rendering text on images. Default is "white".
* `--save_image_examples`: Whether to save example images. Default is False.
* `--save_image_name`: The name of the saved example image. Default is None.
* `--num_examples`: The number of instances in the output dataset. Default is 0 (no limit).
* `--censor_path`: The path to the censor word list for initial dataset filtering. Default is None.
* `--random_seed`: The random seed for dataset generation. Default is 42.
* `--output_dir`: The output directory for the generated VCR dataset. Default is `./data`.

# Citation
If you find VCR useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{zhang2024vcr,
  title   = {VCR: Visual Caption Restoration},
  author  = {Tianyu Zhang and Suyuchen Wang and Lu Li and Ge Zhang and Perouz Taslakian and Sai Rajeswar and Jie Fu and Bang Liu and Yoshua Bengio},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2406.06462}
}
```
