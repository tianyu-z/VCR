# <img src="assets/icon_vcr.jpg" alt="Logo" width="30" style="vertical-align: middle;"> VCR: Visual Caption Restoration

[Tianyu Zhang†](https://ai.t-zhang.com), [Suyuchen Wang†](https://github.com/sheryc), [Lu Li](https://sites.google.com/view/meetluli/home), [Ge Zhang](https://scholar.google.com/citations?user=qyTrq4kAAAAJ), [Perouz Taslakian](https://perouz.github.io/), [Sai Rajeswar](https://sairajeswar.com/), [Jie Fu](https://bigaidream.github.io/), [Bang Liu](https://www-labs.iro.umontreal.ca/~liubang/), [Yoshua Bengio](https://yoshuabengio.org/)

† Equal contribution

[![arXiv](https://img.shields.io/badge/arXiv-2406.06462-blue.svg)](https://arxiv.org/abs/2406.06462)
[![Static Badge](https://img.shields.io/badge/%F0%9F%A4%97-VCR_wiki_full-red)](https://huggingface.co/collections/vcr-org/vcr-visual-caption-recognition-6661393b1761e2aff7b967b9)
[![Static Badge](https://img.shields.io/badge/%F0%9F%A4%97-VCR_wiki_small_test-red)](https://huggingface.co/collections/vcr-org/vcr-visual-caption-restoration-smaller-test-subsets-6667b591329b67db9408b493)

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


# Introduction
We present VCR-wiki, a dataset designed for the Visual Caption Restoration (VCR) task. 

Please refer to our main figure below for an overview of the VCR task.

<div align="center">
  <img src="assets/main_pic_en_easy.jpg" alt="VCR-Wiki Logo" width="300"/>
</div>


VCR challenges models to restore partially obscured text within images, leveraging pixel-level hints and contextual cues. Unlike traditional text-based tasks, VCR necessitates a synergistic understanding of **visual image (VI)**, **string text (ST)**, and **text embedded in image (TEI)**. Our dataset is crafted using a pipeline that generates synthetic images from image-caption pairs with adjustable caption visibility, allowing for varied difficulty levels. The show the pipeline of creating the dataset here and we will release the code of creating the dataset soon.

<div align="center">
  <img src="assets/vcr_pipeline.png" alt="VCR-pipeline" width="900"/>
</div>


VCR-wiki comprises **2.11M** English and **346K** Chinese entities sourced from Wikipedia, offered in both easy and hard variants. Initial results indicate that current vision-language models fall short compared to human performance on this task.


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