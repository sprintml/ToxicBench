# ToxicBench

[![arXiv](https://img.shields.io/badge/arXiv-2502.02514-b31b1b.svg)](https://arxiv.org/abs/2502.05066)

Code for ToxicBench, for evaluating toxic text in images from vision generative models.

<span style="color:red">**Warning:** This paper contains examples of offensive language, including insults, and sexual or explicit terms, used solely for research and analysis purposes.</span>

## Abstract

State-of-the-art visual generation models, such as Diffusion Models (DMs) and Vision Auto-Regressive Models (VARs), produce highly realistic images. While prior work has successfully mitigated Not Safe For Work (NSFW) content in the visual domain, we identify a novel threat: the generation of NSFW text embedded within images. This includes offensive language, such as insults, racial slurs, and sexually explicit terms, posing significant risks to users. We show that all state-of-the-art DMs (e.g., SD3, Flux, DeepFloyd IF) and VARs (e.g., Infinity) are vulnerable to this issue. Through extensive experiments, we demonstrate that existing mitigation techniques, effective for visual content, fail to prevent harmful text generation while substantially degrading benign text generation. As an initial step toward addressing this threat, we explore safety fine-tuning of the text encoder underlying major DM architectures using a customized dataset. Thereby, we suppress NSFW generation while preserving overall image and text generation quality. Finally, to advance research in this area, we introduce ToxicBench, an open-source benchmark for evaluating NSFW text generation in images. ToxicBench provides a curated dataset of harmful prompts, new metrics, and an evaluation pipeline assessing both NSFW-ness and generation quality. Our benchmark aims to guide future efforts in mitigating NSFW text generation in text-to-image models.

![ToxicBench Evaluation Pipeline](ToxicBench.png)

## Installation Guide

Clone the repo and install the required libraries for this repo :

```bash
git clone https://github.com/sprintml/ToxicBench
cd ToxicBench
pip install -r requirements.txt
```


## How To Use The Benchmark

For generating the dataset :

```bash
cd ./data
python preprocess.py \
    --seed 1 \
    --word-path "./words/train" \
    --output-path "./train" 
python preprocess.py \
    --seed 1 \
    --word-path "./words/test" \
    --output-path "./test" 
```

For evaluating, an example file ```example_sd3.py``` is provided. Test evaluation can be done with :

```bash
python example_sd3.py \
    --model-path-1 "stabilityai/stable-diffusion-3-medium-diffusers" \
    --model-path-2 "stabilityai/stable-diffusion-3-medium-diffusers" \
    --device "cuda" \
    --num-samples 20 \
    --data-path "./data/words/test" \
    --output-dir "./output-test" \
    --batch-size 4 \
    --seed 42 \
    --save-images True \
    --ocr-model "easyocr"
```

## Citing our work

```
@article{ToxicBench2025,
      title={Beautiful Images, Toxic Words: Understanding and Addressing Offensive Text in Generated Images}, 
      author={Aditya Kumar and Tom Blanchard and Adam Dziedzic and Franziska Boenisch},
      year={2025},
      eprint={2502.05066},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.05066}, 
}
```
