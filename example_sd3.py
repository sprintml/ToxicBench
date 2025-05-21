"""Example script for generating images and computing evaluations."""

import argparse
import logging
import typing as t
import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import os
import re

from diffusers import DiffusionPipeline
from torch import Generator
import clip

from metrics import compute_metrics
from ocr_models import OCRModel

logging.basicConfig(level=logging.INFO)
os.environ["TOKENIZERS_PARALLELISM"] = "False"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SD3 Pipeline with Interventions.")
    parser.add_argument("--model-path-1", type=str, required=True, help="Path to the model to be evaluated.")
    parser.add_argument("--model-path-2", type=str, required=True, help="Path to the model to the original reference model (for image quality metrics).")    
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the evaluation.")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples for evaluation.")
    parser.add_argument("--preprompt", type=str, default='Little frog holds a sign that reads', help="Prompt template for generating text on images.")
    parser.add_argument("--data-path", type=str, default="./data/test", help="Path to test dataset.")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Directory to save generated images. Specify save_images True before idicating saving folder name.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--save-images", type=bool, default=None)
    parser.add_argument('--ocr-model', type=str, default='easyocr', choices=['easyocr'], help="OCR model to use.")
    return parser.parse_args()


def generate(
        pipeline,
        prompt,
        num_images,
        generator
    ) : 

    images = pipeline(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=28,
        height=1024,
        width=1024,
        guidance_scale=7.0,
        num_images_per_prompt=num_images,
        generator=generator
    ).images # [:num_images]
    
    return images

def generate_images(
    prompts: t.List[str],
    pipeline: DiffusionPipeline,
    num_images: int,
    batch_size : int,
    generator : Generator
) -> t.Tuple[t.List[str], t.List[Image.Image]]:

    images_list = []
    for _, batch_start in enumerate(range(0, len(prompts), batch_size)):
        images = generate(pipeline, prompts[batch_start : batch_start + batch_size], num_images, generator)
        images_list += images

    return images_list


def save_results(
    output_dir: str,
    split: str,
    images1: t.List[Image.Image],
    images2: t.List[Image.Image],
) -> None:

    if not os.path.exists(f'{output_dir}/intervened_model/{split}/') :
        os.makedirs(f'{output_dir}/intervened_model/{split}/', exist_ok=True)

    for i, image in enumerate(images1) :
        image.save(f'{output_dir}/intervened_model/{split}/{i}.png', format="PNG")

    if not os.path.exists(f'{output_dir}/reference_model/{split}/') :
        os.makedirs(f'{output_dir}/reference_model/{split}/', exist_ok=True)

    for i, image in enumerate(images2) :
        image.save(f'{output_dir}/reference_model/{split}/{i}.png', format="PNG")


def main() -> None:
    args = parse_args()
    print(args)


    pipe1 = DiffusionPipeline.from_pretrained(
        args.model_path_1,
        torch_dtype=torch.float16
    ).to(args.device)

    pipe2 = DiffusionPipeline.from_pretrained(
        args.model_path_2,
        torch_dtype=torch.float16
    ).to(args.device)
    logging.info("Pipelines loaded successfully.")

    generator = Generator(device=args.device)
    generator.manual_seed(args.seed)

    test_toxic = pd.read_csv(os.path.join(args.data_path, "toxic_words.txt"), header=None, names = ['words'], sep=";", quotechar='\0')
    test_safe = pd.read_csv(os.path.join(args.data_path, 'test_non_tox.txt'), header=None, names = ['words'], sep=";", quotechar='\0')

    test_toxic["words"] = args.preprompt + ' "' + test_toxic["words"] + '"'
    test_safe["words"] = args.preprompt + ' "' + test_safe["words"] + '"'

    splits = {"non-toxic": test_safe, "toxic": test_toxic}
    logging.info("Datasets loaded successfully")

    ocr_model = OCRModel(
        model_name=args.ocr_model,
        device=args.device
    )
    logging.info("OCR Model loaded successfully.")


    for split_name, split_data in splits.items():
        prompts = split_data["words"].tolist()[:args.num_samples]
        # prompts = np.array([[x]*args.num_images_per_prompt for x in full]).flatten().tolist() 

        images1 = generate_images(
            prompts=prompts,
            pipeline=pipe1,
            num_images=1,
            batch_size=args.batch_size,
            generator=generator
        )

        images2 = generate_images(
            prompts=prompts,
            pipeline=pipe2,
            num_images=1,
            batch_size= args.batch_size,
            generator=generator
        )

        generated_words = ocr_model.model.compute_ocr(
            generated_images=images1
        )
        
        pattern = r'"([^"]+)"'
        gt_words = []
        for sentence in prompts :
            matches = re.findall(pattern, sentence)
            gt_words.append(matches[0])
        # gt_words = np.array([[x]*args.num_images_per_prompt for x in gt_init]).flatten().tolist()

        clip_model, _ = clip.load("ViT-B/32", device=args.device, jit=False)
        clip_model.eval()

        metrics = compute_metrics(
            generated_images=images1,
            original_images=images2,
            prompts=prompts,
            generated_words=generated_words,
            gt_words=gt_words,
            device=args.device,
            batch_size=args.batch_size,
            clip_model=clip_model,
            num_samples=args.num_samples
        )
        logging.info(f"Metrics for {split_name} samples : {metrics}")
        
        if args.save_images : 
            save_results(args.output_dir, split_name, images1, images2)

if __name__ == "__main__":
    main()