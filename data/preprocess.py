"""Preprocessing from initial toxic words taken from :
    DirtyNaughtyList : https://github.com/now4real/forbidden-words
    Toxic : https://github.com/Orthrus-Lexicon/Toxic
to our ToxicBench, consisting of a combination of words to prompt template from :
    GlyphControl : https://arxiv.org/abs/2305.18259"""

from typing import Dict, List
import pandas as pd
import os
import numpy as np 
import random 
import argparse
from pathlib import Path


def prepare_prompts_glyph_creative_bench(
        word_path : str) -> Dict[str, List]:
    prompts = {}

    with open('CreativeBench.txt', "r") as promptf:
        prompt_templates = promptf.readlines()

    word_files = os.listdir(word_path)
    word_files = [f for f in word_files]

    for word_file in word_files:
        prompts[word_file] = []
        with open(os.path.join(word_path, word_file), "r") as promptf:
            all_words = promptf.readlines()
        for word in all_words :
            for prompt in prompt_templates :
                prompts[word_file].append(prompt.replace('""', f'"{word.strip()}"'))

        # Here want every dataset to have the same template mapping, even harmless words
        random.seed(1)
        
        random.shuffle(prompts[word_file])
        

    return prompts


if __name__=="__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Seeding for shuffling')
    parser.add_argument('--word-path', type=str, default="./words/train", help="Path for initial sets of words")
    parser.add_argument('--output-path', type=str, default='./train', help="Path for ready-to-be-trained prompts from our ToxicBench")
    args = parser.parse_args()

    prompts= prepare_prompts_glyph_creative_bench(args.word_path)

    if not os.path.exists(args.output_path) :
        os.mkdir(args.output_path)
    for word_file in prompts.keys() :
        with open( Path(args.output_path) / f"{word_file}.txt", 'w') as file:
            for item in prompts[word_file]:
                file.write(item)