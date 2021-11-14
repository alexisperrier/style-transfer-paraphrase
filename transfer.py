import argparse
import logging
import sys
import torch
import pandas as pd

from style_paraphrase.inference_utils import GPT2Generator

if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("Please check if a GPU is available or your Pytorch installation is correct.")
        sys.exit()

    models = {
        'tweets5':      GPT2Generator(f"./data/models/cds_models/tweets", upper_length="same_5"),
        'tweets':       GPT2Generator(f"./data/models/cds_models/tweets"),
        # 'lyrics':       GPT2Generator(f"./data/models/cds_models/lyrics"),
        # 'bible':       GPT2Generator(f"./data/models/cds_models/bible"),
        'switchboard':  GPT2Generator(f"./data/models/cds_models/switchboard"),
        # 'aae':          GPT2Generator(f"./data/models/cds_models/aae"),
        # 'model_313':    GPT2Generator(f"./data/models/formality_models/model_313"),
    }

    top_ps = [0.1, 0.3, 0.6, 1]

    # load sentence
    input_sentence = input("Enter your sentence, q to quit: ")

    while input_sentence != "q" and input_sentence != "quit" and input_sentence != "exit":
        for model_name, paraphraser in models.items()():
            print(f"====== {model_name}")
            paraphraser.modify_p(top_p=0.0)
            greedy_decoding = paraphraser.generate(input_sentence)
            print(f"greedy sample:{greedy_decoding}")
            for top_p in top_ps:
                print(f"-- top_p {top_p}")
                paraphraser.modify_p(top_p=top_p)
                top_p_60_samples, _ = paraphraser.generate_batch([input_sentence, input_sentence, input_sentence])
                top_p_60_samples = "\n- ".join(top_p_60_samples)
                print("- " + top_p_60_samples+ "\n")

            input_sentence = input("Enter your sentence, q to quit: ")

    print("Exiting...")
