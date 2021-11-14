import argparse
import json
import pickle
import os
import random
import subprocess
import torch
import time
import tqdm

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from style_paraphrase.inference_utils import GPT2Generator

# from profanity_filter import ProfanityFilter
OUTPUT_DIR = "./data/"
if __name__ == "__main__":

    with torch.cuda.device(0):
        print("Loading Tweets model...")
        tweets = GPT2Generator(OUTPUT_DIR + "/models/cds_models/tweets")
