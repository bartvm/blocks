#!/usr/bin/env python
"""Learn to reverse the words in a text."""
import logging
import argparse
from examples.reverse_words import main

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        "Case study of learning to reverse words from a natural text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "mode", choices=["train", "test", "beam_search"],
        help="The mode to run. In the `train` mode a model is trained."
             " In the `test` and `beam_search` modes a trained model is used "
             " to reverse words in the input text.")
    parser.add_argument(
        "save_path", default="chain",
        help="The path to save the training process if the mode"
             " is `train` OR path to an `.npz` files with learned"
             " parameters if the mode is `test`.")
    parser.add_argument(
        "--num-batches", default=20000, type=int,
        help="Train on this many batches.")
    args = parser.parse_args()
    main(**vars(args))
