# Author: Yuhuan Jiang
# Version: 1.0
# Language: Python 2

import argparse
import math
import pickle

# The following lines parse the command line arguments for you. You may ignore this part.
argument_parser = argparse.ArgumentParser("Intrinsic evaluator of language model. ")
argument_parser.add_argument('-m', '--model', help='The path to the directory containing the necessary files to recreate the language model.', required=True)
argument_parser.add_argument('-i', '--input', help='The path to the input file containing the testing sentences. ', required=True)
argument_parser.add_argument('-o', '--output', help='The path to the output file containing the perplexity score on the testing sentences. ', required=True)
args = argument_parser.parse_args()

# The following variables are created for your convenience.
# They are the values from the command line input.
model_dir = args.model
input_path = args.input
output_path = args.output


# START OF YOUR IMPLEMENTATION

# Utilities
def save_perplexity(perplexity):
    with open(output_path, "w") as f:
        f.write(str(perplexity))


# Prints out what this script does
print("Evaluating the perplexity of the model found at " + model_dir)
print("  on sentences found at " + input_path)
print("The perplexity scores will be saved at " + output_path)
print("")


# This variable holds the training sentences. Example content::
#
# train_snts = [
#     ["john", "has", "a",  "cat", "."],
#     ["mary", "has", "a",  "dog", "."],
#     ["john", "'s", "cat",  "is", "not", "a", "dog" "."],
#     ["mary", "'s", "dog",  "is", "not", "a", "cat" "."]"
# ]
#
testing_sentences = [line.rstrip('\n').split(' ') for line in open(input_path)]


# Recreate the LM
with open(model_dir + "/model_type.txt") as f:
    model_type = f.readline().rstrip("\n")
    smooth_or_not = f.readline()

with open(model_dir + "/vocab_size.txt") as f:
    vocab_size = float(f.readline())

if model_type == "1" or model_type == "3" or model_type == "s3":
    # Get the ngram counts
    with open(model_dir + "/ngram_counts.pkl", "rb") as f:
        unigram_dict = pickle.load(f)




if model_type == "dummy":
    # This is 1/vocab_size in log domain,
    # We use log domain to avoid underflow,
    log_prob = -math.log(vocab_size)

    # We compute the perplexity by first computing the cross-entropy h, then use exp(h) as entropy.

    total_log_prob = 0.0
    num_tokens = 0

    for ws in testing_sentences:
        for w in ws:
            total_log_prob += log_prob
            num_tokens += 1

    h = -1.0 * total_log_prob / num_tokens
    perplexity = math.exp(h)

    # In fact, this uniform unigram LM will have a perplexity which is exactly equals to the vocabulary size.

    with open(output_path, "w") as f:
        f.writelines(str(perplexity))

elif model_type == "1":
    # unsmoothed unigram
    # P(w1w1...wk) = P(w1)P(w2)...P(wk)
    prob = 0.0
    log_prob = 0.0
    total_log_prob = 0.0
    num_tokens = 0

    for ws in testing_sentences:    # For every sentences in testing document:
        for w in ws:       # For every work token in every sentence:
            if not unigram_dict.has_key(w):
                w = "<unk>"
            #print(w)
            prob = unigram_dict[w] / vocab_size
            log_prob = math.log(prob)
            total_log_prob += log_prob
            num_tokens += 1
    h = -1.0 * total_log_prob / num_tokens
    perplexity = math.exp(h)

    with open(output_path, "w") as f:
        f.writelines(str(perplexity))


else:
    print("Not implemented!! ")
