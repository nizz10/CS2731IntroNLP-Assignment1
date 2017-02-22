# Author: Ningqian Zhang
# Version: 2.0
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


with open(model_dir + "/vocabulary.pkl", "rb") as f:
    vocab_dict = pickle.load(f)



if model_type == "3" or model_type == "3s":
    with open(model_dir + "/trigram_counts.pkl", "rb") as f:
        trigram_dict = pickle.load(f)




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
    total_log_prob = 0.0
    num_tokens = 0      # number of tokens in testing sentences

    context_size = float(sum(vocab_dict.values()))

    for ws in testing_sentences:    # For every sentences in testing document:
        for w in ws:       # For every work token in every sentence:
            if not vocab_dict.has_key(w):
                w = "<unk>"
            prob = vocab_dict[w] / context_size
            log_prob = math.log(prob)
            total_log_prob += log_prob
            num_tokens += 1

    h = -1.0 * total_log_prob / num_tokens
    perplexity = math.exp(h)

    with open(output_path, "w") as f:
        f.writelines(str(perplexity))

elif model_type == "3":
    # unsmoothed trigram
    context_size = float(sum(vocab_dict.values()) - 1) # minus one <s>

    prob = 0.0
    log_prob = 0.0
    total_log_prob = 0.0
    num_tokens = 0
    isNaN = False

    for ws in testing_sentences:
        for index in range(len(ws)):
            if index > 1:   # Skipped the first two <s>s
                # Replace the words not present in the vocab_dict with <unk>s first
                if not vocab_dict.has_key(ws[index]):
                    ws[index] = "<unk>"
                bigram = ws[index - 2] + " " + ws[index - 1]
                trigram = ws[index - 2] + " " + ws[index - 1] + " " + ws[index]

                # Compare with the trigram_dict generated according to training sentences
                if not trigram_dict.has_key(bigram) or not trigram_dict.has_key(trigram):
                    isNaN = True
                    break
                else:
                    prob = trigram_dict[trigram] / float(trigram_dict[bigram])
                log_prob = math.log(prob)
                total_log_prob += log_prob
                num_tokens += 1
        if isNaN:
            break
    if not isNaN:
        h = -1.0 * total_log_prob / num_tokens
        perplexity = math.exp(h)
    else:   # Output "NaN" in perplexity file if trigram or bigram has count 0
        perplexity = "NaN"
    with open(output_path, "w") as f:
        f.writelines(str(perplexity))

elif model_type == "3s":
    # smoothed trigram_dict
    prob = 0.0
    log_prob = 0.0
    total_log_prob = 0.0
    num_tokens = 0

    vocab_size = float(len(vocab_dict.keys())-1)    # minus one <s>

    for ws in testing_sentences:
        for index in range(len(ws)):
            if index > 1:
                if not vocab_dict.has_key(ws[index]):
                    ws[index] = "<unk>"
                bigram = ws[index - 2] + " " + ws[index - 1]
                trigram = ws[index - 2] + " " + ws[index - 1] + " " + ws[index]
                if not trigram_dict.has_key(bigram):
                    prob = 1 / vocab_size
                elif not trigram_dict.has_key(trigram):
                    prob = 1 / (trigram_dict[bigram] + vocab_size)
                else:
                    prob = (trigram_dict[trigram] + 1) / (trigram_dict[bigram] + vocab_size)
                log_prob = math.log(prob)
                total_log_prob += log_prob
                num_tokens += 1
        h = -1.0 * total_log_prob / num_tokens
        perplexity = math.exp(h)

    with open(output_path, "w") as f:
        f.writelines(str(perplexity))


else:
    print("Not implemented!! ")
