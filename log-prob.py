# Author: Yuhuan Jiang
# Version: 1.0
# Language: Python 2

import argparse
import math

# The following lines parse the command line arguments for you. You may ignore this part.
argument_parser = argparse.ArgumentParser("Log probability script. ")
argument_parser.add_argument('-m', '--model', help='The path to the directory containing the necessary files to recreate the language model.', required=True)
argument_parser.add_argument('-i', '--input', help='The path to the input file containing the n-grams to query. ', required=True)
argument_parser.add_argument('-o', '--output', help='The path to the output file containing the log probabilities for the input n-grams. ', required=True)
args = argument_parser.parse_args()

# The following variables are created for your convenience.
# They are the values from the command line input.
model_dir = args.model
ngram_input_path = args.input
log_prob_output_path = args.output

# Prints out what this script does
print("Computing the log-probabilities of the input n-grams in file: " + ngram_input_path)
print("    using the language model recreated from the directory: " + model_dir)
print("Result will be stored in the file: " + log_prob_output_path)
print("")



# START OF YOUR IMPLEMENTATION

# Recreate the LM
with open(model_dir + "/model_type.txt") as f:
    model_type = f.readline()

with open(model_dir + "/vocab_size.txt") as f:
    vocab_size = float(f.readline())

if model_type == "1" or model_type == "3" or model_type == "s3":
    # Get the ngram counts
    with open(model_dir + "/ngram_counts.pkl", "rb") as f:
        dict = pickle.load(f)

queried_ngrams = [tuple(line.rstrip('\n').split(' ')) for line in open(ngram_input_path)]


if model_type == "dummy":
    # The dummy model always assigns 1/V to any unigram
    log_probs = [-math.log(vocab_size) for ngram in queried_ngrams]
    with open(log_prob_output_path, "w") as f:
        f.write("\n".join([str(x) for x in log_probs]))

elif model_type == "1":
    # unsmoothed unigram
    # P(w1w1...wk) = P(w1)P(w2)...P(wk)

    log_probs = [math.log(dict[ngram] / vocab_size) for ngram in queried_ngrams]

    with open(log_prob_output_path, "w") as f:
        for index, ngram in enumerate(queried_ngrams):
            log_probs[i] = math.log(dict[ngram] / vocab_size)
        f.write("\n".join([str(x) for x in log_probs]))


else:
    print("Not implemented!! ")
