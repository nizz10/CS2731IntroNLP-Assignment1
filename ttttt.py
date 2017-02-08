# Author: Yuhuan Jiang
# Version: 1.0
# Language: Python 2

import argparse
import os
import pickle

# The following lines parse the command line arguments for you. You may ignore this part.
argument_parser = argparse.ArgumentParser("Language model trainer. ")
argument_parser.add_argument('-t', '--type', help='The type of the model. Possible values: 1, 3, 3s, dummy', required=True)
argument_parser.add_argument('-i', '--input', help='The path to the input file containing the training sentences. ', required=True)
argument_parser.add_argument('-m', '--model', help='The path to the directory of the model files.', required=True)
args = argument_parser.parse_args()

# The following variables are created for your convenience.
# They are the values from the command line input.
model_type = args.type
training_data_path = args.input
model_dir = args.model

# Doing some sanity check for you
if model_type not in ['1', '3', '3s', 'dummy']:
    print("The model type " + model_type + " is not supported.")
    exit(1)

# Prints out what this script does
print("Training a " + model_type + " model")
print("    using training data found at " + training_data_path)
print("After training, the model files will be stored at this directory: " + model_dir)
print("")


# START OF YOUR IMPLEMENTATION
# Utilities
def check_oov(dict):
    for key in dict.keys():
        if dict[key] == 1:
            del dict[key]
            if not dict.has_key("<unk>"):
                dict["<unk>"] = 1
            else:
                dict["<unk>"] += 1
    return dict



# Create the folder that holds the necessary files for fast re-building this LM later (i.e., a dump of the trained LM)
# Notice: this will create whatever user specifies as model_dir.
#         Always use the variable model_dir, and never hard code any absolute paths.
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# This variable holds the training sentences. Example content::
#
# train_snts = [
#     ["john", "has", "a",  "cat", "."],
#     ["mary", "has", "a",  "dog", "."],
#     ["john", "'s", "cat",  "is", "not", "a", "dog" "."],
#     ["mary", "'s", "dog",  "is", "not", "a", "cat" "."]"
# ]
#
training_sentences = [line.rstrip('\n').split(' ') for line in open(training_data_path)]


if model_type == "dummy":
    # The dummy model is a very naive unigram model.
    # It assigns 1/V as the probability for every unigram.
    #   where V is the vocabulary size

    words = [w for snt in training_sentences for w in snt]
    vocab = set()
    for w in words:
        vocab.add(w)

    vocab_size = len(vocab)

    # When we re-create the LM later, we will need the model type and the vocabulary size.
    # We don't save the probabilities here, because we will be computing the probabilities only the fly later in other scripts.
    with open(model_dir + "/model_type.txt", "w") as f:
        f.writelines([model_type])

    with open(model_dir + "/vocab_size.txt", "w") as f:
        f.writelines([str(vocab_size)])
elif model_type == "1":
    print("Training the unsmoothed unigram model...")
    words = [w for snt in training_sentences for w in snt]
    vocab = set()
    unigram_dict = {}
    for w in words:
        vocab.add(w)
        if not unigram_dict.has_key(w):
            unigram_dict[w] = 1;
        else:
            unigram_dict[w] += 1;
    # Check and deal with OOV Words
    unigram_dict = check_oov(unigram_dict)
    vocab_size = len(vocab)

    ########### For testing
    for w in unigram_dict:
        print(w, unigram_dict[w])

    with open(model_dir + "/model_type.txt", "w") as f:
        f.writelines([model_type])
        f.writelines("\nunsmoothed")    # for line in f:
                                        #     print(line.rstrip())
    with open(model_dir + "/vocab_size.txt", "w") as f:
        f.writelines([str(vocab_size)])     # The first line of the raw_count file is the counts of contexts

    with open(model_dir + "/ngram_counts.pkl", "wb") as f:
        pickle.dump(unigram_dict, f)

    # No smooth counts for this model.

# elif model_type == "3":
#     print("This is the unsmoothed trigram model")
#     words = [w for snt in training_sentences for w in snt]
#     vocab = set()
#     for w in words:
#         vocab.add(w)
#     vocab_size = len(vocab)

#     with open(model_dir + "/model_property.txt", "w") as f:
#         f.writelines([model_type])
#         f.writelines("unsmoothed")

#     with open(model_dir + "/raw_count.txt", "w") as f:
#         f.writelines([str(vocab_size)])


# elif model_type == "s3":
#     print("This is the smoothed trigram model")
#     words = [w for snt in training _sentences for w in snt]
#     vocab = set()
#     for w in words:
#         vocab.add(w)
#     vocab_size = len(vocab)

#     with open(model_dir + "/model_property.txt", "w") as f:
#         f.writelines([model_type])
#         f.writelines("smoothed")
#     with open(model_dir + "/raw_count.txt", "w") as f:
#         f.writelines([str(vocab_size)])

else:
    print("Not implemented yet.")
