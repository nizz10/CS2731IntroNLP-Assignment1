# Author: Yuhuan Jiang
# Version: 1.0
# Language: Python 2

import argparse
import pickle
import math

argument_parser = argparse.ArgumentParser("Genre detection task runner. ")
argument_parser.add_argument('--wsjmodel', help='The path to the directory of the model files for the WSJ language model.', required=True)
argument_parser.add_argument('--sbmodel', help='The path to the directory of the model files for the Switchboard language model.', required=True)
argument_parser.add_argument('-i', '--input', help='The path to the file containing the sentences to be labeled. ', required=True)
argument_parser.add_argument('-o', '--output', help='The path to the file of the answer output of this script. ', required=True)

args = argument_parser.parse_args()

# Use these variables to detect the model requested by the command line user
wsj_model_dir = args.wsjmodel
sb_model_dir = args.sbmodel
input_path = args.input
output_path = args.output

print("Running the genre detection task with wsj model found at " + wsj_model_dir)
print("                                      sb  model found at " + sb_model_dir)
print("  using the test sentences found at " + input_path)
print("The answers will be output at " + output_path)
print("")

# START OF YOUR IMPLEMENTATION

snts = [line.rstrip('\n').split(' ') for line in open(input_path)]

with open(wsj_model_dir + "/model_type.txt") as f:
    model_type = f.readline().strip("\n")

# Load wsj model

with open(wsj_model_dir + "/vocabulary.pkl", "rb") as f:
    wsj_dict = pickle.load(f)
if model_type == "3" or model_type == "3s":
    with open(wsj_model_dir + "/trigram_counts.pkl", "rb") as f:
        wsj_trigram_dict = pickle.load(f)

# Load sb model

with open(sb_model_dir + "/vocabulary.pkl", "rb") as f:
    sb_dict = pickle.load(f)

if model_type == "3" or model_type == "3s":
    with open(sb_model_dir + "/trigram_counts.pkl", "rb") as f:
        sb_trigram_dict = pickle.load(f)


labels = []

if model_type == "1":

    for snt in snts:

        wsj_context_size = float(sum(wsj_dict.values()))
        sb_context_size = float(sum(sb_dict.values()))

        wsj_total_prob = 0.0

        sb_total_prob = 0.0

        for w in snt:
            # For wsj
            if not wsj_dict.has_key(w):
                wsj_prob = wsj_dict["<unk>"] / wsj_context_size
            else:
                wsj_prob = wsj_dict[w] / wsj_context_size

            wsj_total_prob += wsj_prob

            # For sb
            if not sb_dict.has_key(w):
                sb_prob = sb_dict["<unk>"] / sb_context_size
            else:
                sb_prob = sb_dict[w] / sb_context_size

            sb_total_prob += sb_prob


        if wsj_total_prob < sb_total_prob:
            labels.append("sb")
        elif wsj_total_prob > sb_total_prob:
            labels.append("wsj")

elif model_type == "3":
    for snt in snts:

        wsj_total_prob = 0.0
        sb_total_prob = 0.0

        for index in range(len(snt)):
            if index > 1:

                rep_snt = snt

                if not wsj_dict.has_key(snt[index]):
                    snt[index] = "<unk>"
                bigram = snt[index - 2] + " " + snt[index - 1]
                trigram = snt[index - 2] + " " + snt[index - 1] + " " + snt[index]

                # For wsj
                if not wsj_trigram_dict.has_key(bigram) or not wsj_trigram_dict.has_key(trigram):
                    wsj_prob = 0.0
                else:
                    wsj_prob = wsj_trigram_dict[trigram] / wsj_trigram_dict[bigram]

                wsj_total_prob += wsj_prob

                # For sb
                if not sb_dict.has_key(rep_snt[index]):
                    rep_snt[index] = "<unk>"
                bigram = rep_snt[index - 2] + " " + rep_snt[index - 1]
                trigram = rep_snt[index - 2] + " " + rep_snt[index - 1] + " " + rep_snt[index]
                if not sb_trigram_dict.has_key(bigram) or not sb_trigram_dict.has_key(trigram):
                    sb_prob = 0.0
                else:
                    sb_prob = sb_trigram_dict[trigram] / sb_trigram_dict[bigram]

                sb_total_prob += sb_prob

        # Skip if both probabilities equal to each other (when both are zeros)
        if wsj_total_prob < sb_total_prob:
            labels.append("sb")
        elif wsj_total_prob > sb_total_prob:
            labels.append("wsj")


elif model_type == "3s":
    for snt in snts:

        wsj_vocab_size = float(len(wsj_dict.keys()) - 1)    # minus one <s>
        sb_vocab_size = float(len(sb_dict.keys()) - 1)      # minus one <s>

        wsj_total_prob = 0.0
        sb_total_prob = 0.0

        for index in range(len(snt)):
            if index > 1:

                rep_snt = snt

                # For wsj
                if not wsj_dict.has_key(snt[index]):
                    snt[index] = "<unk>"
                bigram = snt[index - 2] + " " + snt[index - 1]
                trigram = snt[index - 2] + " " + snt[index - 1] + " " + snt[index]

                if not wsj_trigram_dict.has_key(bigram):
                    wsj_prob = 1 / wsj_vocab_size
                elif not wsj_trigram_dict.has_key(trigram):
                    wsj_prob = 1 / (wsj_trigram_dict[bigram] + wsj_vocab_size)
                else:
                    wsj_prob = (1 + wsj_trigram_dict[trigram]) / (wsj_trigram_dict[bigram] + wsj_vocab_size)

                wsj_total_prob += wsj_prob

                # For sb
                if not sb_dict.has_key(rep_snt[index]):
                    rep_snt[index] = "<unk>"
                bigram = rep_snt[index - 2] + " " + rep_snt[index - 1]
                trigram = rep_snt[index - 2] + " " + rep_snt[index - 1] + " " + rep_snt[index]

                if not sb_trigram_dict.has_key(bigram):
                    sb_prob = 1 / sb_vocab_size
                elif not sb_trigram_dict.has_key(trigram):
                    sb_prob = 1 / (sb_trigram_dict[bigram] + sb_vocab_size)
                else:
                    sb_prob = (1 + sb_trigram_dict[trigram]) / (sb_trigram_dict[bigram] + sb_vocab_size)

                sb_total_prob += sb_prob


        if wsj_total_prob < sb_total_prob:
            labels.append("sb")
        elif wsj_total_prob > sb_total_prob:
            labels.append("wsj")

with open(output_path, "w") as f:
    f.writelines("\n".join(labels))
