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
#labels = ["wsj" for snt in snts]
with open(wsj_model_dir + "/model_type.txt") as f:
    model_type = f.readline().strip("\n")

print(model_type)

# Load wsj model
with open(wsj_model_dir + "/context_size.txt") as f:
    wsj_context_size = float(f.readline())
if model_type == "1":
    with open(wsj_model_dir + "/ngram_counts.pkl", "rb") as f:
        wsj_dict = pickle.load(f)
if model_type == "3":
    with open(wsj_model_dir + "/trigram_counts.pkl", "rb") as f:
        wsj_trigram_dict = pickle.load(f)
    with open(wsj_model_dir + "/bigram_counts.pkl", "rb") as f:
        wsj_bigram_dict = pickle.load(f)
#if model_type == "1" or model_type == "3" or model_type == "s3":
# Load sb model

if model_type == "1":
    with open(sb_model_dir + "/ngram_counts.pkl", "rb") as f:
        sb_dict = pickle.load(f)

if model_type == "3":
    with open(sb_model_dir + "/trigram_counts.pkl", "rb") as f:
        sb_trigram_dict = pickle.load(f)
    with open(sb_model_dir + "/bigram_counts.pkl", "rb") as f:
        sb_bigram_dict = pickle.load(f)
with open(sb_model_dir + "/context_size.txt") as f:
    sb_context_size = float(f.readline())

labels = []
if model_type == "1":
    for snt in snts:
        wsj_prob = 0.0
        wsj_log_probs = 0.0
        wsj_total_log_prob = 0.0
        sb_prob = 0.0
        sb_log_probs = 0.0
        sb_total_log_prob = 0.0
        for w in snt:
            # For wsj
            if not wsj_dict.has_key(w):
                wsj_prob = wsj_dict["<unk>"] / wsj_context_size
            else:
                wsj_prob = wsj_dict[w] / wsj_context_size
            wsj_log_prob = math.log(wsj_prob)
            wsj_total_log_prob += wsj_log_prob
            # For sb
            if not sb_dict.has_key(w):
                sb_prob = sb_dict["<unk>"] / sb_context_size
            else:
                sb_prob = sb_dict[w] / sb_context_size
            sb_log_prob = math.log(sb_prob)
            sb_total_log_prob += sb_log_prob
        wsj_result_prob = math.exp(wsj_total_log_prob)
        sb_result_prob = math.exp(sb_total_log_prob)

        if wsj_result_prob < sb_result_prob:
            labels.append("sb")
        elif wsj_result_prob > sb_result_prob:
            labels.append("wsj")
        else:
            labels.append("THIS IS A TIE")

elif model_type == "3":
    print("HERE")
    for snt in snts:
        wsj_prob = 0.0
        wsj_log_probs = 0.0
        wsj_total_log_prob = 0.0
        sb_prob = 0.0
        sb_log_probs = 0.0
        sb_total_log_prob = 0.0
        for index in range(len(snt)):
            if index > 1:
                bigram = snt[index - 2] + " " + snt[index - 1]
                trigram = snt[index - 2] + " " + snt[index - 1] + " " + snt[index]
                # For wsj
                if not wsj_bigram_dict.has_key(bigram):
                    wsj_prob = wsj_trigram_dict["<unk>"] / wsj_bigram_dict["<unk>"]
                elif not wsj_trigram_dict.has_key(trigram):
                    wsj_prob = wsj_trigram_dict["<unk>"] / wsj_bigram_dict[bigram]
                else:
                    wsj_prob = wsj_trigram_dict[trigram] / wsj_bigram_dict[bigram]

                if not wsj_prob == 0:
                    wsj_log_prob = math.log(wsj_prob)
                    wsj_total_log_prob += wsj_log_prob
                # For sb
                if not sb_bigram_dict.has_key(bigram):
                    sb_prob = sb_trigram_dict["<unk>"] / sb_bigram_dict["<unk>"]
                elif not sb_trigram_dict.has_key(trigram):
                    sb_prob = sb_trigram_dict["<unk>"] / sb_bigram_dict[bigram]
                else:
                    sb_prob = sb_trigram_dict[trigram] / sb_bigram_dict[bigram]
                if not sb_prob == 0:
                    sb_log_prob = math.log(sb_prob)
                    sb_total_log_prob += sb_log_prob
        wsj_result_prob = math.exp(wsj_total_log_prob)
        sb_result_prob = math.exp(sb_total_log_prob)
        print(wsj_result_prob)
        print(sb_result_prob)

        if wsj_result_prob < sb_result_prob:
            labels.append("sb")
        elif wsj_result_prob > sb_result_prob:
            labels.append("wsj")
        else:
            labels.append("THIS IS A TIE")

with open(output_path, "w") as f:
    f.writelines("\n".join(labels))
