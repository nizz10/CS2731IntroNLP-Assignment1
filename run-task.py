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

# Load wsj model
with open(wsj_model_dir + "/vocab_size.txt") as f:
    wsj_vocab_size = float(f.readline())

with open(wsj_model_dir + "/ngram_counts.pkl", "rb") as f:
    wsj_dict = pickle.load(f)

# Load sb model
with open(sb_model_dir + "/ngram_counts.pkl", "rb") as f:
    sb_dict = pickle.load(f)

with open(sb_model_dir + "/vocab_size.txt") as f:
    sb_vocab_size = float(f.readline())

labels = []

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
            wsj_prob = wsj_dict["<unk>"] / wsj_vocab_size
        else:
            wsj_prob = wsj_dict[w] / wsj_vocab_size
        wsj_log_prob = math.log(wsj_prob)
        wsj_total_log_prob += wsj_log_prob
        # For sb
        if not sb_dict.has_key(w):
            sb_prob = sb_dict["<unk>"] / sb_vocab_size
        else:
            sb_prob = sb_dict[w] / sb_vocab_size
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

with open(output_path, "w") as f:
    f.writelines("\n".join(labels))
