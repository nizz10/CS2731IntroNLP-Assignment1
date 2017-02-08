# Author: Yuhuan Jiang
# Version: 1.0
# Language: Python 2

import argparse

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
labels = ["wsj" for snt in snts]

with open(output_path, "w") as f:
    f.writelines("\n".join(labels))
