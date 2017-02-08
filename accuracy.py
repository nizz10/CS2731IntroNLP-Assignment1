# Author: Yuhuan Jiang
# Version: 1.0
# Language: Python 2

import argparse

argument_parser = argparse.ArgumentParser("Accuracy script. ")
argument_parser.add_argument('-g', '--gold', help='The path to the gold answer file.', required=True)
argument_parser.add_argument('-a', '--auto', help='The path to the auto answer file.', required=True)
args = argument_parser.parse_args()

gold_path = args.gold
auto_path = args.auto

golds = [line.rstrip('\n') for line in open(gold_path)]
autos = [line.rstrip('\n') for line in open(auto_path)]

num_matchings = 0
num_total = 0

for g, a in zip(golds, autos):
    num_total += 1
    if g == a:
        num_matchings += 1

print("Accuracy = " + str(num_matchings) + "/" + str(num_total) + " (" + "{0:.2f}%".format(num_matchings * 100 / num_total) + ")")
