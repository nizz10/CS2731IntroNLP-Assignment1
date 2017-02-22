# Implementations

* The dummy language model is a uniform unigram language model. It assigns the probability of $\frac{1}{V}$ to every unigram, where $V$ is the vocabulary size.

* The unsmoothed unigram language model assigns the probability of $\frac{Count(wi)}{Context_Size}$ to wi.

* The unsmoothed trigram language model assigns the probability of $\frac{Count(wi-2 wi-1 wi)}{Count(wi-2 wi-1)}$ to trigram "wi-2 wi-1 wi".

* The smoothed trigram language model assigns the probability of $\frac{Count(wi-2 wi-1 wi) + 1}{Count(wi-1 wi-1) + V}$ to trigram "wi-2 wi-1 wi", where $V$ is the vocabulart size.

## Exercises

### Training an LM

To train an unsmoothed unigram model on the WSJ data and save the model in `1_wsj/`, run the follwing
(same with the unsmoothed trigram model and smoothed trigram model)
```bash
python2 train.py -t 1 -i data/wsj/train.txt -m 1_wsj/
python2 train.py -t 1 -i data/sb/train.txt -m 1_sb/

python2 train.py -t 3 -i data/wsj/train.txt -m 3_wsj/
python2 train.py -t 3 -i data/sb/train.txt -m 3_sb/

python2 train.py -t 3s -i data/wsj/train.txt -m 3s_wsj/
python2 train.py -t 3s -i data/sb/train.txt -m 3s_sb/
```

### Computing the Perplexity

To compute the perplexity, on the WSJ data, run

```bash
python2 test.py -m 1_wsj/ -i data/wsj/test.txt -o 1_wsj_perplexity.txt
python2 test.py -m 1_sb/ -i data/sb/test.txt -o 1_sb_perplexity.txt

python2 test.py -m 3_wsj/ -i data/wsj/test.txt -o 3_wsj_perplexity.txt
python2 test.py -m 3_sb/ -i data/sb/test.txt -o 3_sb_perplexity.txt

python2 test.py -m 3s_wsj/ -i data/wsj/test.txt -o 3s_wsj_perplexity.txt
python2 test.py -m 3s_sb/ -i data/sb/test.txt -o 3s_sb_perplexity.txt
```

A file named `1_wsj_perplexity.txt` is created, containing the perplexity score.

### Obtaining Log-Probabilities

```bash
python log-prob.py -m 3s_wsj/ -i trigrams-to-check.txt -o log-probs.txt
```

A file `log-probs.txt` containing the log probabilities is created.

### Running the Genre Detection Task

```
python2 run-task.py --wsjmodel 1_wsj/ --sbmodel 1_sb/ -i data/genre-task/mixed-sentences.txt -o answers.txt

python2 run-task.py --wsjmodel 3_wsj/ --sbmodel 3_sb/ -i data/genre-task/mixed-sentences.txt -o answers.txt

python2 run-task.py --wsjmodel 3s_wsj/ --sbmodel 3s_sb/ -i data/genre-task/mixed-sentences.txt -o answers.txt
```

A file names `answers.txt` will be created.

To evaluate the performance, run

```
python2 accuracy.py -g data/genre-task/gold.txt -a answers.txt
```
