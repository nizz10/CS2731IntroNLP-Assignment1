# A Dummy Implementation

The dummy language model is a uniform unigram language model. IT assigns the probability of $\frac{1}{V}$ to every unigram, where $V$ is the vocabulary size.

## Exercises

### Training an LM

To train a dummy model on the WSJ data and save the model in `dummy_wsj/`, run the follwing

```bash
python2 train.py -t 1 -i data/wsj/train.txt -m 1_wsj/
python2 train.py -t 1 -i data/sb/train.txt -m 1_sb/

python2 train.py -t 3 -i data/wsj/train.txt -m 3_wsj/
python2 train.py -t 3 -i data/sb/train.txt -m 3_sb/

python2 train.py -t 3s -i data/wsj/train.txt -m 3s_wsj/
python2 train.py -t 3s -i data/sb/train.txt -m 3s_sb/
```

Notice that a directory `dummy_wsj/` has been created. Inside, it has all the information that the other scripts will need to re-create this dummy LM. The exact information saved are:

- `dummy_wsj/model_type.txt`: The type of the LM. In this case, it is `dummy`.
- `dummy_wsj/vocab_size.txt`: The vocabulary size.

Of course, your LM should depend on much more information than what is needed for `dummy`.


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

Notice that a file named `dummy_wsj_perplexity.txt` is created, containing the perplexity score. As expected, the perplexity of a uniform unigram LM equals to the vocabulary size. Open `dummy_wsj_perplexity.txt` to compare the number with what was saved in `dummy_wsj/vocab_size.txt`.

### Obtaining Log-Probabilities

In building your LM, you might want to check the probabilities that your model assigns to some ngrams. To do that, create a file that contains ngrams on each line, such as

```
cat
dog
john
united
states
```

if you want to examine a unigram model (which is our case. If you want to examine a trigram model, then every line should contain a trigram). Save this file as `unigrams-to-check`. Next, run

```bash
python log-prob.py -m dummy_wsj/ -i unigrams-to-check.txt -o log-probs.txt
```

Notice that a file `log-probs.txt` containing the log probabilities is created. Open it to verify.

### Running the Genre Detection Task

To train a dummy LM for the other corpus, Switchboard, run:

```
python train.py -t dummy -i data/sb/train.txt -m dummy_sb/
```

You can use similar commands to see its perplexity on the testing set, or examine the log-probabilities it assigns to some unigrams. But we will skip that here. We new run the genre detection task:

```
python2 run-task.py --wsjmodel 1_wsj/ --sbmodel 1_sb/ -i data/genre-task/mixed-sentences.txt -o answers.txt

python2 run-task.py --wsjmodel 3_wsj/ --sbmodel 3_sb/ -i data/genre-task/mixed-sentences.txt -o answers.txt

python2 run-task.py --wsjmodel 3s_wsj/ --sbmodel 3s_sb/ -i data/genre-task/mixed-sentences.txt -o answers.txt
```

Notice that a file names `answers.txt` will be created. Open it, and you will see all `wsj`s, because that is what this dummy implementation of `run-task.py` does.

To evaluate the performance, run

```
python2 accuracy.py -g data/genre-task/gold.txt -a answers.txt
```

The screen will print

```
Accuracy = 500/1000 (50.00%)
```
