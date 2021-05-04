

## Twitter  Datasets

Download the tweet datasets from here:
http://www.da.inf.ethz.ch/teaching/2018/CIL/material/exercise/twitter-datasets.zip


The dataset should have the following files:
- sample_submission.csv
- train_neg.txt :  a subset of negative training samples
- train_pos.txt: a subset of positive training samples
- test_data.txt:
- train_neg_full.txt: the full negative training samples
- train_pos_full.txt: the full positive training samples

## Build the Co-occurence Matrix

To build a co-occurence matrix, run the following commands.  (Remember to put the data files
in the correct locations)

Note that the cooc.py script takes a few minutes to run, and displays the number of tweets processed.

- build_vocab.sh
- cut_vocab.sh
- python3 pickle_vocab.py
- python3 cooc.py

##  Template for Glove Question

Your task is to fill in the SGD updates to the template
glove_template.py

Once you tested your system on the small set of 10% of all tweets, we suggest you run on the full datasets train_pos_full.txt, train_neg_full.txt
