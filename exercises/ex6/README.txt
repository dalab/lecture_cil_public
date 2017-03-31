Download the tweet datasets from here:
http://www.da.inf.ethz.ch/teaching/2017/CIL/material/exercise/twitter-datasets.zip


To build a co-occurence matrix, run the following commands. Note that the cooc.py script takes a few minutes to run, and displays the number of tweets processed.

build_vocab.sh
cut_vocab.sh
python3 pickle_vocab.py
python3 cooc.py


Your task is to fill in the SGD updates to the template
glove_template.py

Once you tested your system on the small set of 10% of all tweets, we suggest you run on the full datasets train_pos_full.txt, train_neg_full.txt
