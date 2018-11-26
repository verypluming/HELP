# LELP
LELP: Linguistics-oriented Corpus for Encouraging Neural Models to Learn Logical Phenomena

## Environment
```bash
git clone https://github.com/verypluming/LELP.git
cd LELP
pyenv virtualenv 3.4.6 lelp
pyenv activate lelp
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet')"
```

## Installing [C&C parser](http://www.cl.cam.ac.uk/~sc609/candc-1.00.html) and [Parallel Meaning Bank (PMB)](http://pmb.let.rug.nl/)
Please download C&C, set it up, and create a file `data/parser_location.txt` with the path to the C&C parser.
Then, please download PMB and put it to `data/` directry.

```bash
echo "candc:/path/to/candc-1.00/" > data/parser_location.txt
```

## Data creation

```bash
python scripts/create_dataset_PMB.py
```

## Data

```bash
output_en/
pmb_train.tsv - Train split of LELP
conj_test.tsv, disj_test.tsv, downwad_test.tsv, upward_test.tsv - Test split of LELP
```

## Reference

