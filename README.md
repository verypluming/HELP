# HELP
HELP: A Dataset for Handling Entailments with Lexical and Logical Phenomena

## Environment
```bash
git clone https://github.com/verypluming/HELP.git
cd HELP
pyenv virtualenv 3.4.6 help
pyenv activate help
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
pmb_train.tsv
```

## Reference
* Hitomi Yanaka, Koji Mineshima, Daisuke Bekki, Kentaro Inui, Satoshi Sekine, Lasha Abzianidze, and Johan Bos. HELP: A Dataset for Identifying Shortcomings of Neural Models in Monotonicity Reasoning. Proceedings of the Eighth Joint Conference on Lexical and Computational Semantics (\*SEM2019), Mineapolis, USA, 2019. [arXiv](https://arxiv.org/pdf/XXX.pdf)

```
@InProceedings{yanaka-EtAl:2019:starsem,
  author    = {Yanaka, Hitomi and Mineshima, Koji  and  Bekki, Daisuke and Inui, Kentaro and Sekine, Satoshi and Abzianidze, Lasha and Bos, Johan},
  title     = {HELP: A Dataset for Identifying Shortcomings of Neural Models in Monotonicity Reasoning},
  booktitle = {Proceedings of the Eighth Joint Conference on Lexical and Computational Semantics (*SEM2019)},
  year      = {2019},
}
```

