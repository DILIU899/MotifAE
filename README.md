# MotifAE

This is the code base for **MotifAE**

## Installation

```bash
conda create -n MotifAE python=3.9
conda activate MotifAE

pip install torch numpy pandas fair-esm click
```

### External software

#### Required

We use soft rank technique introduced in [this paper](https://arxiv.org/abs/2002.08871) to calculate the loss. Please download the externel code base from [here](https://github.com/google-research/fast-soft-sort).

#### Optional

[MMseqs2](https://github.com/soedinglab/MMseqs2) can be used to do the domain level train/test split. You can download the software using the following command.

```bash
conda activate MotifAE
conda install -c conda-forge -c bioconda mmseqs2
```

## Data

We get AFDB from [this paper](https://www.nature.com/articles/s41586-023-06510-w).

We get human domain stability data from [this paper](https://www.nature.com/articles/s41586-024-08370-4).


## Usage

### Precomputed ESM2 Embedding

Please first generate ESM2 embedding using the `utils/esm.inference.py`. The provided file should at least have `uniprotID` and `sequence` domains. You will need the precomputed representation for the model training.

### MotifAE

1. Prepare your data. The data should have at least the following fields:
- `split`: train/test split
- `uniprot`: the name of the sequence, which can be used to find the saved embedding. Not necessarily using `uniprot`, as long as consistent with `df_name_col_representative` field in the config file.

2. Please modify the config file in `MotifAE/config.py`. Be sure to at least modify the following fields
- `embed_logit_path_representative`: Path to precomputed ESM2 embedding dir
- `df_path_representative`: Path to metadata file. 
- `save_dir`: Path to where you save all the model results

3. Train the model by running `train_motifae.sh`

### MotifAE-G

1. Prepare your data (Please separate your train/valid/test files into individual files). The data should have at least the following fields:
- `uniprotID` and `domainID`: `uniprotID` corresponds to the id for the full length sequence and `domainID` corresponds to the id for the chunked domain sequence. The `uniprotID` is used to find the saved embedding and the `domainID` is designed just for index different domain DMS experiments since when uniprot sequence can have more than one domains.
- `seq.start`, `seq.end`: optional, needed when `chunk_seq` set to `true` in the config file. Chunk the embedding based on the position. 
- `ref`: wt aa
- `alt`: mut aa
- `pos`: mutation position, based on the sequence finally used (i.e if you use full length sequence to generate the embedding and chunk the sequence, then provide the relative position in the chunked sequence)
- `score`: label name, not necessarily using `score`, as long as consistent with `label` field in the config file.

We have a sample data file in `data/sample_data.csv`.

2. Please modify the config file. A sample config file is provided in `exp_setting/sample.yaml`. Be sure to at least modify the following fields
- `std_model_checkpoint`: Path to pretrained MotifAE model checkpoint
- `*_meta_data`: Path to metadata file. 
- `embedding_path`: Path to precomputed ESM2 embedding dir
- `log_dir`: Path to where you save all the model results

3. Go to `train_test_motifaeg.sh` for some predefined tasks you can use. Run the script after you change the `task_name` and `mode` fields.
