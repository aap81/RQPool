# RQPool

## 1. Install Packages

Run the following command to install the required packages:

```bash
pip install numpy pandas matplotlib scipy scikit-learn tensorflow torch torch-geometric
```


## 2. Download Datasets

### a. TUDataset

1. Open the file `split_dataset.py` and specify which TUDatasets you need.
2. Run the script with:
```bash
python split_dataset.py
```

### b. OGBG Dataset

- OGBG datasets are downloaded on demand. If your chosen dataset name starts with `ogbg`, you do not need to perform any additional download steps.

## 3. Run RQPool

Pass any command-line arguments supported by `main.py` using the format `--<option> <value>`. For example, to run RQPool on the `ogbg_molhiv` dataset using sort-based inter-graph pooling for 100 epochs with printing enabled, execute:
```bash
python main.py --data ogbg_molhiv --intergraph sort --nepoch 100 --enableprint 1
```
