# Drug Sensitivity Prediction

ML pipeline to predict cancer drug sensitivity from gene expression data, developed as part of the Bioinformatics II course at Universität des Saarlandes.

Data: GDSC database (1000 cell lines, 10 drugs)

## Models

Both classifier and regressor use Random Forest with GridSearchCV hyperparameter tuning and 5-fold cross-validation. The classifier is evaluated using MCC, sensitivity, and specificity; the regressor using MSE. Feature selection is based on a list of known cancer-relevant genes (`cancer_gene_list.txt`).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Split data (once)
python SplitTestTraining.py

# Classifier
python Klassifikator.py expression_data.txt binary_data.txt train_ids.txt test_ids.txt

# Regressor
python Regressor.py expression_data.txt IC50_data.txt train_ids.txt test_ids.txt
```

## Note

Raw data (GDSC) is not included in this repository. Required files: `expression_data.txt`, `binary_data.txt`, `IC50_data.txt`, `cancer_gene_list.txt`.

## Technologies

Python · scikit-learn · pandas
