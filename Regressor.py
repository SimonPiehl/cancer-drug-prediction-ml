import pandas as pd
import sys

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_val_score

# Einlesen der Daten
gene_expression = pd.read_csv(sys.argv[1], delimiter='\t', header=0, index_col=0)
drug_sensitivity = pd.read_csv(sys.argv[2], delimiter='\t', header=0, index_col=0)
train_ids = pd.read_csv(sys.argv[3], header=None, delimiter='\t', index_col=0)
test_ids = pd.read_csv(sys.argv[4], header=None, delimiter='\t', index_col=0)

# Feature Selection anhand der gegebenen Gen-Liste
feature_selection_names = 'cancer_gene_list.txt'
with open(feature_selection_names, 'r') as f:
    selected_genes = [line.strip() for line in f]
selected_cols = [col for col in gene_expression.columns if col in selected_genes]
gene_expression_selected = gene_expression[selected_cols]

# Train und Test IDs in Liste laden
train_ids_list = train_ids.index.tolist()
test_ids_list = test_ids.index.tolist()

# Herausfiltern der IDs, die nicht in beiden Sets vorhanden sind
missing_train_ids_gene = [id_ for id_ in train_ids_list if id_ not in gene_expression.index]
missing_train_ids_drug = [id_ for id_ in train_ids_list if id_ not in drug_sensitivity.index]
train_ids_list = [id_ for id_ in train_ids_list if id_ not in missing_train_ids_gene and id_ not in missing_train_ids_drug]
missing_test_ids_gene = [id_ for id_ in test_ids_list if id_ not in gene_expression.index]
missing_test_ids_drug = [id_ for id_ in test_ids_list if id_ not in drug_sensitivity.index]
test_ids_list = [id_ for id_ in test_ids_list if id_ not in missing_test_ids_gene and id_ not in missing_test_ids_drug]

# Filtern der Datensätze mit übrigen IDs
gene_train = gene_expression_selected.loc[train_ids_list]
gene_test = gene_expression_selected.loc[test_ids_list]
drug_train = drug_sensitivity.loc[train_ids_list]
drug_test = drug_sensitivity.loc[test_ids_list]

# Kombinieren der Daten
train_data = pd.concat([gene_train, drug_train], axis=1).dropna()
test_data = pd.concat([gene_test, drug_test], axis=1).dropna()

best_params_list = []
# Durchführen des Modells für alle Medikamente
drug_names = drug_sensitivity.columns[0:]
for drug_name in drug_sensitivity.columns[0:]:

    combined_train = pd.concat([gene_train, drug_train[drug_name]], axis=1).dropna()
    combined_test = pd.concat([gene_test, drug_test[drug_name]], axis=1).dropna()

    # Aufteilen in Set X und Zielvariable Y
    X_train = combined_train.drop(drug_name, axis=1)
    Y_train = combined_train[drug_name]
    X_test = combined_test.drop(drug_name, axis=1)
    Y_test = combined_test[drug_name]

    # Festlegen der Modell-Parameter
    parameter_grid = {
        'n_estimators': list(range(2, 22, 2)),
        'max_features': list(range(40, 201, 20)),
    }

    # RandomFOrest Regression Modell initialisieren
    rf = RandomForestRegressor()

    # Ausführen der 5-fachen Kreuzvalidierung auf den Trainingsdatensatz
    clf = GridSearchCV(estimator=rf, param_grid=parameter_grid, cv=5)
    clf.fit(X_train, Y_train)

    # Auslesen des besten Modells1
    best_estimator = clf.best_estimator_
    best_params_list.append({
        'drug_name': drug_name,
        'best_params': clf.best_params_
    })

    # Berechnen des MSE
    Y_pred = best_estimator.predict(X_test)
    Y_pred_cv = best_estimator.predict(X_train)

    mse = mean_squared_error(Y_test, Y_pred)
    cv_mse_scores = cross_val_score(best_estimator, X_train, Y_train, cv=5, scoring='neg_mean_squared_error')
    mse_cv = -cv_mse_scores.mean()

    # Ausgabe Predicted Response
    output_data_test = pd.DataFrame({
        'Predicted Response': Y_test.index,
        '': Y_pred,
    })
    output_data_test.to_csv(f'{drug_name}_regression_test.txt', sep='\t', index=False)

    output_data_train = pd.DataFrame({
        'Predicted Response':  Y_train.index,
        '': Y_pred_cv,
    })
    output_data_train.to_csv(f'{drug_name}_regression_train.txt', sep='\t', index=False)

    # Ausgabe der Ergebnisse
    results_data = pd.DataFrame({
        'MSE': ['Cv error', 'test error'],
        '': [mse, mse_cv],
    })
    results_data.to_csv(f'{drug_name}_regression_results.txt', sep='\t', index=False)

# Ausgabe der besten Parameter für alle Modelle
best_params_df = pd.DataFrame(best_params_list)
best_params_df.to_csv('best_params_reg.txt', index=False)
