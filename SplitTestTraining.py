import pandas as pd
from sklearn.model_selection import train_test_split

# Einleses des Datensatzes
df = pd.read_csv('expression_data.txt', delimiter='\t', header=0)

# Extrahieren der ersten Spalte
cosmic_ids = df.index.to_series()

# Shuffeln der Cosmic-ID
cosmic_ids = cosmic_ids.sample(frac=1, random_state=42).reset_index(drop=True)

# Aufteilen in 20% Test, 80% training
train_ids, test_ids = train_test_split(cosmic_ids, test_size=0.2, random_state=0)

# Schreiben von Test- und Training-IDs in seperate Dateien
with open('train_ids.txt', 'w') as f:
    for item in train_ids:
        f.write("%s\n" % item)

with open('test_ids.txt', 'w') as f:
    for item in test_ids:
        f.write("%s\n" % item)
