import pandas as pd
import matplotlib.pyplot as plt

# Load the DataFrame
data = pd.DataFrame({
    'Company': ['Seclore', 'Citrix', 'Axway', 'GoAnywhere', 'PreVeil', 'Egress', 'Egnyte', 'Echoworx', 'Kiteworks'],
    'ORG': [8, 5, 17, 27, 13, 19, 10, 13, 12],
    'PERSON': [4, 7, 3, 7, 6, 9, 5, 13, 2],
    'ORDINAL': [1, 0, 0, 1, 1, 0, 1, 0, 3],
    'WORK_OF_ART': [1, 5, 2, 3, 8, 2, 3, 1, 8],
    'GPE': [1, 0, 0, 2, 0, 0, 0, 0, 2],
    'CARDINAL': [3, 4, 0, 3, 10, 2, 5, 11, 5],
    'DATE': [2, 0, 2, 4, 1, 5, 1, 5, 1],
    'PRODUCT': [0, 6, 0, 4, 1, 0, 0, 0, 2],
    'LOC': [0, 1, 0, 0, 0, 1, 0, 0, 1],
    'PERCENT': [0, 1, 0, 0, 0, 1, 0, 0, 0],
    'NORP': [0, 0, 1, 0, 1, 0, 0, 0, 1],
    'TIME': [0, 0, 1, 0, 1, 0, 0, 0, 0],
    'MONEY': [0, 0, 0, 1, 0, 0, 0, 2, 0],
    'FAC': [0, 0, 0, 2, 0, 0, 0, 2, 0],
    'LAW': [0, 0, 0, 0, 1, 1, 0, 0, 0]
})

# Plot bar charts and export them to individual png files
for col in data.columns[1:]:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(data['Company'], data[col])
    ax.set_title(col)
    ax.set_xlabel('Company')
    ax.set_ylabel('Count')
    ax.set_xticklabels(data['Company'], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{col}.png')
    plt.close()
