import pandas as pd
import spacy

# load pre-trained English language model
nlp = spacy.load('en_core_web_sm')

# define function to perform NER on text data and return a dictionary of entity counts
def ner(text):
    doc = nlp(text)
    entities = [ent.label_ for ent in doc.ents]
    counts = {}
    for entity in entities:
        counts[entity] = counts.get(entity, 0) + 1
    return counts

# read the input CSV file into a Pandas DataFrame
file = r"C:\Users\yotam.twersky\Downloads\Competitor Content.csv"
df = pd.read_csv(file)
# apply NER function to text data and create a new DataFrame to store the results
ner_results = pd.DataFrame()
ner_results['Company'] = df['Company']

for idx, row in df.iterrows():
    entity_counts = ner(row['Homepage Text'])
    for entity, count in entity_counts.items():
        ner_results.loc[idx, entity] = count

# fill NaN values with 0
ner_results = ner_results.fillna(0)

# save the results to a CSV file
ner_results.to_csv('ner_results.csv', index=False)
