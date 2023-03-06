import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the CSV file
file = r"C:\Users\yotam.twersky\Downloads\Competitor Content.csv"
df = pd.read_csv(file)

# Named Entity Recognition (NER)
nlp = spacy.load('en_core_web_sm')

def ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

df['ner_entities'] = df['Homepage Text'].apply(lambda x: ner(x))

# Topic Modeling
def topic_modeling(text):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(text)
    lda = LatentDirichletAllocation(n_components=5, random_state=0)
    lda.fit(X)
    return vectorizer, lda, X

vectorizer, model, X = topic_modeling(df['Homepage Text'])
feature_names = vectorizer.get_feature_names()

topics = model.transform(X)
for idx, topic in enumerate(model.components_):
    df[f'Topic #{idx}'] = [', '.join([feature_names[i] for i in topic.argsort()[:-6:-1]]) for j in range(len(df))]

# Emotion Analysis
def extract_emotions(text):
    blob = TextBlob(text)
    emotions = {}
    for sentence in blob.sentences:
        emotion = sentence.sentiment.polarity
        if emotion > 0:
            emotions['happy'] = emotions.get('happy', 0) + emotion
        elif emotion < 0:
            emotions['sad'] = emotions.get('sad', 0) + abs(emotion)
        subjectivity = sentence.sentiment.subjectivity
        if subjectivity > 0.5:
            emotions['surprise'] = emotions.get('surprise', 0) + subjectivity
        else:
            emotions['disgust'] = emotions.get('disgust', 0) + (1 - subjectivity)
    return emotions

emotions_df = df['Homepage Text'].apply(lambda x: pd.Series(extract_emotions(x)))
df[['emotion_happy', 'emotion_sad', 'emotion_surprise', 'emotion_disgust']] = emotions_df

# Word Cloud
def generate_wordcloud_for_company(df, company_name):
    text = ' '.join(df[df['Company'] == company_name]['Homepage Text'])
    wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(text)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(f'{company_name}_word_cloud.png')


for company in df['Company'].unique():
    generate_wordcloud_for_company(df, company)

# Save DataFrame to CSV file
df.to_csv('analysis_results.csv', index=False)
