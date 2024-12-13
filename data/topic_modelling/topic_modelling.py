#%%
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from matplotlib import pyplot as plt
import warnings
from datasets import load_dataset
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import warnings
from nltk import FreqDist
import numpy as np
warnings.filterwarnings("ignore")
#%%
def read_medical_chatbot_dataset():
    # Read
    dataset = load_dataset("ruslanmv/ai-medical-chatbot")
    df = pd.DataFrame(dataset['train'])
    return df

df = read_medical_chatbot_dataset()

columns = ['Description', 'Doctor', 'Patient']

#%%
def clean_and_plot_frequency_distribution(text_column , col_name = ""):
    text = " ".join(text_column.astype(str).tolist())
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = cleaned_text.split()
    freq_dist = FreqDist(words)
    plt.figure(figsize=(12, 6))
    freq_dist.plot(200, cumulative=False , title=f'Frequency Distribution of {col_name}')
    plt.show()

    return freq_dist

freq_dist_desc = clean_and_plot_frequency_distribution(df['Description'] , col_name='Description')
freq_dist_doctor = clean_and_plot_frequency_distribution(df['Doctor'] , col_name='Doctor')
freq_dist_patient = clean_and_plot_frequency_distribution(df['Patient'] , col_name='Patient')

#%%
thresh_desc = 20000
thresh_doctor = 100000
thresh_patient = 100000

def filter_words_by_frequency(freq_dist, threshold):
    word_freq_array = np.array(list(freq_dist.items()), dtype=object)
    filtered_words = word_freq_array[np.array(word_freq_array[:, 1], dtype=int) >= threshold, 0]
    return list(filtered_words)

stopwords_desc = filter_words_by_frequency(freq_dist_desc , thresh_desc)
stopwords_doc = filter_words_by_frequency(freq_dist_doctor , thresh_doctor)
stopwords_patient = filter_words_by_frequency(freq_dist_patient , thresh_patient)

custom_stopwords = list(set(stopwords_desc + stopwords_doc + stopwords_patient))

stop_words = list(set(custom_stopwords + stopwords.words('english')))

stop_words_pattern = r'\b(' + '|'.join(stop_words) + r')\b'
lemmatizer = WordNetLemmatizer()


#%%
def preprocess_text(df):
    df = df.drop_duplicates()
    def clean_text(desc):

        desc = desc.lower()

        desc = re.sub(r'^q\. ', '', desc)

        desc = re.sub(r'\?$', '', desc)

        desc = desc.strip()

        desc = re.sub(r'[^a-z0-9\s.,!?]', '', desc)

        desc = re.sub(r'\s+', ' ', desc)

        desc = re.sub(r'\d+', '', desc)

        desc = re.sub(r'<.*?>', '', desc)

        desc = re.sub(r'http\S+|www\S+', '', desc)

        desc = re.sub(stop_words_pattern, '', desc)

        desc = re.sub(r'\s+', ' ', desc).strip()
        # Lemmatize the words
        desc = ' '.join([lemmatizer.lemmatize(word) for word in desc.split()])
        return desc

    df['Description'] = df['Description'].apply(clean_text)
    df['Doctor'] = df['Doctor'].apply(clean_text)
    df['Patient'] = df['Patient'].apply(clean_text)
    return df

df = preprocess_text(df)

#%%

topics_df = pd.DataFrame(index=['Description', 'Doctor', 'Patient'],
                         columns=[f'Topic {i + 1}' for i in range(10)])

def lsa_topic_modelling(descriptions,column_name):
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions)
    lsa = TruncatedSVD(n_components=5, n_iter=100, random_state=42)
    lsa_topics = lsa.fit_transform(tfidf_matrix)
    terms = tfidf_vectorizer.get_feature_names_out()
    print("\nLSA Topics: ")

    for i, comp in enumerate(lsa.components_):
        termsInComp = zip(terms, comp)
        sortedterms = sorted(termsInComp, key=lambda x: x[1], reverse=True)[:10]
        print("Topics %d:" % i)

        for term in sortedterms:
            print(term[0])
        topic_terms = [term[0] for term in sortedterms]
        importance = [term[1] for term in sortedterms]
        topics_df.loc[column_name, f'Topic {i + 1}'] = topic_terms
        plt.figure()
        plt.barh(topic_terms, importance)
        plt.xlabel("Importance")
        plt.title(f"LSA Topic {i + 1} for '{column_name}'")
        plt.gca().invert_yaxis()
        plt.show()
        print(" ")

# Perform LSA
for column_name in ['Description', 'Doctor', 'Patient']:
    lsa_topic_modelling(df[column_name].tolist(), column_name)

topics_df.to_excel('lsa_topics.xlsx', index=True)
print(topics_df)