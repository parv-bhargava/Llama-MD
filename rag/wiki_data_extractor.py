import os
import pandas as pd
import numpy as np
from nltk.corpus.reader import documents
from tqdm import tqdm
from preprocessor import DataProcessor , Document
from dotenv import load_dotenv
from vector_db import VectorDB
load_dotenv()


path_to_wiki = '../data/wiki_texts'
wiki_texts = os.path.abspath(path_to_wiki)
txt_files = [file for file in os.listdir(wiki_texts)]

processed_docs = []
for txt_file in tqdm(txt_files):
    documents = []
    with open(os.path.abspath(path_to_wiki + os.path.sep + txt_file), 'r') as f:
        text = f.read()
        doc = Document(
        page_content = text,
        metadata={"Title": txt_file[:-4]})
        documents.append([doc])
    data_processor = DataProcessor(documents)
    processed_documents = data_processor.process_documents()
    for doc_pro in processed_documents:
        processed_docs.append(doc_pro)



pinecone_key = os.getenv('PINECONE_API_KEY')
pinecone_env = os.getenv('PINECONE_ENV')
cloud = os.getenv('PINECONE_CLOUD')
index_name = os.getenv('PINECONE_INDEX_NAME')
dimension = int(os.getenv('DIMENSION'))
metric = os.getenv('METRIC')

embedding_manager = VectorDB(
    pinecone_api_key=pinecone_key,
    pinecone_env=pinecone_env,
    cloud=cloud,
    index_name=index_name,
    dimension=dimension,
    metric=metric
)

print(embedding_manager.model , embedding_manager.platform)
embedding_manager.process_and_store_documents(processed_docs)