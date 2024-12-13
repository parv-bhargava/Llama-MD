# Dr. GPT

## Directory Structure
```bash

├── Code
│   ├── data
│   ├── finetune
│   ├── models
│   ├── rag
│   ├── utils
│   ├── .env
│   ├── app.py
│   ├── ReadMe.md
│   ├── requirements.txt

```

## Data Directory

```bash
├── data
│   ├── pdfreader
        ├── pdfreader.py
│   ├── topic_modelling
        ├── topic_modelling.py
│   ├── wikipedia
        ├── wiki_scrape.py
│   ├── Raw_Text
```
### topic_modelling.py
This file reads the dataset from hugging face and returns the most common topics.

### wiki_scrape.py
This file scrapes extra data from wikipedia for our RAG

### pdfreader.py
This file converts pdf books on Pregnancy and Gynecology to raw texts files and saves it in Raw_Text. 

## Finetune Directory

```bash
├── finetune
│   ├── finetune.py
│   ├── inference.py

```

### finetune.py
This file contains our code for finetuning the LLama model

### inference.py
This file contains our code that loads the saved model and tokenizer of the finetuned model performs inference on it.

## Models Directory

This directory contains the pre-trained models used in the project.

## RAG Directory

```bash
├── rag
│   ├── embeddings.py
│   ├── vector_db.py
│   ├── test.py
│   ├── utils.py
│   ├── pdf_data_extractor.py
│   ├── wiki_data_extractor.py
```

### embeddings.py
This file contains the code to generate the embeddings for the documents in the dataset. The embedding can be generated using the following methods:
1. Hugging Face Model
    - WhereIsAI/UAE-Large-V1
    - dmis-lab/biobert-base-cased-v1.1
2. Bedrock API
    - amazon.titan-embed-text-v2:0

Sample code is provided in the file to generate the embeddings using the above methods.

### vector_db.py
This file contains the code to set up and query the vector database. The vector database is used to store the embeddings of the documents in the dataset.
Sample code is provided in the file to set up and query the vector database.

### test.py
This file contains the code to test the RAG.
- The code generates the embeddings for the query.
- Retrieve the context from Pinecone.
- Generate the answer using the RAG.

### utils.py
 This file contains code to get response for the query from bedrock for RAG testing purpose.

### pdf_data_extractor.py
This file converts the raw text from the pdf books to embeddings and stores them in Pinecone.
### wiki_data_extractor.py
This file converts the raw text from wikipedia to embeddings and stores them in Pinecone.

## Utils Directory

```bash 
├── utils
│   ├── preprocess.py

```
Code to preprocess the data for RAG.

## Streamlit App

```bash
├── Code
│   ├── app.py

```
`app.py` is the streamlit app for the project which contains a chatbot interface for the user to interact with the model.

Run Command:
```bash
streamlit run app.py --server.port 8888
```

Note: Please make sure to add `.env` file to connect to our vector database.
