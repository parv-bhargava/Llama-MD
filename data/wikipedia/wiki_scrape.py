import pandas as pd
import nltk
import string
import re
import numpy as np
import wikipediaapi
from langchain.document_loaders import WikipediaLoader
import warnings
import os
warnings.filterwarnings("ignore")
print(os.getcwd())
#%%
query="pregnancy"
loader = WikipediaLoader(query=query,load_max_docs=50, lang="en")

# Load the document (retrieves content from Wikipedia)
documents = loader.load()
titles = []
for idx,meta in enumerate(documents):
    # print(documents[idx].metadata)
    title = meta.metadata.get('title', 'No Title Found')  # Use .get() to avoid KeyErrors if title is not present
    titles.append(title)

#%%
wiki_wiki = wikipediaapi.Wikipedia(
    user_agent='NLP-Project (raghav.agarwal@gwmail.gwu.edu)',
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
)

def save_wikipedia_pages_to_txt(titles, folder_path="../wiki_texts"):
    os.makedirs(os.path.abspath(folder_path), exist_ok=True)

    for title in titles:
        p_wiki = wiki_wiki.page(title)

        if p_wiki.exists():
            important_text = []
            for section in p_wiki.sections:
                if section.title.lower() not in ["history", "references", "see also",
                                                 "external links"]:
                    important_text.append(section.text)

            if important_text:
                file_path = os.path.join(os.path.abspath(folder_path), f"{title.replace(' ', '_')}.txt")
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write("\n\n".join(important_text))
                print(f"Saved filtered content: {title} -> {file_path}")
            else:
                print(f"No important content found for: {title}")
        else:
            print(f"Page not found: {title}")


save_wikipedia_pages_to_txt(titles)