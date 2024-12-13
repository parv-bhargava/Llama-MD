import re
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DataProcessor:
    def __init__(self, documents: List[List[Document]], chunk_size: int = 1500, chunk_overlap: int = 150):
        self.documents = documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def clean_text(self, text: str) -> str:
        """Clean text by removing unwanted characters and formatting."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[\d+\]', '', text)
        text = text.replace('\n', ' ')
        text = re.sub(r'[^a-zA-Z0-9.,;:!?()\'\" ]', '',
                      text)
        return text.strip()

    def process_documents(self):
        """Process all documents by cleaning, chunking, and appending metadata."""
        processed_documents = []
        for sublist in self.documents:
            for document in sublist:
                cleaned_text = self.clean_text(document.page_content)
                chunks = self.text_splitter.split_text(cleaned_text)
                for chunk in chunks:
                    processed_document = Document(page_content=chunk, metadata=document.metadata.copy())
                    processed_documents.append(processed_document)
        print(f"Processed {len(processed_documents)} chunks from {len(self.documents)} documents.")
        return processed_documents


# example usage
if __name__ == '__main__':
    documents = [[Document(
        page_content="This is a test document. It contains some text.",
        metadata={"Title": "Test Document"})]]
    data_processor = DataProcessor(documents)
    processed_documents = data_processor.process_documents()
    print(processed_documents)
