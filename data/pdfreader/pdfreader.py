import os
from langchain.document_loaders import PDFPlumberLoader
books_dir = os.path.join(os.path.dirname(__file__), '..','..',  'Books')
output_dir = os.path.join(os.path.dirname(__file__), '..','..',  'Raw_Text')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# loop through files
for filename in os.listdir(books_dir):
    if filename.endswith('.pdf'):
        book_path = os.path.join(books_dir, filename)
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)
        try:
            loader = PDFPlumberLoader(book_path)
            documents = loader.load()
            # Extract text
            full_text = ""
            for i, page in enumerate(documents):
                try:
                    page_text = page.page_content
                    if page_text:
                        full_text += page_text + "\n"
                    else:
                        print(f"Warning: No text extracted from page {i + 1}")
                except Exception as e:
                    print(f"Failed to extract text from page {i + 1}: {e}")

            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(full_text)
            print(f"Successfully saved text from {filename} to {txt_filename}")

        except Exception as e:

            print(f"Failed to process {filename} with PDFPlumberLoader: {e}")
