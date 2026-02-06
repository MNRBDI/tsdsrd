# @title Default title text
from google.colab import files
from PyPDF2 import PdfReader
import re

uploaded = files.upload()
pdf_path = list(uploaded.keys())[0]

reader = PdfReader(pdf_path)
pdf_text = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        pdf_text += text + "\n"

def chunk_by_headers(text, pattern=r'\n?(?=\d+\.\d+\.\s)'):
    """
    Split the text into chunks based on section headers like '1.0 ', '2.0 ', '3.0 ', '3.1. ', '3.2. ', etc.
    Default pattern: matches headers like '1.0 ', '2.0 ', '3.0 ', '3.1. ', '3.2. ', etc.
    """
    # Optional cleanup: ensure consistent spacing
    text = re.sub(r'\n+', '\n', text).strip()

    # Split using regex with lookahead
    chunks = re.split(pattern, text)

    # Filter out empty or whitespace-only chunks
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    return chunks

chunks = chunk_by_headers(pdf_text)

for i, chunk in enumerate(chunks, 1):
    print(f"\n🔹 Chunk {i}:\n{chunk}")

print(chunks)