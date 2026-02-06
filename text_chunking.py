from extraction import get_extracted_text


def receive_text(pdf_path='document.pdf', chunk_size=1000, overlap=200):
	"""Return a list of text chunks extracted from `pdf_path`.

	- `chunk_size` is the target number of characters per chunk.
	- `overlap` is the number of characters each chunk should overlap with the previous one.
	"""
	text = get_extracted_text(pdf_path)
	if not text:
		return []
	if overlap >= chunk_size:
		raise ValueError('overlap must be smaller than chunk_size')

	step = chunk_size - overlap
	chunks = []
	for i in range(0, len(text), step):
		chunk = text[i:i + chunk_size]
		chunks.append(chunk)
		if i + chunk_size >= len(text):
			break
	return chunks


if __name__ == '__main__':
	pdf = '/home/amir/Desktop/MRE TSD/5. Risk Improvement Benchmark 1.pdf'  # replace with your PDF path
	chunks = receive_text(pdf, chunk_size=2000, overlap=200)
	print('Chunks:', len(chunks))
	if chunks:
		print(chunks[0][:500])