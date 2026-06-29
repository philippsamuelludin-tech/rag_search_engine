import re


def semantic_chunk(text: str, max_chunk_size: int, overlap: int) -> list[str]:
    text = text.strip()
    if text == "":
        return []
    
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1 and sentences[0].endswith(("!", "?", ".")) == False:
        sentences = [text]
    chunks = []
    i = 0
    n_sentences = len(sentences)
    while i < n_sentences:
        chunk_sentences = sentences[i : i + max_chunk_size]
        # stop if this chunk is too small to be useful
        if chunks and len(chunk_sentences) <= overlap:
            break

        cleaned = [s.strip() for s in chunk_sentences if s.strip()]
        if cleaned:
            chunks.append(" ".join(cleaned))
        i += max_chunk_size - overlap
    return chunks