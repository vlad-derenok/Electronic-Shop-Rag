# evaluator.py
from fuzzywuzzy import fuzz

def precision_at_k_fuzzy(retrieved_chunks, relevant_chunks, threshold=70):
    k = len(retrieved_chunks)
    relevant_count = 0

    for r_chunk in retrieved_chunks:
        for rel_chunk in relevant_chunks:
            if fuzz.partial_ratio(r_chunk, rel_chunk) >= threshold:
                relevant_count += 1
                break

    return relevant_count / k if k > 0 else 0