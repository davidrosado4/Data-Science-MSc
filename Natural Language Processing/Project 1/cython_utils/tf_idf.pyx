from math import log
from collections import defaultdict
from numpy import sqrt
from scipy.sparse import csr_array

def compute_tf_idf(docs):
    '''
    Args:
        docs (List[str]): Entire corpus of the dataset, idealy already preprocessed.
    
    Returns:
        List[List[float]]: List of the tf-idf embeddings for each sentence.
    '''
    # Compute the term frequencies for each document
    term_freqs = [defaultdict(int) for _ in range(len(docs))]
    for i, doc in enumerate(docs):
        for word in doc.split():
            term_freqs[i][word] += 1

    # Compute the document frequencies for each term
    doc_freqs = defaultdict(int)
    for term_freq in term_freqs:
        for term in term_freq.keys():
            doc_freqs[term] += 1

    # Compute the tf-idf scores for each term in each document    
    tf_idfs = []
    num_docs = len(docs)
    for i, term_freq in enumerate(term_freqs):
        tf_idf_scores = defaultdict(int)
        norm = 0.0
        for term, freq in term_freq.items():
            tf_idf = tf_idf_c(freq, doc_freqs[term], num_docs)
            tf_idf_scores[term] = tf_idf
            norm = norm + tf_idf**2
        norm = sqrt(norm)
        #norm = np_norm(list(tf_idf_scores.values()))
        tf_idf_vec = [tf_idf_scores[term]/norm for term in doc_freqs.keys()]
        tf_idfs.append(csr_array(tf_idf_vec))
    
    return tf_idfs, list(doc_freqs.keys())

cdef double tf_idf_c(int term_freq, int doc_freq, int num_docs):
    cdef double tf = term_freq
    cdef double idf = log(num_docs / doc_freq)
    return tf * idf



