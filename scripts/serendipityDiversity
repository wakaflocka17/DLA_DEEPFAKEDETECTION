import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def serendipity(recommended_items, relevant_items, item_embeddings):
    """
    Computes the Serendipity of recommendations.

    :param recommended_items: List of lists containing recommended items for each user
    :param relevant_items: List of lists containing previously liked items for each user
    :param item_embeddings: Dictionary {item_id: embedding_vector}
    :return: Serendipity score (value between 0 and 1)
    """
    serendipity_scores = []

    for rec_list, rel_list in zip(recommended_items, relevant_items):
        rel_vectors = [item_embeddings[item] for item in rel_list if item in item_embeddings]
        rec_vectors = [item_embeddings[item] for item in rec_list if item in item_embeddings]

        if rel_vectors and rec_vectors:
            sim_matrix = cosine_similarity(rec_vectors, rel_vectors)
            avg_similarity = np.mean(sim_matrix)  # How similar recommendations are to previously liked items
            serendipity_scores.append(1 - avg_similarity)  # Lower similarity → Higher serendipity

    return np.mean(serendipity_scores) if serendipity_scores else 0.0

def diversity(recommended_items, item_embeddings):
    """
    Computes the Diversity of recommendations.

    :param recommended_items: List of lists containing recommended items for each user
    :param item_embeddings: Dictionary {item_id: embedding_vector}
    :return: Diversity score (value between 0 and 1)
    """
    diversity_scores = []

    for rec_list in recommended_items:
        rec_vectors = [item_embeddings[item] for item in rec_list if item in item_embeddings]

        if len(rec_vectors) > 1:
            sim_matrix = cosine_similarity(rec_vectors)
            mean_similarity = np.mean(sim_matrix[np.triu_indices(len(rec_vectors), k=1)])
            diversity_scores.append(1 - mean_similarity)  # Lower similarity → Higher diversity

    return np.mean(diversity_scores) if diversity_scores else 0.0
