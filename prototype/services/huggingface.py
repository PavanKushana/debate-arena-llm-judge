from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load sentence transformer model
_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading all-mpnet-base-v2 model...")
        _model = SentenceTransformer("all-mpnet-base-v2")
    return _model

def compute_tfidf_similarity(argument: str, references: list) -> list:
    """Compute TF-IDF cosine similarity between argument and all references"""
    if not references:
        return []

    ref_texts = [ref.get("argument", "") for ref in references]
    all_texts = [argument] + ref_texts

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1
        )
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        return similarities.tolist()
    except Exception as e:
        print(f"TF-IDF error: {e}")
        return [0.0] * len(references)

def compute_mpnet_similarity(argument: str, reference: str) -> float:
    """Compute semantic similarity using all-mpnet-base-v2"""
    try:
        model = get_model()
        embeddings = model.encode([argument, reference], convert_to_tensor=True)
        score = util.cos_sim(embeddings[0], embeddings[1]).item()
        return round(float(score), 2)
    except Exception as e:
        print(f"MPNet scoring error: {e}")
        return 0.0

def compute_combined_score(
    tfidf_score: float,
    mpnet_score: float,
    tfidf_weight: float = 0.3,
    mpnet_weight: float = 0.7
) -> float:
    """Combine TF-IDF and MPNet scores with weights"""
    return round(
        tfidf_weight * tfidf_score + mpnet_weight * mpnet_score,
        2
    )

def score_arguments_hf(
    extracted_args: list,
    references: list,
    match_threshold: float = 0.4,
    quality_threshold: float = 0.3
) -> dict:
    """
    Score all arguments using TF-IDF + all-mpnet-base-v2.

    Thresholds:
    - match_threshold: minimum score to count as matched (coverage)
    - quality_threshold: minimum score to include in results
    """
    if not extracted_args or not references:
        return {
            "extracted_arguments": [],
            "coverage": 0.0,
            "quality": 0.0,
            "final_score": 0.0,
            "scoring_details": {
                "model": "TF-IDF + all-mpnet-base-v2",
                "match_threshold": match_threshold,
                "quality_threshold": quality_threshold,
                "tfidf_weight": 0.3,
                "mpnet_weight": 0.7
            }
        }

    matched = 0
    total_quality = 0.0
    results = []

    # Pre-encode all references at once for efficiency
    model = get_model()
    ref_texts = [ref.get("argument", "") for ref in references]
    ref_embeddings = model.encode(ref_texts, convert_to_tensor=True)

    for arg in extracted_args:
        claim = arg.get("argument", "")
        if not claim.strip():
            continue

        # Step 1 — TF-IDF similarity
        tfidf_scores = compute_tfidf_similarity(claim, references)

        # Step 2 — MPNet similarity (encode once, compare all)
        arg_embedding = model.encode(claim, convert_to_tensor=True)
        mpnet_scores = util.cos_sim(arg_embedding, ref_embeddings)[0].tolist()

        # Step 3 — Combine scores and find best match
        best_match = None
        best_combined = 0.0
        best_tfidf = 0.0
        best_mpnet = 0.0

        for ref_idx, ref in enumerate(references):
            tfidf_score = tfidf_scores[ref_idx] if ref_idx < len(tfidf_scores) else 0.0
            mpnet_score = float(mpnet_scores[ref_idx]) if ref_idx < len(mpnet_scores) else 0.0
            combined = compute_combined_score(tfidf_score, mpnet_score)

            if combined > best_combined:
                best_combined = combined
                best_match = ref.get("argument", "")
                best_tfidf = round(tfidf_score, 2)
                best_mpnet = round(mpnet_score, 2)

        # Step 4 — Apply quality threshold
        if best_combined < quality_threshold:
            results.append({
                "argument": claim,
                "best_match": best_match,
                "quality_score": best_combined,
                "tfidf_score": best_tfidf,
                "mpnet_score": best_mpnet,
                "quality_flag": "low"
            })
            continue

        # Step 5 — Apply match threshold for coverage
        if best_combined >= match_threshold:
            matched += 1

        total_quality += best_combined
        results.append({
            "argument": claim,
            "best_match": best_match,
            "quality_score": best_combined,
            "tfidf_score": best_tfidf,
            "mpnet_score": best_mpnet,
            "quality_flag": "high" if best_combined >= match_threshold else "medium"
        })

    # Step 6 — Final scores
    quality_args = [r for r in results if r.get("quality_flag") != "low"]
    total_args = len(extracted_args)

    coverage = round(matched / total_args, 2) if total_args > 0 else 0.0
    quality = round(
        sum(r["quality_score"] for r in quality_args) / len(quality_args), 2
    ) if quality_args else 0.0
    final_score = round(coverage * quality, 2)

    return {
        "extracted_arguments": results,
        "coverage": coverage,
        "quality": quality,
        "final_score": final_score,
        "scoring_details": {
            "model": "TF-IDF + all-mpnet-base-v2",
            "match_threshold": match_threshold,
            "quality_threshold": quality_threshold,
            "tfidf_weight": 0.3,
            "mpnet_weight": 0.7,
            "total_args": total_args,
            "matched_args": matched,
            "low_quality_args": len([r for r in results if r.get("quality_flag") == "low"])
        }
    }