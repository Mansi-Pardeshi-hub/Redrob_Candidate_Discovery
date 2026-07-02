import os
import json
import pandas as pd
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. DOCUMENT TEXT PARSING LAYER
def extract_text_from_docx(docx_path):
    """Extracts raw text content from Microsoft Word Document (.docx) files."""
    doc = Document(docx_path)
    full_text = [para.text for para in doc.paragraphs]
    return '\n'.join(full_text)

# 2. HIGH-PERFORMANCE HYBRID RANKING ENGINE
def build_ranking_engine(jd_path, jsonl_path, output_xlsx_path):
    print("[1/4] Extracting Job Description text tokens...")
    jd_text = extract_text_from_docx(jd_path)
    
    profile_corpus = []
    meta_records = []
    line_count = 0
    
    print("[2/4] Streaming 100,000 candidate profiles from JSONL dataset...")
    with open(jsonl_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
                line_count += 1
                
                # Dynamic fallback architecture to safely resolve structural key naming variations
                cand_id = candidate.get("candidate_id") or candidate.get("id") or candidate.get("uid") or f"CAND_{line_count}"
                cand_name = candidate.get("name") or candidate.get("candidate_name") or candidate.get("full_name") or f"Applicant_{line_count}"
                
                # Robust flattening logic converts entire schema contents to string tokens
                # This guarantees that raw text is captured regardless of unexpected key variations
                profile_text = " ".join([str(val) for val in candidate.values() if val is not None])
                profile_corpus.append(profile_text)
                
                # Extract and normalize behavioral platform engagement values securely
                activity_score = 0.0
                for act_key in ['platform_activity_score', 'activity_score', 'score']:
                    if act_key in candidate and candidate[act_key] is not None:
                        try:
                            score_val = float(candidate[act_key])
                            activity_score = score_val / 100.0 if score_val > 1.0 else score_val
                            break
                        except:
                            pass
                    
                meta_records.append({
                    "candidate_id": cand_id,
                    "candidate_name": cand_name,
                    "activity_score": round(activity_score, 4)
                })
            except Exception as e:
                continue

    print(f"Total entries verified and loaded in matrix: {len(meta_records)}")
    if not meta_records:
        print("ERROR: Ingestion phase failed! Core data list is empty.")
        return

    print("[3/4] Vectorizing text token spaces using optimized TF-IDF similarities...")
    # Computes high-speed semantic matching metrics across massive data pools safely
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([jd_text] + profile_corpus)
    
    jd_vector = tfidf_matrix[0]
    candidate_vectors = tfidf_matrix[1:]
    similarities = cosine_similarity(jd_vector, candidate_vectors).flatten()
    
    # Building final structural priority list
    ranked_list = []
    for idx, record in enumerate(meta_records):
        semantic_score = float(similarities[idx])
        act_score = record["activity_score"]
        
        # CORE CHALLENGE FORMULA LAYERING: 70% Semantic Fit Weight + 30% Active Metric Weight
        final_score = (0.70 * semantic_score) + (0.30 * act_score)
        
        ranked_list.append({
            "rank": 0,
            "candidate_id": record["candidate_id"],
            "candidate_name": record["candidate_name"],
            "semantic_fit_score": round(semantic_score, 4),
            "activity_score": round(act_score, 4),
            "final_hybrid_score": round(final_score, 4)
        })
        
    print("[4/4] Sorting matrix profiles and generating final XLSX spreadsheet layout...")
    ranked_df = pd.DataFrame(ranked_list)
    ranked_df = ranked_df.sort_values(by="final_hybrid_score", ascending=False).reset_index(drop=True)
    ranked_df['rank'] = ranked_df.index + 1
    
    # Save the output to your project directory
    ranked_df.to_excel(output_xlsx_path, index=False)
    print(f"Success! Final structured prioritized spreadsheet file saved at: {output_xlsx_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
    
    # Dynamic tree explorer automated matching logic
    base_data_dir = None
    top_folder = os.path.join(current_dir, "[PUB] India_runs_data_and_ai_challenge")
    
    if os.path.exists(top_folder):
        for root, dirs, files in os.walk(top_folder):
            if "candidates.jsonl" in files and "job_description.docx" in files:
                base_data_dir = root
                break

    if base_data_dir:
        JD_FILE = os.path.join(base_data_dir, "job_description.docx")
        CANDIDATES_FILE = os.path.join(base_data_dir, "candidates.jsonl")
        OUTPUT_FILE = os.path.join(current_dir, "ranked_candidates_output.xlsx")
        
        build_ranking_engine(JD_FILE, CANDIDATES_FILE, OUTPUT_FILE)
    else:
        print("ERROR: Workspace tracking broken! Confirm competition data files reside inside project directory tree structure.")