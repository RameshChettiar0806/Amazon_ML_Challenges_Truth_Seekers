import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import rapidfuzz.process as process
from rapidfuzz.distance import JaroWinkler

# Data directories
def find_data_dir():
    cur = os.getcwd()
    candidates = [
        os.path.abspath(os.path.join(cur, "../../../../dataset")),
        os.path.abspath(os.path.join(cur, "../../../dataset")),
        os.path.abspath(os.path.join(cur, "../../dataset")),
        os.path.abspath(os.path.join(cur, "dataset")),
        r"c:\Users\Ramesh\Personal Folders\Hackathons\Amazon ML\Dataset\6ab10eb3b23ba_student_resource\student_resource\dataset"
    ]
    for d in candidates:
        if os.path.exists(os.path.join(d, "test")):
            return d
    return candidates[-1]

# Vectorized text clean
def clean_for_blocking(text_series: pd.Series) -> pd.Series:
    import unicodedata
    s = text_series.fillna("").astype(str)
    s = s.apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('utf-8').lower())
    s = s.str.replace(r'^(?:https?://)?(?:www\.)?([^/]+?)\.(?:com|org|net|in|fr).*', r'\1', regex=True)
    s = s.str.replace(r'https?://(?:www\.)?', '', regex=True)
    s = s.str.replace('&', ' and ', regex=False)
    s = s.str.replace('.', ' ', regex=False)
    legal = r'\b(pvt|private|ltd|limited|corp|corporation|inc|incorporated|llc|co|company|sarl|sas|sa|eurl|sci|sasu)\b'
    s = s.str.replace(legal, ' ', regex=True)
    s = s.str.replace(r'[^a-z0-9\s]', ' ', regex=True)
    return s.str.replace(r'\s+', ' ', regex=True).str.strip()

def retrieve_top_k_vectorized(X_queries: csr_matrix, X_pool: csr_matrix, pool_ids: np.ndarray, pool_raw_names: np.ndarray, top_k: int = 30, min_sim: float = 0.12, batch_size: int = 2000):
    n_queries = X_queries.shape[0]
    cand_ids_list = []
    cand_names_list = []
    
    n_batches = int(np.ceil(n_queries / batch_size))
    for b in range(n_batches):
        b_start = b * batch_size
        b_end = min((b + 1) * batch_size, n_queries)
        
        sim_batch = X_queries[b_start:b_end].dot(X_pool.T)
        indptr = sim_batch.indptr
        indices = sim_batch.indices
        data = sim_batch.data
        
        for i in range(b_end - b_start):
            start = indptr[i]
            end = indptr[i + 1]
            
            if start == end:
                cand_ids_list.append([])
                cand_names_list.append([])
                continue
                
            row_indices = indices[start:end]
            row_scores = data[start:end]
            
            mask = row_scores >= min_sim
            if not np.any(mask):
                cand_ids_list.append([])
                cand_names_list.append([])
                continue
                
            filt_indices = row_indices[mask]
            filt_scores = row_scores[mask]
            
            if len(filt_scores) > top_k:
                top_sub = np.argpartition(filt_scores, -top_k)[-top_k:]
                sort_ord = np.argsort(filt_scores[top_sub])[::-1]
                selected_idx = filt_indices[top_sub[sort_ord]]
            else:
                sort_ord = np.argsort(filt_scores)[::-1]
                selected_idx = filt_indices[sort_ord]
                
            cand_ids_list.append(pool_ids[selected_idx].tolist())
            cand_names_list.append(pool_raw_names[selected_idx].tolist())
            
    return cand_ids_list, cand_names_list

def main():
    DATA_DIR = find_data_dir()
    OUTPUT_DIR = os.path.abspath(os.path.join(DATA_DIR, "..", "output"))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    cand_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    match_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    
    print(f"Data Dir: {DATA_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    
    s1_path = os.path.join(DATA_DIR, "test", "test_source1.tsv")
    s2_path = os.path.join(DATA_DIR, "test", "test_source2.tsv")
    s3_path = os.path.join(DATA_DIR, "test", "test_source3.tsv")
    
    print("Loading test data...")
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str)
    df_s2 = pd.read_csv(s2_path, sep="\t", dtype=str)
    df_s3 = pd.read_csv(s3_path, sep="\t", dtype=str)
    
    df_pool = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3
    
    print("Normalizing names...")
    df_s1["clean_name"] = clean_for_blocking(df_s1["business_name"])
    df_pool["clean_name"] = clean_for_blocking(df_pool["business_name"])
    
    countries = df_s1["country"].dropna().unique()
    
    print(f"Found countries: {countries}")
    
    total_processed = 0
    t0 = time.time()
    
    with open(cand_path, "w", encoding="utf-8") as fc, open(match_path, "w", encoding="utf-8") as fm:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        
        for country in countries:
            print(f"\n--- Processing country: {country} ---")
            s1_c = df_s1[df_s1["country"] == country].reset_index(drop=True)
            pool_c = df_pool[df_pool["country"] == country].reset_index(drop=True)
            
            if len(pool_c) == 0:
                print(f"No candidates for {country}. Writing empty candidates.")
                for sid in s1_c["entity_id"]:
                    fc.write(f"{sid}\t\n")
                    fm.write(f"{sid}\t\n")
                continue
                
            print(f"  S1 queries: {len(s1_c):,}, Pool size: {len(pool_c):,}")
            
            # --- Fast Path: Exact Match Index ---
            print("  Building Exact Match Index...")
            pool_exact_map = {}
            for idx, cname in enumerate(pool_c["clean_name"]):
                if cname: # skip empty
                    if cname not in pool_exact_map:
                        pool_exact_map[cname] = []
                    pool_exact_map[cname].append(idx)
                    
            # Separate queries into Exact Match vs Fallback
            exact_queries_idx = []
            fallback_queries_idx = []
            
            for idx, qname in enumerate(s1_c["clean_name"]):
                if qname and qname in pool_exact_map:
                    exact_queries_idx.append(idx)
                else:
                    fallback_queries_idx.append(idx)
                    
            print(f"  Queries with Exact Match: {len(exact_queries_idx):,} | Fallback to TF-IDF: {len(fallback_queries_idx):,}")
            
            # Arrays to store candidate results for each query
            final_cand_ids = [[] for _ in range(len(s1_c))]
            final_cand_names = [[] for _ in range(len(s1_c))]
            
            pool_ids = pool_c["entity_id"].values
            pool_raw_names = pool_c["business_name"].fillna("").astype(str).str.lower().values
            
            # 1. Populate Exact Matches
            for idx in exact_queries_idx:
                qname = s1_c["clean_name"].iloc[idx]
                matched_pool_indices = pool_exact_map[qname]
                final_cand_ids[idx] = pool_ids[matched_pool_indices].tolist()
                final_cand_names[idx] = pool_raw_names[matched_pool_indices].tolist()
                
            # 2. Run TF-IDF only for Fallback Queries
            if len(fallback_queries_idx) > 0:
                print("  TF-IDF Vectorization for Fallback...")
                vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=2, max_df=0.01, max_features=50000, dtype=np.float32)
                X_pool = vec.fit_transform(pool_c["clean_name"])
                
                fallback_s1_c = s1_c.iloc[fallback_queries_idx]
                X_s1_fallback = vec.transform(fallback_s1_c["clean_name"])
                
                print("  Sparse Retrieval for Fallback...")
                # PREVENT MEMORY HANG: Use a very small batch size (100) so the resulting dot-product sparse matrix doesn't consume all your RAM and stall!
                fb_cand_ids_list, fb_cand_names_list = retrieve_top_k_vectorized(
                    X_s1_fallback, X_pool, pool_ids, pool_raw_names, top_k=50, min_sim=0.15, batch_size=100
                )
                
                for i, original_idx in enumerate(fallback_queries_idx):
                    final_cand_ids[original_idx] = fb_cand_ids_list[i]
                    final_cand_names[original_idx] = fb_cand_names_list[i]
            
            print("  RapidFuzz Matching...")
            s1_raw_names = s1_c["business_name"].fillna("").astype(str).str.lower().values
            s1_ids = s1_c["entity_id"].values
            
            for i in range(len(s1_c)):
                sid = s1_ids[i]
                c_ids = final_cand_ids[i]
                c_names = final_cand_names[i]
                
                fc.write(f"{sid}\t{','.join(c_ids)}\n")
                
                # Matching
                matched_ids = []
                if c_ids:
                    s1_n = s1_raw_names[i]
                    choices = dict(zip(c_ids, c_names))
                    
                    # Extract up to 2 top matches if their score is >= 0.88
                    results = process.extract(s1_n, choices, scorer=JaroWinkler.similarity, score_cutoff=0.88, limit=2)
                    matched_ids = [res[2] for res in results]
                
                fm.write(f"{sid}\t{','.join(matched_ids)}\n")
                
                total_processed += 1
                if total_processed % 50000 == 0:
                    print(f"  Processed {total_processed:,} queries...")
                    
    print(f"\nPipeline finished in {time.time() - t0:.2f} seconds.")
    print("Outputs generated successfully!")

if __name__ == "__main__":
    main()
