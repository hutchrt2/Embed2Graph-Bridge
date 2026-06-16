import argparse
import os
import glob
import torch
import faiss
import numpy as np
import pandas as pd
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel

def get_device(device_override=None):
    if device_override:
        return device_override
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_model(model_name="facebook/esm2_t12_35M_UR50D", device="cpu"):
    print(f"Loading model {model_name} on {device}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Apply float16 optimization on GPU to speed up and save memory
        if device in ("cuda", "mps"):
            model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
        else:
            model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        return tokenizer, model
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}: {e}")

def embed_sequences(sequences, tokenizer, model, device="cpu", batch_size=8):
    embeddings = []
    total_seqs = len(sequences)
    total_batches = (total_seqs + batch_size - 1) // batch_size
    
    for batch_idx, i in enumerate(range(0, total_seqs, batch_size)):
        batch_seqs = sequences[i:i + batch_size]
        inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            
            sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            
            # L2 Normalize for cosine similarity calculation via inner product
            mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
            
            # Convert back to float32 to prevent numeric type mismatch downstream (e.g., in FAISS)
            embeddings.append(mean_pooled.to(torch.float32).cpu().numpy())
            
        if (batch_idx + 1) % 5 == 0 or batch_idx + 1 == total_batches:
            print(f"  Processed batch {batch_idx + 1}/{total_batches} ({min((batch_idx + 1) * batch_size, total_seqs)}/{total_seqs} sequences)...")
            
    if not embeddings:
        return np.array([])
    return np.vstack(embeddings)

def init_database(args):
    print("Phase A: Initializing database...")
    fasta_path = args.db_fasta
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"Reference FASTA not found at {fasta_path}")
        
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        raise ValueError(f"No valid sequences found in reference FASTA at {fasta_path}")
        
    sequences = [str(r.seq) for r in records]
    uniprot_ids = [r.id.split('|')[1] if '|' in r.id else r.id for r in records]
    
    device = get_device(args.device)
    tokenizer, model = load_model(args.model, device=device)
    print(f"Embedding {len(sequences)} reference sequences...")
    embeddings = embed_sequences(sequences, tokenizer, model, device=device, batch_size=args.batch_size)
    
    os.makedirs(args.db_dir, exist_ok=True)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    
    try:
        index.add(embeddings)
        faiss.write_index(index, os.path.join(args.db_dir, "faiss_index.bin"))
        np.save(os.path.join(args.db_dir, "esm2_embeddings.npy"), embeddings)
        with open(os.path.join(args.db_dir, "index_uniprot_ids.txt"), "w") as f:
            for uid in uniprot_ids:
                f.write(f"{uid}\n")
    except Exception as e:
        raise RuntimeError(f"Failed to create or save FAISS index: {e}")
        
    print(f"Initialization complete. Files saved in '{args.db_dir}'.")

def query_database(args):
    print("Phase B: Query Embedding & Vector Search...")
    index_path = os.path.join(args.db_dir, "faiss_index.bin")
    id_path = os.path.join(args.db_dir, "index_uniprot_ids.txt")
    if not os.path.exists(index_path) or not os.path.exists(id_path):
        raise FileNotFoundError(f"Database files not found in '{args.db_dir}'. Run with --init first.")
        
    try:
        index = faiss.read_index(index_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS index: {e}")
        
    with open(id_path, "r") as f:
        uniprot_ids = [line.strip() for line in f]
        
    device = get_device(args.device)
    tokenizer, model = load_model(args.model, device=device)
    
    meta_path = args.metadata
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata CSV not found at {meta_path}")
    metadata_df = pd.read_csv(meta_path)
    if "uniprot_id" not in metadata_df.columns:
        raise ValueError(f"Metadata CSV at {meta_path} is missing the required 'uniprot_id' column.")
    
    query_files = []
    if os.path.isdir(args.query):
        query_files = sorted(glob.glob(os.path.join(args.query, "*.fasta")) + glob.glob(os.path.join(args.query, "*.fa")))
    else:
        if not os.path.exists(args.query):
            raise FileNotFoundError(f"Query path not found: {args.query}")
        query_files = [args.query]
        
    all_results = []
    
    for q_file in query_files:
        print(f"Processing query file: {q_file}")
        try:
            records = list(SeqIO.parse(q_file, "fasta"))
        except Exception as e:
            print(f"Warning: Failed to parse {q_file} as FASTA: {e}. Skipping.")
            continue
            
        if not records:
            print(f"Warning: No sequences found in {q_file}. Skipping.")
            continue
            
        q_sequences = [str(r.seq) for r in records]
        q_ids = [r.id for r in records]
        
        q_embeddings = embed_sequences(q_sequences, tokenizer, model, device=device, batch_size=args.batch_size)
        
        k = min(args.k, index.ntotal)
        distances, indices = index.search(q_embeddings, k)
        
        # Phase C: Blueprint Mimic
        for i, (dists, idxs) in enumerate(zip(distances, indices)):
            q_id = q_ids[i]
            for d, idx in zip(dists, idxs):
                if idx == -1: continue
                t_id = uniprot_ids[idx]
                cosine_sim = d 
                pident = min(100.0, max(0.0, float(cosine_sim) * 100))
                evalue = max(0.0, 1.0 - float(cosine_sim))
                
                all_results.append({
                    "query": q_id,
                    "target": t_id,
                    "pident": pident,
                    "evalue": evalue,
                    "qcov": 1.0,
                    "tcov": 1.0
                })
                
    if not all_results:
        print("No search results found.")
        return
        
    results_df = pd.DataFrame(all_results)
    
    # Phase D: The Relational Handshake (The Join)
    print("Phase D: Joining with knowledge graph metadata...")
    joined_df = pd.merge(results_df, metadata_df, left_on="target", right_on="uniprot_id", how="inner")
    
    if joined_df.empty:
        print("=" * 60)
        print("WARNING: The join with metadata resulted in an empty DataFrame.")
        print("Troubleshooting checks:")
        print(" 1. Ensure reference FASTA headers yield target IDs matching metadata 'uniprot_id'.")
        sample_targets = list(results_df["target"].unique()[:5])
        sample_meta = list(metadata_df["uniprot_id"].unique()[:5])
        print(f"    Sample target IDs extracted: {sample_targets}")
        print(f"    Sample metadata uniprot_ids: {sample_meta}")
        print(" 2. Make sure they match exactly (case-sensitive, no extra whitespace).")
        print("=" * 60)
        
    # Phase E: Output Serialization
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    joined_df.to_csv(args.output, index=False)
    print(f"Phase E: Results saved to {args.output} ({len(joined_df)} entries joined)")

def main():
    parser = argparse.ArgumentParser(description="Embed2Graph Bridge - Protein Sequence Vector Search")
    parser.add_argument("--init", action="store_true", help="Initialize the database by embedding reference sequences")
    parser.add_argument("--query", type=str, help="Path to a query FASTA file or directory containing FASTA files")
    parser.add_argument("--output", type=str, default="output/vector_query_results.csv", help="Path to output CSV")
    parser.add_argument("--model", type=str, default="facebook/esm2_t12_35M_UR50D", help="HuggingFace ESM-2 model name")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors to retrieve")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for model inference")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"], help="Specify compute device")
    
    # Configurable database/metadata input options
    parser.add_argument("--db-fasta", type=str, default="input_database/psfd_sequences.fasta", help="Path to reference FASTA database")
    parser.add_argument("--metadata", type=str, default="input_database/sequence_metadata.csv", help="Path to sequence metadata CSV")
    parser.add_argument("--db-dir", type=str, default="blastdb", help="Directory to save/load vector database index")
    
    args = parser.parse_args()
    
    if args.init:
        init_database(args)
    if args.query:
        query_database(args)
    if not args.init and not args.query:
        parser.print_help()
        
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
