# Embed2Graph Bridge

Embed2Graph Bridge is a high-performance Python CLI tool designed to link protein sequence databases to relational knowledge graphs using deep learning sequence embeddings. 

By leveraging the state-of-the-art ESM-2 transformer model (`facebook/esm2_t12_35M_UR50D`) and the FAISS vector database library, this tool enables rapid similarity searches on protein sequences and performs a join operation against metadata tables to yield query-annotated datasets.

---

## Key Features

- **Transformer Embeddings**: Employs Hugging Face ESM-2 models for embedding protein sequences into dense vectors.
- **Fast Vector Similarity**: Uses FAISS (`IndexFlatIP`) for low-latency cosine similarity search.
- **Compute Optimization (FP16)**: Automatically loads the model in half-precision (FP16) on CUDA and Apple Silicon MPS devices, halving memory usage and speeding up inference.
- **Relational Handshake (Join)**: Merges the nearest-neighbor search results with a knowledge graph metadata CSV on UniProt IDs, producing an annotated output.
- **Configurable & Flexible**: Supports full customization of batch sizes, compute devices, reference databases, input paths, and query inputs via CLI arguments.
- **Robust Error Checks**: Includes diagnostic warnings for mismatched sequence identifiers between databases and metadata.

---

## Project Structure

```text
embed2graph-bridge/
├── embed2graph_bridge.py   # Main CLI execution script
├── requirements.txt         # Package dependencies list
├── README.md               # Tool documentation
├── .gitignore              # Configured Git tracking instructions
├── input_database/         # Directory for reference sequences & metadata
│   ├── .gitkeep            # Tracked directory placeholder
│   ├── psfd_sequences.fasta (Ignored/Local database sequences)
│   └── sequence_metadata.csv (Ignored/Local sequence metadata)
├── input_FASTA/            # Directory for query sequences
│   ├── .gitkeep            # Tracked directory placeholder
│   └── query.fasta         (Ignored/Local query sequences)
└── blastdb/                # Directory generated for storing the FAISS index (Ignored)
```

---

## Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd embed2graph-bridge
   ```

2. **Set up a Python Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Quick Start Guide

### Step 1: Database Initialization (Phase A)
Process reference sequences from the reference FASTA to build and save the FAISS vector database.
```bash
python3 embed2graph_bridge.py --init
```
*Outputs are saved in the `blastdb/` directory.*

### Step 2: Query Embedding & Vector Search (Phases B to E)
Embed query sequences, find matching neighbors in the database, join with metadata, and serialize to CSV.
```bash
python3 embed2graph_bridge.py --query input_FASTA/query.fasta
```
*Results are saved to `output/vector_query_results.csv`.*

---

## CLI Reference

Run the help menu to see all available command options:
```bash
python3 embed2graph_bridge.py --help
```

### Options Details:
- `--init`: Run database initialization. Parses reference sequences, generates ESM-2 embeddings, and writes the FAISS vector index.
- `--query PATH`: Run query pipeline. Path can be a single FASTA file or a directory containing `.fasta` / `.fa` files.
- `--output PATH`: Destination path for final query results CSV. *(Default: `output/vector_query_results.csv`)*
- `--model NAME`: Hugging Face model identifier for ESM-2. *(Default: `facebook/esm2_t12_35M_UR50D`)*
- `--k INT`: Number of nearest neighbors to retrieve per query sequence. *(Default: `5`)*
- `--batch-size INT`: Batch size for transformer inference. *(Default: `8`)*
- `--device {cuda,mps,cpu}`: Specify a compute device. If not set, the tool auto-detects CUDA (NVIDIA GPU), MPS (Apple Silicon GPU), and CPU in priority order.
- `--db-fasta PATH`: Path to reference database FASTA. *(Default: `input_database/psfd_sequences.fasta`)*
- `--metadata PATH`: Path to reference metadata CSV containing a `uniprot_id` column. *(Default: `input_database/sequence_metadata.csv`)*
- `--db-dir PATH`: Directory where FAISS indices and vector database records are stored. *(Default: `blastdb`)*
