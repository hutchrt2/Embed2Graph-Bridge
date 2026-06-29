# ESM-C Migration Status & Background Job Note

**Date**: 2026-06-26
**Previous Conversation ID**: `00a055b6-6fb3-4591-bdca-2570dab2a214`

### Current Status
- **Migration**: Completed transitioning from ESM-2 to ESM-C (`esmc_300m`).
- **Verification**: Code verification and end-to-end testing (querying and joining on `global_node_id` with metadata) succeeded using a smaller subset of reference sequences.
- **Background Execution**: The full database initialization command is running in the background with 20 threads to leverage the server's multi-core CPU:
  ```bash
  OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 .venv/bin/python embed2graph_bridge.py --init
  ```
  *(Started on June 26, 2026. Task ID: `task-204`)*
- **Monitoring Log**: The script writes a timestamped log file (`monitor_20260626_154236.log`) in the project root. It records both the command’s own progress output and periodic CPU/memory snapshots.

### Files & Locations
- **Code**: [embed2graph_bridge.py](file:///local/storage/thomas/4_embed2graph-bridge-test/embed2graph_bridge.py)
- **FAISS Cache & Metadata**: Output is saved in the `blastdb/` directory.
- **Query Results Output**: Output of the query pipeline is serialized to `output/vector_query_results.csv`.
- **System Task Log**: You can check the running process output or logs at:
  `/home/rth72/.gemini/antigravity-ide/brain/00a055b6-6fb3-4591-bdca-2570dab2a214/.system_generated/tasks/task-204.log`

### Next Steps when Returning
1. Verify that the background process finished successfully by checking that `blastdb/faiss_index.bin` and `blastdb/index_metadata.json` have updated timestamps and contain all 6,990 sequences.
2. Run a full query search:
   ```bash
   .venv/bin/python embed2graph_bridge.py --query input_FASTA/query.fasta
   ```
