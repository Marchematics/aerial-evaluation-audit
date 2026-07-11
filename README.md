# Anonymous review bundle

This data-free bundle contains the frozen rule contract, ontology, raw-cache manifest and SHA-256 values, coverage audits, corrected derived records, and evaluator scripts. Raw images and prediction Parquet files are not redistributed: they are licensed external artifacts and their paths/hashes are recorded in `outputs/candidates/prediction_manifest_v2.csv`.

Reproduction order: run the coverage audit; evaluate AP and the corrected absolute/normalized grids; build corrected artifacts; then evaluate the Hungarian, removed-as-ignore, and paired-bootstrap controls. `SHA256SUMS` verifies every bundled file.

The submission workflow must publish this directory to an anonymous archival location and insert its URL into the manuscript. No URL is fabricated by this repository.
