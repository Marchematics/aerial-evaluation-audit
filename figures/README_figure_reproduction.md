# Composite-figure reproduction

Run `python scripts/build_single_column_composite_figures.py` from the project root.

Outputs are 88-mm-wide vector PDF/SVG and 600-dpi PNG previews.  Figure A contains only empirical source/coverage panels (a)--(f); the requested workflow/architecture panel is intentionally omitted. Figure B uses a common 0--0.52 `F1 policy band` range because the released grid maximum is 0.512; a 0--0.35 range would clip material values.

`figure_source_data/Fig_A_source_data.csv`, `Fig_B_heatmap_source_data.csv`, and `Fig_B_bootstrap_source_data.csv` are the records used to draw the composite panels.  Bootstrap intervals are read from the setting-level released files, not reconstructed from manuscript prose.
