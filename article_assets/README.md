# IEEE Article Assets

This folder contains generated tables, SVG figures, and a short result report for
the CMAPSS TCN-BiGRU + XGBoost near-failure detection article.

Regenerate everything with:

```bash
python article_assets/generate_ieee_assets.py
```

Outputs:

- `ieee_results_report.md`: concise result narrative, comparison notes, and figure captions.
- `ieee_tables_all.tex`: all LaTeX table snippets in one file.
- `index.html`: quick browser preview of generated figures and table links.
- `tables/`: each table as `.csv`, `.md`, and `.tex`.
- `figures/`: standalone SVG figures suitable for conversion to PDF/PNG.

The model result numbers are taken from existing executed notebook outputs. The
dataset counts are computed directly from the local `data/` files.
