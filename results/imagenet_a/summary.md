# imagenet_a: run summary

Accuracy is top-1 on every image in the run. Published numbers are ViT-B/16 results from the papers. Δ is shown only for runs that reproduce a published setting exactly. ≈ Total is images × median time per image, so it leaves out model loading.

| Method | File | Images | Acc % | Published % | Δ | Median ms/img | ≈ Total | Notes | Saved | Commit |
|---|---|---|---|---|---|---|---|---|---|---|
| CLIP, 1 prompt | `clip_photo` | 7500 | 47.8 |  |  |  |  | 1 prompt | 2026-10-28 22:54:31 | dffbdbc |
| CLIP, TDA's prompts | `clip` | 7500 | 50.67 |  |  |  |  | 7 prompts | 2026-10-28 23:04:31 | dffbdbc |
| CLIP, 80 prompts | `clip_ens80` | 7500 | 49.95 | 49.89 | +0.06 |  |  | 80 prompts | 2026-10-28 22:49:50 | dffbdbc |
| TDA, 1 view | `tda_1view` | 7500 | 51.05 |  |  | 29.5 | 4 min | 1 view | 2026-10-28 23:57:05 | 6a9fa24 |
| TDA, 64 views (official) | `tda` | 7500 | 60.11 | 60.11 | +0.00 | 641.5 | 80 min | 64 views | 2026-10-29 01:19:25 | 6a9fa24 |
| ZERO | `zero` | 7500 | 59.24 |  |  | 450.9 | 56 min | fp16 votes 59.35; ties 168 | 2026-10-29 02:19:06 | 6a9fa24 |
| TPT | `tpt` | 7500 | 54.59 | 54.77 | -0.18 | 1322.8 | 2.8 h |  | 2026-10-29 05:06:32 | 6a9fa24 |
