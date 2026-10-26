# Batch integrity summary

Processed 15 files: 12 analysed, 3 flagged/skipped. Runtime 30.0s.

| Document | Status | Risk | Peer % | External % | AI % | Top peer source |
|---|---|---|---|---|---|---|
| [pitch_10_sanjay.mp4](pitch_10_sanjay.mp4.md) | ok | HIGH | 81.2 | 0.0 | n/a | sub_01_aarav.pdf |
| [sub_01_aarav.pdf](sub_01_aarav.pdf.md) | ok | HIGH | 44.3 | 0.0 | 0 | sub_02_meera.pdf |
| [sub_02_meera.pdf](sub_02_meera.pdf.md) | ok | HIGH | 45.2 | 0.0 | 1 | sub_01_aarav.pdf |
| [sub_03_kabir.docx](sub_03_kabir.docx.md) | ok | HIGH | 89.4 | 0.0 | 0 | sub_04_diya.docx |
| [sub_04_diya.docx](sub_04_diya.docx.md) | ok | HIGH | 71.1 | 0.0 | 0 | sub_03_kabir.docx |
| [sub_06_ananya.pdf](sub_06_ananya.pdf.md) | ok | HIGH | 0.0 | 50.8 | 0 | - |
| [sub_07_ishaan.docx](sub_07_ishaan.docx.md) | ok | MEDIUM | 0.0 | 0.0 | 100 | - |
| [sub_15_tanvi.pdf](sub_15_tanvi.pdf.md) | ok | MEDIUM | 0.0 | 0.0 | 86 | - |
| [pitch_11_leela.mp4](pitch_11_leela.mp4.md) | ok | LOW | 0.0 | 0.0 | n/a | - |
| [sub_05_rohan.docx](sub_05_rohan.docx.md) | ok | LOW | 0.0 | 0.0 | 0 | - |
| [sub_08_zara.pdf](sub_08_zara.pdf.md) | ok | LOW | 0.0 | 0.0 | 1 | - |
| [sub_09_vikram.docx](sub_09_vikram.docx.md) | ok | LOW | 0.0 | 0.0 | 0 | - |
| [corrupt_14.docx](corrupt_14.docx.md) | flagged | UNASSESSED | 0.0 | 0.0 | n/a | - |
| [empty_13.pdf](empty_13.pdf.md) | flagged | UNASSESSED | 0.0 | 0.0 | n/a | - |
| [silent_12.mp4](silent_12.mp4.md) | flagged | UNASSESSED | 0.0 | 0.0 | n/a | - |

## Flagged / skipped files

- `corrupt_14.docx`: DOCX is corrupted or not an Office Open XML file
- `empty_13.pdf`: File is empty (0 bytes)
- `silent_12.mp4`: Video has no audio track; nothing to transcribe

## Peer similarity matrix

Cell = % of the **row** document's scored text covered by passages matched in the **column** document.

| # | Document |
|---|---|
| D1 | pitch_10_sanjay.mp4 |
| D2 | pitch_11_leela.mp4 |
| D3 | sub_01_aarav.pdf |
| D4 | sub_02_meera.pdf |
| D5 | sub_03_kabir.docx |
| D6 | sub_04_diya.docx |
| D7 | sub_05_rohan.docx |
| D8 | sub_06_ananya.pdf |
| D9 | sub_07_ishaan.docx |
| D10 | sub_08_zara.pdf |
| D11 | sub_09_vikram.docx |
| D12 | sub_15_tanvi.pdf |

| row \ col | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 | D11 | D12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D1 | - | . | **81** | . | . | . | . | . | . | . | . | . |
| D2 | . | - | . | . | . | . | . | . | . | . | . | . |
| D3 | **18** | . | - | **27** | . | . | . | . | . | . | . | . |
| D4 | . | . | **45** | - | . | . | . | . | . | . | . | . |
| D5 | . | . | . | . | - | **89** | . | . | . | . | . | . |
| D6 | . | . | . | . | **71** | - | . | . | . | . | . | . |
| D7 | . | . | . | . | . | . | - | . | . | . | . | . |
| D8 | . | . | . | . | . | . | . | - | . | . | . | . |
| D9 | . | . | . | . | . | . | . | . | - | . | . | . |
| D10 | . | . | . | . | . | . | . | . | . | - | . | . |
| D11 | . | . | . | . | . | . | . | . | . | . | - | . |
| D12 | . | . | . | . | . | . | . | . | . | . | . | - |
