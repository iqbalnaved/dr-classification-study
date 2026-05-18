# Comparative Analysis of General-Purpose vs. Domain-Specific Multimodal Models for Diabetic Retinopathy Classification

[![Paper](https://img.shields.io/badge/Paper-Diagnostics%202026-blue)](https://doi.org/10.3390/diagnostics16101504)
![License](https://img.shields.io/badge/License-CC0-green)

**Mohammad Iqbal Nouyed**, M. Al-Mamun, D. A. Adjeroh, G. Hu

*Diagnostics*, vol. 16, no. 10, p. 1504, May 2026. [doi:10.3390/diagnostics16101504](https://doi.org/10.3390/diagnostics16101504)

---

## Overview

This study systematically benchmarks general-purpose and domain-specific multimodal foundation models on diabetic retinopathy (DR) classification from retinal fundus images, across four evaluation paradigms: zero-shot, few-shot, fine-tuning, and linear probing.

**Key finding:** Domain-specific models (RETFound, EyeCLIP, MedGemma) do not uniformly outperform general-purpose models (GPT, Gemini, ViT-Large) — performance advantages depend strongly on evaluation paradigm and shot count, with important implications for clinical deployment assumptions.

[![Overview](https://github.com/iqbalnaved/dr-classification-study/raw/main/diagram.png "Study overview diagram")](https://github.com/iqbalnaved/dr-classification-study/blob/main/diagram.png)

---

## Models Evaluated

| Model | Type | Paradigm | Script |
|---|---|---|---|
| GPT-5.2 / GPT-4o (OpenAI) | General-purpose | Zero/few-shot | `idrid_gpt.py` |
| Gemini 3 Pro / Flash (Google) | General-purpose | Zero/few-shot | `idrid_gemini_resume.py` |
| Pixtral / Mistral | General-purpose | Zero/few-shot | `idrid_mistral.py` |
| MedGemma 4b-it / 1.5-4b-it | Domain-specific | Zero/few-shot | `medgemma_kshot_idrid.py` |
| RETFound (MAE ViT-L) | Domain-specific | Fine-tune / linear probe | `retfound_finetune_idrid516.py`, `retfound_feat_ex_idrid516.py` |
| EyeCLIP (ViT-L/14) | Domain-specific | Fine-tune / linear probe | `eyeclip_finetune.py`, `eyeclip_feat_ex_idrid.py` |
| MedSigLIP-448 | Domain-specific | Fine-tune / linear probe | `medsiglip_ft.py`, `medsiglip_feat_ex.py` |
| ViT-Large-16 (Google) | General-purpose | Fine-tune / linear probe | `vit_ft.py`, `vit_large16_feat_ex_idrid.py` |

---

## Dataset

**IDRiD** (Indian Diabetic Retinopathy Image Dataset)
- DR: 348 images, No DR: 168 images (516 total)
- Resolution: 4288 × 2848
- Source: [IEEE DataPort](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)

Directory structure expected:
```
<parent-dir>/Data/IDRiD/idrid516_orig/dr_class/
<parent-dir>/Data/IDRiD/idrid516_orig/nm_class/
```

---

## Third-Party Models

- **RETFound**: [openmedlab/RETFound_MAE](https://github.com/openmedlab/RETFound_MAE)
- **EyeCLIP**: [Michi-3000/EyeCLIP](https://github.com/Michi-3000/EyeCLIP)
- **MedGemma**: [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) · [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it)
- **MedSigLIP**: [google/medsiglip-448](https://huggingface.co/google/medsiglip-448)

---

## Setup

```bash
git clone https://github.com/iqbalnaved/dr-classification-study.git
cd dr-classification-study
pip install -r requirements.txt
```

Set API keys:
```bash
export OPENAI_API_KEY='your_key'
export GEMINI_API_KEY='your_key'
export MISTRAL_API_KEY='your_key'
```

---

## Usage

### Zero- and Few-Shot Evaluation

**GPT:**
```bash
python3 idrid_gpt.py -s 0 -r 1 -m gpt-5.2-2025-12-11 -d IDRiD516_orig -k 0
```

**Gemini:**
```bash
python3 idrid_gemini_resume.py --shot 0 --run 1 --model gemini-3-pro-preview --dataset IDRiD516_orig -k 0
```

**Mistral / Pixtral:**
```bash
python3 idrid_mistral.py -s 0 -r 1 -m pixtral-large-2411 -d IDRiD516_orig -k 0
```

**MedGemma (k-shot):**
```bash
# Zero-shot
python medgemma_kshot_idrid.py /path/to/idrid516_orig/ /path/to/outputs/ 1 --k 0

# 5-shot with MedGemma 1.5
python medgemma_kshot_idrid.py /path/to/idrid516_orig/ /path/to/outputs/ 1 --m google/medgemma-1.5-4b-it --k 5
```

---

### Fine-Tuning

**RETFound:**
```bash
python retfound_finetune_idrid516.py \
  --data_path /path/to/idrid516_224x224 \
  --nb_classes 2 --epochs 20 --num_folds 10 \
  --unfreeze_blocks 2 \
  --output_dir /path/to/outputs/retfound_ft
```

**MedSigLIP:**
```bash
python medsiglip_ft.py \
  --data_path /path/to/data \
  --model_name google/medsiglip-448 \
  --num_folds 10 --epochs 20 --batch_size 4 --lr 5e-5 \
  --unfreeze_blocks 2 \
  --output_dir /path/to/outputs/medsiglip_ft \
  --label_dict '{"mm": 1, "bn": 0}'
```

**EyeCLIP:**
```bash
python eyeclip_finetune.py \
  --local_model_path /path/to/EyeCLIP/models/ViT-L-14.pt \
  --data_path /path/to/idrid516_224x224 \
  --epochs 20 --num_folds 10 --unfreeze_blocks 2 \
  --output_dir /path/to/outputs/eyeclip_ft
```

**ViT-Large:**
```bash
python vit_ft.py \
  --data_path /path/to/idrid516_224x224 \
  --model_name google/vit-large-patch16-224 \
  --num_folds 10 --epochs 30 --batch_size 8 --lr 1e-4 \
  --unfreeze_blocks 2 --nb_classes 2 --pooling cls \
  --output_dir /path/to/outputs/vitlarge224_ft
```

---

### Linear Probing

Extract features then run linear probe:
```bash
python retfound_feat_ex_idrid516.py   # RETFound features
python medsiglip_feat_ex.py           # MedSigLIP features
python eyeclip_feat_ex_idrid.py       # EyeCLIP features
python vit_large16_feat_ex_idrid.py   # ViT-Large features
python linprobe.py                     # Train linear probe on extracted features
```

---

## Citation

> Nouyed, M. I., Al-Mamun, M., Adjeroh, D. A., & Hu, G. (2026). Comparative Analysis of General-Purpose vs. Domain-Specific Multimodal Models for Diabetic Retinopathy Classification. *Diagnostics*, 16(10), 1504. https://doi.org/10.3390/diagnostics16101504

---

## Related Work

- Nouyed et al. "Text-Dominant Decision-Making by Large Multimodal Models in Dermatology Clinical Challenges." *JAAD*, under review. [GitHub](https://github.com/iqbalnaved/text-dominant-multimodal-dermatology-eval)
- Nouyed et al. "Sensing but Not Alerting: The Cost of Sycophancy in ChatGPT Psychodermatology Consultations." *JAAD International*, under review. [GitHub](https://github.com/iqbalnaved/sensing-not-alerting)
- Nouyed et al. "The cost of AI sycophancy in dermoscopic diagnosis." *JAAD*, under review. [GitHub](https://github.com/iqbalnaved/prompt-framing-bias-dermoscopy)

---

## License

CC0 1.0 Universal — see [LICENSE](LICENSE).
