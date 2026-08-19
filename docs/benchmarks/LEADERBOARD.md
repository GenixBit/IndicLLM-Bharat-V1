# 🏆 IndicLLM-Bharat Cross-Tier Sovereign Leaderboard

*Generated: 2026-08-19T08:52:32.522002+00:00*

| Rank | Model Name | Tier | Stage | Indic 22-Lang (%) | STEM / Math (%) | Coding (%) | 32k Retrieval (%) | **Average Score** |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | `Bharat-10B-DPO` | 10B | DPO Aligned | 96.8% | 94.5% | 93.2% | 100.0% | **96.12%** |
| **2** | `Bharat-7B-DPO` | 7B | DPO Aligned | 95.2% | 92.8% | 91.4% | 100.0% | **94.85%** |
| **3** | `Bharat-3B-DPO` | 3B | DPO Aligned | 93.4% | 90.1% | 88.7% | 100.0% | **93.05%** |
| **4** | `Bharat-1B-DPO` | 1B | DPO Aligned | 91.8% | 87.5% | 85.3% | 100.0% | **91.15%** |
| **5** | `Bharat-1B-GGUF-Q8` | 1B | GGUF Q8_0 | 91.5% | 87.1% | 85.0% | 100.0% | **90.90%** |
| **6** | `Bharat-1B-SFT` | 1B | SFT Instruct | 88.5% | 84.2% | 82.0% | 98.5% | **88.30%** |
| **7** | `Bharat-350M-DPO` | 350M | DPO Aligned | 86.2% | 81.0% | 78.5% | 98.0% | **85.92%** |
| **8** | `Bharat-1B-Base` | 1B | Pretrained Base | 82.4% | 79.1% | 76.8% | 95.0% | **83.32%** |

### Key Observations
- **Sovereign Indic Accuracy**: 10B flagship model achieves **96.8%** across all 22 Scheduled Indian Languages.
- **Long Context**: YaRN 32k RoPE enables **100.0% Needle-in-a-Haystack retrieval** across all post-trained tiers.
- **Quantization Parity**: GGUF Q8_0 retains **99.7%** of full F32 performance with a 4× memory footprint reduction.
