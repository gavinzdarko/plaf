# PLAF — Privacy Leakage Audit Framework

A security auditing tool that detects training data memorization in ML models and LLMs. Point it at any model, and it tells you how much private data an attacker can extract through API queries alone.

Built at the **Intelligence at the Frontier Hackathon** (Mar 14–15, 2026) — AI Safety & Evaluation track.

## What It Does

### Classification Audit
- **Membership Inference**: Determines if specific records were used in training by analyzing model confidence patterns
- **Attribute Reconstruction**: Reconstructs the distribution of sensitive features (diagnosis, insurance type, etc.) in the training data by sweeping input attributes
- **Defense Comparison**: Tests 5 defense strategies (DP-SGD, output noise, confidence rounding, top-K, temperature scaling) and shows the privacy/utility tradeoff
- **Leakage Score**: Unified 0–100 risk score with random-baseline calibration

### LLM Audit
- **GPT-2 Perplexity Analysis**: Local membership inference using token-level perplexity — known training text gets lower perplexity than novel text (AUC ~0.94)
- **OpenAI Logprob Analysis**: Real per-token log-probabilities from the production API to detect memorized text (AUC ~0.81 on gpt-4o-mini)
- **Gemini Continuation Matching**: Tests Google's Gemini API by measuring how accurately it reproduces known training text (AUC ~0.64)
- **Active Extraction Attack**: 4 prompt strategies (direct recall, prefix completion, persona manipulation, template extraction) with refusal detection, specificity scoring, and cross-query consistency analysis

## Results

| Model | AUC | Method |
|-------|-----|--------|
| GPT-2 (local, no defenses) | 0.94 | Perplexity |
| OpenAI gpt-4o-mini (production) | 0.81 | Logprobs |
| Gemini 2.5-flash (production) | 0.64 | Continuation |

Every model tested leaks its training data — the question is how much.

## Quick Start

```bash
# Clone and install
git clone https://github.com/gavinzdarko/plaf.git
cd plaf
pip install -r requirements.txt

# Generate data and train models
python3 mock/generate_data.py
python3 mock/train_model.py
python3 mock/train_model_dp.py

# Download GPT-2 and generate LLM test data
python3 llm/download_gpt2.py
python3 llm/test_data.py

# (Optional) Add API keys for live testing
cp .env.example .env
# Edit .env with your GEMINI_API_KEY and OPENAI_API_KEY

# Launch dashboard
streamlit run app.py
```

## Project Structure

```
plaf/
├── app.py                          # Streamlit dashboard (4 tabs)
├── config.py                       # Attack/model configuration
├── core/
│   ├── target.py                   # Model wrapper (in-process + API)
│   ├── membership_probe.py         # Membership inference (Salem et al. 2019)
│   ├── attribute_reconstructor.py  # Attribute inference via input sweeping
│   ├── leakage_scorer.py           # 0-100 leakage score + mitigations
│   └── defenses.py                 # DP noise, rounding, top-K, temperature
├── llm/
│   ├── llm_probe.py                # GPT-2 perplexity-based probe
│   ├── openai_probe.py             # OpenAI API logprob probe
│   ├── gemini_probe.py             # Gemini API continuation probe
│   └── extraction_probe.py         # Active prompt extraction attacks
├── mock/
│   ├── generate_data.py            # 5K synthetic healthcare records
│   ├── train_model.py              # Overfit + regularized + random models
│   └── train_model_dp.py           # DP-SGD model via Opacus
└── validation/
    └── random_baseline.py          # Attack methodology validation
```

## Academic Grounding

- Salem et al. 2019 — "ML-Leaks" (relaxed membership inference, no shadow models)
- Fredrikson et al. 2015 — Model inversion via confidence scores
- Carlini et al. — LiRA, training data extraction from language models
- Yeom et al. 2018 — Overfitting ↔ privacy leakage connection

## Tech Stack

Python, PyTorch, Streamlit, Plotly, Opacus, HuggingFace Transformers, OpenAI API, Google Gemini API

## License

MIT
