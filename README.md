# Raven
Fake news Detection
## Installation

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running with `qwen2.5:3b` pulled
- **Windows only**: [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) 
  required for IndicTransToolkit

### Install dependencies
pip install -r requirements.txt

### Download translation model
The Telugu→English translation model (~400-800MB) downloads automatically
from Hugging Face on first use. To pre-download it explicitly:

python -c "
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
AutoTokenizer.from_pretrained('ai4bharat/indictrans2-indic-en-dist-200M', trust_remote_code=True)
AutoModelForSeq2SeqLM.from_pretrained('ai4bharat/indictrans2-indic-en-dist-200M', trust_remote_code=True)
print('Model downloaded and cached.')
"

This only needs to run once. After that, the model loads from local cache.

### Optional SearXNG discovery

SearXNG discovery is disabled by default and is not required to run Raven. To
enable an existing SearXNG service explicitly, set `SEARXNG_BASE_URL` in the
process environment or local `.env` file. Raven makes no localhost SearXNG
request when the setting is absent or empty. Search results only identify
candidate URLs; fetched source pages must pass Raven's evidence quality,
relevance, and fusion checks before verification. Without SearXNG, Raven uses
the configured news/fact-check providers and returns Unverified when evidence
is insufficient.
