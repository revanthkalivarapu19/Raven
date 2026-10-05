"""
translate.py
============
RAVEN Project - Local Sentence-Aware Translation Module

Purpose:
    Translates Telugu and mixed Telugu-English text into English using:
    ai4bharat/indictrans2-indic-en-dist-200M

Improvements:
    - Lazy model loading (avoids heavy CPU/GPU memory footprint when not in use)
    - Sentence/chunk-level segmentation (prevents quality degradation on long paragraphs)
    - Skips already-English sentences (prevents unnecessary model inference and distortion)
    - URL and entity value protection
    - Fallback protection: preserves original text safely if translation fails
"""

import logging
import re
import sys
from typing import Dict, Any, List, Tuple
import torch

logger = logging.getLogger(__name__)

# Transformers compatibility shim for IndicTransToolkit and IndicTrans2 configuration
try:
    import transformers.tokenization_utils_base
    import transformers.tokenization_utils
    if not hasattr(transformers.tokenization_utils, 'PreTrainedTokenizerBase'):
        transformers.tokenization_utils.PreTrainedTokenizerBase = transformers.tokenization_utils_base.PreTrainedTokenizerBase
except Exception:
    pass

try:
    import transformers.onnx
except Exception:
    import types
    onnx_mod = types.ModuleType("transformers.onnx")
    onnx_mod.__path__ = []
    class OnnxConfig:
        pass
    class OnnxSeq2SeqConfigWithPast:
        pass
    onnx_mod.OnnxConfig = OnnxConfig
    onnx_mod.OnnxSeq2SeqConfigWithPast = OnnxSeq2SeqConfigWithPast
    sys.modules["transformers.onnx"] = onnx_mod
    utils_mod = types.ModuleType("transformers.onnx.utils")
    utils_mod.compute_effective_axis_dimension = lambda *args, **kwargs: 1
    sys.modules["transformers.onnx.utils"] = utils_mod

MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"
SRC_LANG = "tel_Telu"
TGT_LANG = "eng_Latn"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RE_TELUGU = re.compile(r'[\u0C00-\u0C7F]')
RE_URL = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)

# Lazy singletons
_TOKENIZER = None
_MODEL = None
_INDIC_PROCESSOR = None


def get_translation_engine():
    """
    Lazily load and cache the IndicTrans2 model, tokenizer, and processor.
    """
    global _TOKENIZER, _MODEL, _INDIC_PROCESSOR
    if _MODEL is None:
        logger.info(f"Loading IndicTrans2 model on {DEVICE}...")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit.processor import IndicProcessor

        _TOKENIZER = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        _MODEL = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        ).to(DEVICE)
        _INDIC_PROCESSOR = IndicProcessor(inference=True)
        logger.info("IndicTrans2 engine loaded successfully.")

    return _TOKENIZER, _MODEL, _INDIC_PROCESSOR


def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentence-level chunks, respecting English punctuation,
    Telugu danda (।), and newlines.
    """
    if not text:
        return []

    # Replace newlines with sentence boundary markers
    chunks = re.split(r'([.?!।\n]+(?:\s+|$))', text)
    sentences = []
    current = ""

    for chunk in chunks:
        current += chunk
        if any(p in chunk for p in [".", "?", "!", "।", "\n"]):
            s = current.strip()
            if s and any(c.isalnum() for c in s):
                sentences.append(s)
            current = ""

    if current.strip() and any(c.isalnum() for c in current.strip()):
        sentences.append(current.strip())

    return sentences if sentences else [text.strip()]


def _protect_urls(sentence: str) -> Tuple[str, Dict[str, str]]:
    """
    Replaces URLs with temporary placeholder tokens to prevent mistranslation.
    """
    placeholders = {}
    matches = RE_URL.findall(sentence)
    protected = sentence

    for idx, url in enumerate(matches):
        key = f"__PROTECTED_URL_{idx}__"
        placeholders[key] = url
        protected = protected.replace(url, key, 1)

    return protected, placeholders


def _restore_urls(sentence: str, placeholders: Dict[str, str]) -> str:
    """
    Restores original URLs from placeholders.
    """
    restored = sentence
    for key, url in placeholders.items():
        restored = restored.replace(key, url)
    return restored


def translate_sentence(sentence: str) -> str:
    """
    Translates a single sentence from Telugu to English using IndicTrans2.
    Preserves protected values (URLs) and falls back safely on error.
    """
    if not sentence or not sentence.strip():
        return sentence

    # If sentence does not contain Telugu script, skip translation
    if not RE_TELUGU.search(sentence):
        return sentence

    try:
        tokenizer, model, ip = get_translation_engine()

        protected_sentence, placeholders = _protect_urls(sentence)

        batch = ip.preprocess_batch(
            [protected_sentence],
            src_lang=SRC_LANG,
            tgt_lang=TGT_LANG
        )

        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                max_length=256,
                num_beams=5,
                num_return_sequences=1
            )

        decoded = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        postprocessed = ip.postprocess_batch(
            decoded,
            lang=TGT_LANG
        )

        translated = postprocessed[0]
        final_sentence = _restore_urls(translated, placeholders)
        return final_sentence.strip()

    except Exception as e:
        logger.warning(f"Translation failed for sentence '{sentence[:50]}...': {e}. Falling back to original.")
        return sentence


def translate_to_english(text: str) -> str:
    """
    Sentence-aware translation of text:
    - Splits text into sentence units
    - Translates Telugu/mixed sentences individually
    - Preserves already-English sentences verbatim
    - Combines results back into natural text
    """
    if not text or not text.strip():
        return ""

    # If entire text contains zero Telugu characters, return immediately
    if not RE_TELUGU.search(text):
        return text.strip()

    sentences = split_into_sentences(text)
    translated_sentences = []

    for s in sentences:
        if RE_TELUGU.search(s):
            translated_sentences.append(translate_sentence(s))
        else:
            # English sentence: keep unmodified
            translated_sentences.append(s)

    return " ".join(translated_sentences).strip()


def translate_with_metadata(text: str) -> Dict[str, Any]:
    """
    Translates text and returns rich translation metadata.
    """
    if not text or not text.strip():
        return {
            "translated_text": "",
            "was_translated": False,
            "sentences_translated": 0,
            "sentences_skipped": 0
        }

    has_telugu = bool(RE_TELUGU.search(text))
    if not has_telugu:
        return {
            "translated_text": text.strip(),
            "was_translated": False,
            "sentences_translated": 0,
            "sentences_skipped": len(split_into_sentences(text))
        }

    sentences = split_into_sentences(text)
    translated_parts = []
    translated_count = 0
    skipped_count = 0

    for s in sentences:
        if RE_TELUGU.search(s):
            translated_parts.append(translate_sentence(s))
            translated_count += 1
        else:
            translated_parts.append(s)
            skipped_count += 1

    final_text = " ".join(translated_parts).strip()

    return {
        "translated_text": final_text,
        "was_translated": translated_count > 0,
        "sentences_translated": translated_count,
        "sentences_skipped": skipped_count
    }


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    test_cases = [
        "This is already an English sentence. The RBI kept the repo rate at 6.5%.",
        "రిజర్వ్ బ్యాంక్ వడ్డీ రేటును తగ్గించింది.",
        "PM Modi announced a new policy. తెలంగాణ ప్రభుత్వం కొత్త పథకాన్ని ప్రకటించింది. Visit https://example.com/info for details.",
        "నేను school కి వెళ్తాను and I study computer science.",
    ]

    print("=" * 65)
    print("  RAVEN - Sentence-Aware Translation Module Test")
    print("=" * 65)

    for i, sample in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"  Input: {sample}")
        meta = translate_with_metadata(sample)
        print(f"  Output: {meta['translated_text']}")
        print(f"  Translated={meta['was_translated']}, Translated sentences={meta['sentences_translated']}, Skipped sentences={meta['sentences_skipped']}")
