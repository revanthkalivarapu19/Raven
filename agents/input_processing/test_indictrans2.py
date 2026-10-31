"""Optional offline IndicTrans smoke test."""

import pytest


MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"


def test_indictrans2_local_model():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    processor_module = pytest.importorskip("IndicTransToolkit.processor")

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            MODEL_NAME, trust_remote_code=True, local_files_only=True
        )
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME, trust_remote_code=True, local_files_only=True
        )
    except Exception as exc:
        pytest.skip(f"IndicTrans model is not available in the local cache: {exc}")

    processor = processor_module.IndicProcessor(inference=True)
    batch = processor.preprocess_batch(
        ["A short test sentence."], src_lang="tel_Telu", tgt_lang="eng_Latn"
    )
    inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt")
    with torch.no_grad():
        generated = model.generate(**inputs, max_length=32, num_beams=1)
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    assert processor.postprocess_batch(decoded, lang="eng_Latn")[0]
