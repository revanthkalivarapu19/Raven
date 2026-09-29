import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"

SRC_LANG = "tel_Telu"
TGT_LANG = "eng_Latn"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)
print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
).to(DEVICE)

print("Model loaded successfully!")

ip = IndicProcessor(inference=True)

test_cases = [
    # 1. General news
    "రాష్ట్రంలో ప్రజా రవాణా సేవలను తాత్కాలికంగా నిలిపివేశారు.",

    # 2. Politics
    "ప్రభుత్వం వచ్చే నెలలో కొత్త పథకాన్ని ప్రారంభించనుంది.",

    # 3. Medical
    "ఈ వ్యాక్సిన్ వ్యాధి నుంచి రక్షణ కల్పిస్తుంది.",

    # 4. Finance
    "రిజర్వ్ బ్యాంక్ వడ్డీ రేటును తగ్గించింది.",

    # 5. Technology
    "కొత్త సాంకేతికతను ప్రభుత్వం ప్రజలకు అందుబాటులోకి తీసుకువచ్చింది.",

    # 6. Longer Telugu sentence
    "ఆర్థిక వ్యవస్థను మెరుగుపరచడానికి ప్రభుత్వం అనేక కొత్త చర్యలను ప్రకటించింది.",

    # 7. Telugu + English mixed
    "నేను school కి వెళ్తాను and I study computer science."
]


import time

for text in test_cases:
    print("\n" + "-" * 60)
    print("Input:")
    print(text)

    start = time.time()

    batch = ip.preprocess_batch(
        [text],
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

    translations = ip.postprocess_batch(
        decoded,
        lang=TGT_LANG
    )

    elapsed = time.time() - start

    print("Output:")
    print(translations[0])
    print(f"Time: {elapsed:.2f} seconds")