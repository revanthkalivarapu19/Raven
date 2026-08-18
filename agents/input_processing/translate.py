"""
RAVEN - LOCAL TRANSLATION MODULE

Purpose:
This module translates Telugu and Telugu-English mixed text into
English using the locally running:
ai4bharat/indictrans2-indic-en-dist-200M
"""

import logging
import time
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
#auto tokenizer converts the text into tokens/token id's
#automodelforseq2seqlm  loads the actual translation model our input is one telugu sentence and the output is english sentence 
from IndicTransToolkit.processor import IndicProcessor

logger = logging.getLogger(__name__)#monitoring and debugging 

MODEL_NAME = "ai4bharat/indictrans2-indic-en-dist-200M"
SRC_LANG = "tel_Telu"#telugu written in telugu script 
TGT_LANG = "eng_Latn"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
).to(DEVICE)
#we load the model outside the translation function becuase it may load the model for every claim
ip = IndicProcessor(inference=True) 

def translate_to_english(text: str) -> str:
    if not text:
        return ""

    try:
        batch = ip.preprocess_batch(
            [text],
            src_lang=SRC_LANG,
            tgt_lang=TGT_LANG
        )
        #preprocess batch expects list/batch of sentences because it was the format expected by indictrans2 

        inputs = tokenizer(
            batch,
            truncation=True,#if the input is too long it allows to modelss allowed length
            padding="longest",
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                max_length=256,
                num_beams=5,#it is known as beam search where instead of choosing next possible word immeadiately , the model considers multiple sequences it keeps upto 5 candidate sequnces and eventually chooses the best one.
                num_return_sequences=1
            )
            
        decoded = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )#after generate we have token id and no english yet
        
        postprocessed = ip.postprocess_batch(
            decoded,
            lang=TGT_LANG
        )
        
        return postprocessed[0]#we pass one sentence as a list there fore output is also an list 
    except Exception as e:
        logger.exception("Translation failed")
        return text

if __name__ == "__main__":
    test_cases = [
        "రాష్ట్రంలో ప్రజా రవాణా సేవలను తాత్కాలికంగా నిలిపివేశారు.",
        "ప్రభుత్వం వచ్చే నెలలో కొత్త పథకాన్ని ప్రారంభించనుంది.",
        "ఈ వ్యాక్సిన్ వ్యాధి నుంచి రక్షణ కల్పిస్తుంది.",
        "రిజర్వ్ బ్యాంక్ వడ్డీ రేటును తగ్గించింది.",
        "నేను school కి వెళ్తాను and I study computer science.",
        "గీతా ఆంటీ"
    ]
    
    print(f"Using device: {DEVICE}")
    for text in test_cases:
        start_time = time.perf_counter()
        output = translate_to_english(text)
        elapsed_time = time.perf_counter() - start_time#measures how long translation taken
        
        print("Input:", text)
        print("Output:", output)
        print(f"Translation time: {elapsed_time:.4f} seconds\n")
