import json
import os
from pathlib import Path

import ctranslate2
from huggingface_hub import snapshot_download
from transformers import M2M100Tokenizer

CT2_MODEL_ID = "entai2965/m2m100-418M-ctranslate2"
BASE_TOKENIZER = "facebook/m2m100_418M"
SAMPLES = [
    "Rama said: Please explain to me the cause of our error in viewing the objective world as real, and tell me where this world comes from.",
    "Vasishta said: All objects known by words and their meanings exist only as ideas in consciousness; no separate material substance is established by a name.",
    "The quality of a bracelet is not different from the gold of which it is made, nor is a wave separate from water; likewise the world is not different from the spirit of God.",
    "The Spirit of God does not reside inside creation as one object inside another. The relation between them is like the relation between a wave and water.",
    "The demoness Karkati was dark as ink, strong as a rock, and afflicted by an insatiable hunger that no food could satisfy."
]

def main():
    model_path = snapshot_download(CT2_MODEL_ID)
    tokenizer = M2M100Tokenizer.from_pretrained(BASE_TOKENIZER)
    tokenizer.src_lang = "en"
    translator = ctranslate2.Translator(
        model_path, device="cpu", compute_type="int8",
        inter_threads=max(1, min(4, os.cpu_count() or 2)), intra_threads=1,
    )
    sources = [tokenizer.convert_ids_to_tokens(tokenizer.encode(s)) for s in SAMPLES]
    prefix = tokenizer.convert_ids_to_tokens([tokenizer.get_lang_id("zh")])
    results = translator.translate_batch(
        sources,
        target_prefix=[prefix] * len(sources),
        beam_size=4,
        max_decoding_length=256,
    )
    outputs = [
        tokenizer.decode(tokenizer.convert_tokens_to_ids(result.hypotheses[0][1:]), skip_special_tokens=True).strip()
        for result in results
    ]
    Path("generated/book3b_nllb_test").mkdir(parents=True, exist_ok=True)
    Path("generated/book3b_nllb_test/sample.json").write_text(
        json.dumps([{"en":e,"zh":z} for e,z in zip(SAMPLES,outputs)],ensure_ascii=False,indent=2),
        encoding="utf-8"
    )
    print(json.dumps(outputs,ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
