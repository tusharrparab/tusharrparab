import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

import ctranslate2
import requests
from huggingface_hub import snapshot_download
from opencc import OpenCC
from transformers import M2M100Tokenizer

SOURCE_URL = "https://www.gutenberg.org/cache/epub/71063/pg71063.txt"
BASE_TOKENIZER = "facebook/m2m100_418M"
CT2_MODEL_ID = "entai2965/m2m100-418M-ctranslate2"
OUT_DIR = Path("generated/book3b")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(value: str) -> int:
    total, previous = 0, 0
    for char in reversed(value.upper()):
        number = ROMAN[char]
        if number < previous:
            total -= number
        else:
            total += number
            previous = number
    return total


def download_text() -> str:
    response = requests.get(SOURCE_URL, timeout=180)
    response.raise_for_status()
    return response.content.decode("utf-8-sig", errors="replace")


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(
        r"(?m)^(\d+\.\s*)?([A-ZÁÉÍÓÚ])\s+([a-záéíóú])",
        lambda m: (m.group(1) or "") + m.group(2) + m.group(3),
        text,
    )
    return text


def extract_chapters(text: str) -> list[dict]:
    text = clean_text(text)
    markers = [m.start() for m in re.finditer(r"(?m)^\s*YOGA\s+V[ÁA]SISHTHA\.\s*$", text)]
    if not markers:
        raise RuntimeError("Could not locate the actual Book III text")
    body = text[markers[-1]:]
    matches = list(re.finditer(r"(?m)^\s*CHAPTER\s+([IVXLCDM]+)\.\s*$", body))
    by_number = {}
    for match in matches:
        number = roman_to_int(match.group(1))
        if 61 <= number <= 122:
            by_number.setdefault(number, (number, match.start(), match.end(), match.group(1)))
    missing = [n for n in range(61, 123) if n not in by_number]
    if missing:
        raise RuntimeError(f"Missing chapter headings: {missing}")
    ordered = [by_number[n] for n in range(61, 123)]
    chapters = []
    for index, (number, start, end, roman) in enumerate(ordered):
        next_start = ordered[index + 1][1] if index + 1 < len(ordered) else len(body)
        raw = body[end:next_start].strip()
        raw = re.split(r"(?m)^\s*FOOTNOTES\s*$", raw, maxsplit=1)[0].rstrip()
        raw = re.split(r"(?m)^\s*\*\*\* END OF", raw, maxsplit=1)[0].rstrip()
        paragraphs = [
            re.sub(r"\s*\n\s*", " ", block).strip()
            for block in re.split(r"\n\s*\n+", raw)
            if block.strip()
        ]
        blocks = []
        for p_index, paragraph in enumerate(paragraphs):
            if re.fullmatch(r"[-*• ]+", paragraph):
                continue
            kind, verse = "text", None
            if p_index == 0:
                kind = "title"
            elif paragraph.upper().startswith("SECTION "):
                kind = "section"
            elif paragraph.startswith("Argument"):
                kind = "argument"
            elif re.match(r"^(Gloss|Glossary|Note|Remark)[.:—-]", paragraph, flags=re.I):
                kind = "note"
            numbered = re.match(r"^(\d+)\.\s*(.*)$", paragraph, flags=re.S)
            if numbered:
                verse = int(numbered.group(1))
                paragraph = numbered.group(2).strip()
                kind = "verse"
            blocks.append({"kind": kind, "number": verse, "en": paragraph})
        for i, block in enumerate(blocks):
            if block["kind"] == "text":
                if 2 in [b.get("number") for b in blocks[i + 1:i + 4]]:
                    block["kind"], block["number"] = "verse", 1
                    break
        if not blocks or not any(b.get("number") is not None for b in blocks):
            raise RuntimeError(f"No numbered content extracted for chapter {number}")
        chapters.append({"chapter": number, "roman": roman, "blocks": blocks})
    return chapters


def ascii_source(text: str) -> str:
    text = text.replace("’", "'").replace("“", '"').replace("”", '"').replace("—", " - ")
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def split_for_model(text: str, tokenizer, max_tokens: int = 125) -> list[str]:
    text = ascii_source(text)
    if len(tokenizer.tokenize(text)) <= max_tokens:
        return [text]
    units = re.split(r"(?<=[.!?;:])\s+|(?<=,)\s+", text)
    chunks, current = [], ""
    for unit in units:
        trial = (current + " " + unit).strip()
        if current and len(tokenizer.tokenize(trial)) > max_tokens:
            chunks.append(current)
            current = unit
        else:
            current = trial
    if current:
        chunks.append(current)
    final = []
    for chunk in chunks:
        if len(tokenizer.tokenize(chunk)) <= max_tokens:
            final.append(chunk)
            continue
        words, current_words = chunk.split(), []
        for word in words:
            trial = " ".join(current_words + [word])
            if current_words and len(tokenizer.tokenize(trial)) > max_tokens:
                final.append(" ".join(current_words))
                current_words = [word]
            else:
                current_words.append(word)
        if current_words:
            final.append(" ".join(current_words))
    return final


cc = OpenCC("t2s")


def terminology(source: str, text: str) -> str:
    text = cc.convert(text)
    replacements = {
        "拉玛": "罗摩", "拉马": "罗摩", "罗摩王子": "罗摩",
        "瓦西斯塔": "婆悉多", "瓦西什塔": "婆悉多", "瓦希斯塔": "婆悉多",
        "卡尔卡蒂": "卡尔卡蒂", "卡卡蒂": "卡尔卡蒂",
        "吉瓦": "个体生命（jīva）", "阿特曼": "自性（Ātman）",
        "玛雅": "幻相（māyā）", "萨马迪": "三摩地（samādhi）",
        "莫克沙": "解脱（mokṣa）", "桑卡尔帕": "意志构想（saṅkalpa）",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    if re.search(r"\bBrahman\b", source):
        for old in ("婆罗门", "布拉曼", "梵文", "布拉赫曼"):
            text = text.replace(old, "梵（Brahman）")
    if re.search(r"\bBrahm[aá]\b", source, flags=re.I):
        for old in ("布拉玛", "布拉马", "婆罗摩", "梵天"):
            text = text.replace(old, "梵天（Brahmā）")
    if re.search(r"\bAtman\b", source, flags=re.I):
        text = text.replace("阿特曼", "自性（Ātman）")
    if re.search(r"\bj[íi]va\b", source, flags=re.I):
        text = text.replace("吉瓦", "个体生命（jīva）")
    text = text.replace("<unk>", "〔专名〕")
    text = re.sub(r"\s+([，。；：！？、）】》])", r"\1", text)
    text = re.sub(r"([（【《])\s+", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def translate(chapters: list[dict]) -> list[dict]:
    tokenizer = M2M100Tokenizer.from_pretrained(BASE_TOKENIZER)
    tokenizer.src_lang = "en"
    model_path = snapshot_download(CT2_MODEL_ID)
    translator = ctranslate2.Translator(
        model_path, device="cpu", compute_type="int8",
        inter_threads=max(1, min(4, os.cpu_count() or 2)), intra_threads=1,
    )
    target_prefix = tokenizer.convert_ids_to_tokens([tokenizer.get_lang_id("zh")])
    completed = []
    for chapter in chapters:
        jobs, refs, sources = [], [], []
        for block_index, block in enumerate(chapter["blocks"]):
            parts = split_for_model(block["en"], tokenizer)
            for part_index, part in enumerate(parts):
                jobs.append(tokenizer.convert_ids_to_tokens(tokenizer.encode(part)))
                refs.append((block_index, part_index))
                sources.append(block["en"])
        outputs = []
        for start in range(0, len(jobs), 48):
            batch = jobs[start:start + 48]
            results = translator.translate_batch(
                batch,
                target_prefix=[target_prefix] * len(batch),
                beam_size=4,
                max_decoding_length=384,
                batch_type="tokens",
                max_batch_size=4096,
            )
            for source, result in zip(sources[start:start + 48], results):
                ids = tokenizer.convert_tokens_to_ids(result.hypotheses[0][1:])
                output = tokenizer.decode(ids, skip_special_tokens=True).strip()
                outputs.append(terminology(source, output))
        grouped = {}
        for (block_index, part_index), translated in zip(refs, outputs):
            grouped.setdefault(block_index, []).append((part_index, translated))
        for block_index, values in grouped.items():
            chapter["blocks"][block_index]["zh"] = "".join(v for _, v in sorted(values))
        completed.append(chapter)
        print(f"Translated chapter {chapter['chapter']}/122", flush=True)
    return completed


def suspicious_repetition(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 80:
        return False
    grams = [compact[i:i+12] for i in range(len(compact)-11)]
    counts = Counter(grams)
    return counts and counts.most_common(1)[0][1] >= 5


def audit(chapters: list[dict]) -> dict:
    chapter_numbers = [chapter["chapter"] for chapter in chapters]
    missing_chapters = [n for n in range(61, 123) if n not in chapter_numbers]
    untranslated, numbering_gaps, short_blocks, repetition_blocks, unknown_blocks = [], {}, [], [], []
    numbered_passages = 0
    for chapter in chapters:
        numbers = []
        for index, block in enumerate(chapter["blocks"]):
            zh = block.get("zh", "")
            if block.get("number") is not None:
                numbered_passages += 1
                numbers.append(block["number"])
            if not zh:
                untranslated.append([chapter["chapter"], index])
                continue
            cjk = len(re.findall(r"[\u3400-\u9fff]", zh))
            if len(block.get("en", "")) > 90 and cjk < max(8, len(block["en"]) * 0.12):
                short_blocks.append([chapter["chapter"], index, len(block["en"]), cjk])
            if suspicious_repetition(zh):
                repetition_blocks.append([chapter["chapter"], index])
            if "〔专名〕" in zh:
                unknown_blocks.append([chapter["chapter"], index])
        if numbers:
            expected = set(range(min(numbers), max(numbers) + 1))
            gaps = sorted(expected - set(numbers))
            if gaps:
                numbering_gaps[str(chapter["chapter"])] = gaps
    return {
        "chapters": len(chapters),
        "chapter_range": [min(chapter_numbers), max(chapter_numbers)] if chapter_numbers else [],
        "missing_chapters": missing_chapters,
        "numbered_passages": numbered_passages,
        "source_numbering_gaps": numbering_gaps,
        "untranslated_blocks": untranslated,
        "short_translation_blocks": short_blocks,
        "repetition_blocks": repetition_blocks,
        "unknown_name_blocks": unknown_blocks,
    }


def write_chapter_text(chapter: dict) -> None:
    lines = []
    title_block = next((b for b in chapter["blocks"] if b["kind"] == "title"), None)
    title = title_block.get("zh", "") if title_block else ""
    lines.extend([f"第{chapter['chapter']}章　{title}", ""])
    for block in chapter["blocks"]:
        if block is title_block:
            continue
        zh = block.get("zh", "").strip()
        if not zh:
            continue
        if block.get("number") is not None:
            lines.append(f"{block['number']}. {zh}")
        elif block["kind"] == "argument":
            lines.append(f"论旨：{zh}")
        else:
            lines.append(zh)
        lines.append("")
    (OUT_DIR / f"ch{chapter['chapter']:03d}.txt").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    chapters = extract_chapters(download_text())
    translated = translate(chapters)
    report = audit(translated)
    if report["missing_chapters"] or report["untranslated_blocks"] or report["repetition_blocks"]:
        raise RuntimeError(json.dumps(report, ensure_ascii=False))
    for chapter in translated:
        write_chapter_text(chapter)
    payload = {
        "work": "Yoga Vasishta",
        "book": "Book III-B: On Creation",
        "scope": "Chapters 61-122",
        "language": "Simplified Chinese",
        "translation_method": "Machine-assisted M2M100 translation with structural and terminology audit",
        "source": "Vihari-Lala Mitra public-domain English translation, Project Gutenberg ebook 71063",
        "audit": report,
        "chapters": translated,
    }
    (OUT_DIR / "book3b_zh.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
