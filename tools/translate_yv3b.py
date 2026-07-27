import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

SOURCE_URL = "https://www.gutenberg.org/cache/epub/71064/pg71064.txt"
MODEL_ID = "Helsinki-NLP/opus-mt-en-zh"
OUT_DIR = Path("generated/book3b")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(value: str) -> int:
    total = 0
    previous = 0
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
    text = re.sub(r"(?m)^(\d+\.\s*)?([A-Z])\s+([a-z])", lambda m: (m.group(1) or "") + m.group(2) + m.group(3), text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def extract_chapters(text: str) -> list[dict]:
    text = clean_text(text)
    matches = list(re.finditer(r"(?m)^\s*CHAPTER\s+([IVXLCDM]+)(?:\.|\s*$).*", text))
    selected = []
    for match in matches:
        number = roman_to_int(match.group(1))
        if 61 <= number <= 122:
            selected.append((number, match.start(), match.end(), match.group(1)))
    by_number = {}
    for item in selected:
        by_number.setdefault(item[0], item)
    missing = [n for n in range(61, 123) if n not in by_number]
    if missing:
        raise RuntimeError(f"Missing chapter headings in source: {missing}")
    ordered = [by_number[n] for n in range(61, 123)]
    chapters = []
    for index, (number, start, end, roman) in enumerate(ordered):
        next_start = ordered[index + 1][1] if index + 1 < len(ordered) else len(text)
        raw = text[start:next_start].strip()
        raw = re.split(r"(?m)^\s*FOOTNOTES\s*$", raw, maxsplit=1)[0].rstrip()
        paragraphs = [re.sub(r"\s*\n\s*", " ", block).strip() for block in re.split(r"\n\s*\n+", raw) if block.strip()]
        blocks = []
        for paragraph in paragraphs:
            if re.fullmatch(r"CHAPTER\s+[IVXLCDM]+(?:\.|\s*)", paragraph):
                continue
            if paragraph.startswith("*** END OF") or "Project Gutenberg" in paragraph:
                break
            kind = "text"
            verse = None
            if paragraph.upper().startswith("SECTION "):
                kind = "section"
            elif paragraph.startswith("Argument"):
                kind = "argument"
            match = re.match(r"^(\d+)\.\s*(.*)$", paragraph, flags=re.S)
            if match:
                verse = int(match.group(1))
                paragraph = match.group(2).strip()
                kind = "verse"
            blocks.append({"kind": kind, "number": verse, "en": paragraph})
        if not blocks:
            raise RuntimeError(f"No content extracted for chapter {number}")
        chapters.append({"chapter": number, "roman": roman, "blocks": blocks})
    return chapters


TERM_REPLACEMENTS = [
    (r"Yoga[- ]V[aá]sishtha", "《瑜伽婆悉多》"),
    (r"V[aá]sishtha", "婆悉多（Vasiṣṭha）"),
    (r"R[aá]ma", "罗摩（Rāma）"),
    (r"\bBrahman\b", "梵（Brahman）"),
    (r"\bBrahm[aá]\b", "梵天（Brahmā）"),
    (r"\bAtman\b", "自性（Ātman）"),
    (r"\bj[ií]va\b", "个体生命（jīva）"),
    (r"\bmoksha\b", "解脱（mokṣa）"),
    (r"\bm[aá]y[aá]\b", "幻相（māyā）"),
    (r"\bsam[aá]dhi\b", "三摩地（samādhi）"),
    (r"\bavidy[aá]\b", "无明（avidyā）"),
    (r"\bv[aá]san[aá]\b", "习气（vāsanā）"),
    (r"\bsankalpa\b", "意志构想（saṅkalpa）"),
    (r"subtle body", "微细身"),
    (r"gross body", "粗重身"),
    (r"living liberation", "现生解脱（jīvanmukti）"),
    (r"final liberation", "究竟解脱"),
]


def protect_terms(text: str) -> str:
    for pattern, replacement in TERM_REPLACEMENTS:
        text = re.sub(pattern, replacement, text, flags=re.I)
    return text


def split_for_model(text: str, tokenizer, max_tokens: int = 430) -> list[str]:
    if len(tokenizer(text, add_special_tokens=True).input_ids) <= max_tokens:
        return [text]
    sentences = re.split(r"(?<=[.!?;:])\s+", text)
    chunks = []
    current = ""
    for sentence in sentences:
        trial = (current + " " + sentence).strip()
        if current and len(tokenizer(trial, add_special_tokens=True).input_ids) > max_tokens:
            chunks.append(current)
            current = sentence
        else:
            current = trial
    if current:
        chunks.append(current)
    final = []
    for chunk in chunks:
        if len(tokenizer(chunk, add_special_tokens=True).input_ids) <= max_tokens:
            final.append(chunk)
            continue
        words = chunk.split()
        current_words = []
        for word in words:
            trial = " ".join(current_words + [word])
            if current_words and len(tokenizer(trial, add_special_tokens=True).input_ids) > max_tokens:
                final.append(" ".join(current_words))
                current_words = [word]
            else:
                current_words.append(word)
        if current_words:
            final.append(" ".join(current_words))
    return final


def normalize_zh(text: str) -> str:
    text = re.sub(r"\s+([，。；：！？、）】》])", r"\1", text)
    text = re.sub(r"([（【《])\s+", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def translate(chapters: list[dict]) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    model.eval()
    torch.set_num_threads(max(2, os.cpu_count() or 2))
    completed = []
    checkpoint = OUT_DIR / "checkpoint.json"
    if checkpoint.exists():
        try:
            completed = json.loads(checkpoint.read_text(encoding="utf-8"))
        except Exception:
            completed = []
    completed_numbers = {item["chapter"] for item in completed}
    for chapter in chapters:
        if chapter["chapter"] in completed_numbers:
            continue
        jobs = []
        refs = []
        for block_index, block in enumerate(chapter["blocks"]):
            source = protect_terms(block["en"])
            parts = split_for_model(source, tokenizer)
            for part_index, part in enumerate(parts):
                jobs.append(part)
                refs.append((block_index, part_index))
        outputs = []
        batch_size = 12
        for start in range(0, len(jobs), batch_size):
            batch = jobs[start:start + batch_size]
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.inference_mode():
                generated = model.generate(**encoded, max_new_tokens=512, num_beams=3, length_penalty=1.0)
            outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        grouped = {}
        for (block_index, part_index), translated in zip(refs, outputs):
            grouped.setdefault(block_index, []).append((part_index, normalize_zh(translated)))
        for block_index, values in grouped.items():
            chapter["blocks"][block_index]["zh"] = "".join(value for _, value in sorted(values))
        completed.append(chapter)
        completed.sort(key=lambda item: item["chapter"])
        checkpoint.write_text(json.dumps(completed, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Translated chapter {chapter['chapter']}/122", flush=True)
    return completed


def audit(chapters: list[dict]) -> dict:
    chapter_numbers = [chapter["chapter"] for chapter in chapters]
    missing_chapters = [n for n in range(61, 123) if n not in chapter_numbers]
    untranslated = []
    numbered_passages = 0
    for chapter in chapters:
        for index, block in enumerate(chapter["blocks"]):
            if block.get("number") is not None:
                numbered_passages += 1
            if not block.get("zh"):
                untranslated.append([chapter["chapter"], index])
    return {
        "chapters": len(chapters),
        "chapter_range": [min(chapter_numbers), max(chapter_numbers)] if chapter_numbers else [],
        "missing_chapters": missing_chapters,
        "numbered_passages": numbered_passages,
        "untranslated_blocks": untranslated,
    }


def main() -> None:
    text = download_text()
    (OUT_DIR / "source.txt").write_text(text, encoding="utf-8")
    chapters = extract_chapters(text)
    translated = translate(chapters)
    report = audit(translated)
    if report["missing_chapters"] or report["untranslated_blocks"]:
        raise RuntimeError(json.dumps(report, ensure_ascii=False))
    payload = {
        "work": "Yoga Vasishta",
        "book": "Book III-B: On Creation",
        "scope": "Chapters 61-122",
        "language": "Simplified Chinese",
        "source": "Vihari-Lala Mitra public-domain English translation, Project Gutenberg ebook 71064",
        "audit": report,
        "chapters": translated,
    }
    (OUT_DIR / "book3b_zh.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
