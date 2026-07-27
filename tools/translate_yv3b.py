import json
import os
import re
from pathlib import Path

import ctranslate2
import requests
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

SOURCE_URL = "https://www.gutenberg.org/cache/epub/71063/pg71063.txt"
MODEL_ID = "Helsinki-NLP/opus-mt-en-zh"
CT2_MODEL_ID = "gaudi/opus-mt-en-zh-ctranslate2"
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
    text = re.sub(
        r"(?m)^(\d+\.\s*)?([A-ZÁÉÍÓÚ])\s+([a-záéíóú])",
        lambda m: (m.group(1) or "") + m.group(2) + m.group(3),
        text,
    )
    return text


def extract_chapters(text: str) -> list[dict]:
    text = clean_text(text)
    # Discard the table of contents. The actual Book III text begins after this marker.
    markers = [m.start() for m in re.finditer(r"(?m)^\s*YOGA\s+V[ÁA]SISHTHA\.\s*$", text)]
    if not markers:
        raise RuntimeError("Could not locate the start of the actual Book III text")
    body = text[markers[-1]:]

    matches = list(re.finditer(r"(?m)^\s*CHAPTER\s+([IVXLCDM]+)\.\s*$", body))
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
            kind = "text"
            verse = None
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

        # The first prose paragraph often has a decorative drop-cap and no visible '1.'.
        for i, block in enumerate(blocks):
            if block["kind"] == "text":
                next_numbers = [b.get("number") for b in blocks[i + 1:i + 4]]
                if 2 in next_numbers:
                    block["kind"] = "verse"
                    block["number"] = 1
                    break

        if not blocks or not any(b.get("number") is not None for b in blocks):
            raise RuntimeError(f"No numbered content extracted for chapter {number}")
        chapters.append({"chapter": number, "roman": roman, "blocks": blocks})
    return chapters


TERM_REPLACEMENTS = [
    (r"Yoga[- ]V[áa]sishtha", "《瑜伽婆悉多》"),
    (r"V[áa]sishtha", "婆悉多（Vasiṣṭha）"),
    (r"R[áa]ma", "罗摩（Rāma）"),
    (r"\bBrahman\b", "梵（Brahman）"),
    (r"\bBrahm[áa]\b", "梵天（Brahmā）"),
    (r"\bAtman\b", "自性（Ātman）"),
    (r"\bj[íi]va\b", "个体生命（jīva）"),
    (r"\bmoksha\b", "解脱（mokṣa）"),
    (r"\bm[áa]y[áa]\b", "幻相（māyā）"),
    (r"\bsam[áa]dhi\b", "三摩地（samādhi）"),
    (r"\bavidy[áa]\b", "无明（avidyā）"),
    (r"\bv[áa]san[áa]\b", "习气（vāsanā）"),
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
    if len(tokenizer.tokenize(text)) <= max_tokens:
        return [text]
    sentences = re.split(r"(?<=[.!?;:])\s+", text)
    chunks, current = [], ""
    for sentence in sentences:
        trial = (current + " " + sentence).strip()
        if current and len(tokenizer.tokenize(trial)) > max_tokens:
            chunks.append(current)
            current = sentence
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


def normalize_zh(text: str) -> str:
    text = re.sub(r"\s+([，。；：！？、）】》])", r"\1", text)
    text = re.sub(r"([（【《])\s+", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def translate(chapters: list[dict]) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model_path = snapshot_download(CT2_MODEL_ID)
    translator = ctranslate2.Translator(
        model_path,
        device="cpu",
        compute_type="int8",
        inter_threads=max(1, min(4, os.cpu_count() or 2)),
        intra_threads=1,
    )

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
        jobs, refs = [], []
        for block_index, block in enumerate(chapter["blocks"]):
            source = protect_terms(block["en"])
            parts = split_for_model(source, tokenizer)
            for part_index, part in enumerate(parts):
                jobs.append(tokenizer.tokenize(part))
                refs.append((block_index, part_index))

        outputs = []
        batch_size = 64
        for start in range(0, len(jobs), batch_size):
            batch = jobs[start:start + batch_size]
            results = translator.translate_batch(
                batch,
                beam_size=3,
                max_decoding_length=512,
                batch_type="tokens",
                max_batch_size=4096,
            )
            outputs.extend(
                normalize_zh(tokenizer.convert_tokens_to_string(result.hypotheses[0]))
                for result in results
            )

        grouped = {}
        for (block_index, part_index), translated in zip(refs, outputs):
            grouped.setdefault(block_index, []).append((part_index, translated))
        for block_index, values in grouped.items():
            chapter["blocks"][block_index]["zh"] = "".join(
                value for _, value in sorted(values)
            )
        completed.append(chapter)
        completed.sort(key=lambda item: item["chapter"])
        checkpoint.write_text(json.dumps(completed, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Translated chapter {chapter['chapter']}/122", flush=True)
    return completed


def audit(chapters: list[dict]) -> dict:
    chapter_numbers = [chapter["chapter"] for chapter in chapters]
    missing_chapters = [n for n in range(61, 123) if n not in chapter_numbers]
    untranslated, numbering_gaps = [], {}
    numbered_passages = 0
    for chapter in chapters:
        numbers = []
        for index, block in enumerate(chapter["blocks"]):
            if block.get("number") is not None:
                numbered_passages += 1
                numbers.append(block["number"])
            if not block.get("zh"):
                untranslated.append([chapter["chapter"], index])
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
    }


def write_chapter_text(chapter: dict) -> None:
    lines = []
    title_block = next((b for b in chapter["blocks"] if b["kind"] == "title"), None)
    title = title_block.get("zh", "") if title_block else ""
    lines.append(f"第{chapter['chapter']}章　{title}")
    lines.append("")
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
    text = download_text()
    chapters = extract_chapters(text)
    translated = translate(chapters)
    report = audit(translated)
    if report["missing_chapters"] or report["untranslated_blocks"]:
        raise RuntimeError(json.dumps(report, ensure_ascii=False))
    for chapter in translated:
        write_chapter_text(chapter)
    payload = {
        "work": "Yoga Vasishta",
        "book": "Book III-B: On Creation",
        "scope": "Chapters 61-122",
        "language": "Simplified Chinese",
        "source": "Vihari-Lala Mitra public-domain English translation, Project Gutenberg ebook 71063",
        "audit": report,
        "chapters": translated,
    }
    (OUT_DIR / "book3b_zh.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
