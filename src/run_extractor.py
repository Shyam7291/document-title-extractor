import os
import re
import csv
import html
import hashlib
from urllib.parse import urlparse, unquote, urljoin
from email.message import Message

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
import fitz


INPUT_FILE = "data/input/links_300.csv"
OUTPUT_FILE = "data/output/extracted_titles.csv"
DOWNLOAD_DIR = "downloads"
REQUEST_TIMEOUT = 15


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


IGNORE_WORDS = {
    "download", "view", "open", "read", "click", "file", "document",
    "pdf", "link", "here", "more", "details", "english"
}

BAD_SECTION_TITLES = {
    "disclosures",
    "reports",
    "announcements",
    "investor relations",
    "financial reports",
    "company announcements",
    "corporate governance",
    "board & governance",
    "board and governance",
    "financial information",
    "governance documents",
    "sustainability reports",
    "historical reports",
    "publications for investors",
    "esg documents",
    "table of contents",
}

GOOD_WORDS = [
    "annual", "annual report", "integrated report", "sustainability",
    "sustainability report", "esg", "climate", "governance",
    "financial", "financial statements", "statement", "report",
    "results", "presentation", "proxy", "agm", "csr", "brsr",
    "modern slavery", "committee", "charter", "policy", "notice",
    "minutes", "agenda", "voting", "remuneration", "human rights",
    "tax transparency", "tcfd", "tnfd", "cdp", "assurance",
    "annual return", "form of proxy", "certificate"
]

BAD_CONTAINS = [
    "all rights reserved",
    "enable javascript",
    "cookie",
    "terms of use",
    "click here",
    "download pdf",
    "general meeting can also be followed",
    "meritis, keizersgracht",
    "categories of third",
    "classification: c1 - controlled",
]

SOURCE_BASE_SCORE = {
    "html_context_static_file": 95,
    "html_context": 92,
    "link_text_static_file": 88,
    "link_text": 84,
    "raw_html_json": 86,
    "raw_html_context": 82,
    "content_disposition": 82,
    "url": 78,
    "pdf_metadata": 72,
    "pdf_first_page": 60,
}


session = requests.Session()
session.headers.update(HEADERS)
html_cache = {}


def normalize_text(text):
    if not text:
        return ""

    text = str(text)
    text = html.unescape(text)
    text = unquote(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_generic_action_title(text):
    text = normalize_text(text).lower()

    if not text:
        return True

    generic_patterns = [
        r"^download\s*(pdf|report|document|file)?$",
        r"^view\s*(pdf|report|document|file)?$",
        r"^open\s*(pdf|report|document|file)?$",
        r"^read\s*(more|report|document)?$",
        r"^click\s*here$",
        r"^learn\s*more$",
        r"opens?\s+in\s+(a\s+)?new\s+(window|tab)",
        r"^pdf$",
        r"^download$",
        r"^view$",
        r"^open$",
        r"^more$",
        r"^details$",
        r"^file$",
        r"^document$",
    ]

    return any(re.search(pattern, text, re.IGNORECASE) for pattern in generic_patterns)


def is_date_only(text):
    text = normalize_text(text)

    date_patterns = [
        r"^\d{1,2}\s+\w+,\s+\d{4}$",
        r"^\d{1,2}\s+\w+\s+\d{4}$",
        r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$",
        r"^\d{4}[/-]\d{1,2}[/-]\d{1,2}$",
    ]

    return any(re.match(pattern, text, re.IGNORECASE) for pattern in date_patterns)


def is_uuid_like(text):
    text = normalize_text(text).lower()
    text = text.replace(" ", "-")

    uuid_pattern = r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$"
    compact_hex = re.sub(r"[^a-f0-9]", "", text)

    if re.match(uuid_pattern, text):
        return True

    if len(compact_hex) >= 24 and re.fullmatch(r"[a-f0-9]+", compact_hex):
        return True

    return False


def is_number_file(text):
    text = normalize_text(text).lower()
    text = re.sub(r"\.(pdf|ashx|aspx|html?)$", "", text)
    text = text.replace("_", "-")
    return bool(re.match(r"^[0-9\-]+$", text))


def is_bad_title(text):
    text = normalize_text(text)
    lower = text.lower()

    if not text:
        return True

    if is_generic_action_title(text):
        return True

    if lower in IGNORE_WORDS:
        return True

    if lower in BAD_SECTION_TITLES:
        return True

    for bad in BAD_CONTAINS:
        if bad in lower:
            return True

    if len(text) < 4:
        return True

    if len(text) > 220:
        return True

    if is_date_only(text):
        return True

    if is_number_file(text):
        return True

    if is_uuid_like(text):
        return True

    if not re.search(r"[a-zA-Z]", text):
        return True

    return False


def has_good_signal(title):
    lower = normalize_text(title).lower()

    for word in GOOD_WORDS:
        if word in lower:
            return True

    if re.search(r"\b20\d{2}\b", lower):
        return True

    if re.search(r"\b(q[1-4]|h1|h2|fy|year[-\s]?end)\b", lower):
        return True

    return False


def title_quality_score(title, source):
    title = normalize_text(title)

    if is_bad_title(title):
        return 0

    score = SOURCE_BASE_SCORE.get(source, 40)

    word_count = len(title.split())

    if 3 <= word_count <= 18:
        score += 12

    if has_good_signal(title):
        score += 25

    if re.search(r"\b20\d{2}\b", title):
        score += 12

    if len(title) > 160:
        score -= 25

    if source == "pdf_first_page" and not has_good_signal(title):
        score -= 30

    return max(score, 0)


def add_candidate(candidates, title, source):
    title = normalize_text(title)

    if not title or is_bad_title(title):
        return

    candidates.append({
        "title": title,
        "source": source,
        "score": title_quality_score(title, source)
    })


def clean_title_from_url(doc_link):
    parsed = urlparse(doc_link)
    path = parsed.path
    filename = os.path.basename(path.strip("/"))

    if not filename:
        return ""

    filename = unquote(filename)
    filename = html.unescape(filename)
    filename = filename.split("?")[0]

    filename = re.sub(r"\.(pdf|ashx|aspx|html?|docx?|pptx?|xlsx?)$", "", filename, flags=re.I)

    if is_uuid_like(filename):
        return ""

    filename = filename.replace("+", " ")
    filename = filename.replace("-", " ")
    filename = filename.replace("_", " ")
    filename = normalize_text(filename)

    if is_bad_title(filename):
        return ""

    return filename


def normalize_url_key(url):
    parsed = urlparse(str(url))
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".lower().rstrip("/")


def get_last_path_segment(url):
    try:
        parsed = urlparse(str(url))
        return parsed.path.rstrip("/").split("/")[-1]
    except Exception:
        return ""


def links_match(a, b):
    a_key = normalize_url_key(a)
    b_key = normalize_url_key(b)

    if a_key == b_key:
        return True

    a_last = get_last_path_segment(a_key)
    b_last = get_last_path_segment(b_key)

    if a_last and b_last and a_last == b_last:
        return True

    return False


def fetch_html(page_url):
    if page_url in html_cache:
        return html_cache[page_url]

    try:
        response = session.get(page_url, timeout=REQUEST_TIMEOUT, allow_redirects=True)

        if response.status_code in [403, 406, 429]:
            parsed = urlparse(page_url)
            headers = HEADERS.copy()
            headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}/"
            response = session.get(page_url, timeout=REQUEST_TIMEOUT, headers=headers, allow_redirects=True)

        response.raise_for_status()
        html_cache[page_url] = response.text
        return response.text

    except Exception as error:
        print(f"HTML fetch failed for {page_url}: {error}")
        html_cache[page_url] = ""
        return ""


def collect_text_candidates_from_container(container):
    candidates = []

    if not container:
        return candidates

    for element in container.find_all(
        ["td", "th", "p", "span", "div", "h1", "h2", "h3", "h4", "strong", "a"],
        recursive=True
    ):
        text = normalize_text(element.get_text(" ", strip=True))

        if not is_bad_title(text):
            candidates.append(text)

    unique = []
    seen = set()

    for item in candidates:
        key = item.lower()

        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


def get_title_from_html_context(link):
    row = link.find_parent("tr")
    if row:
        row_candidates = collect_text_candidates_from_container(row)
        if row_candidates:
            return max(row_candidates, key=lambda x: title_quality_score(x, "html_context"))

    li = link.find_parent("li")
    if li:
        li_candidates = collect_text_candidates_from_container(li)
        if li_candidates:
            return max(li_candidates, key=lambda x: title_quality_score(x, "html_context"))

    current = link.parent
    levels_checked = 0

    while current and levels_checked < 5:
        if current.name in ["div", "section", "article", "p"]:
            block_candidates = collect_text_candidates_from_container(current)

            if block_candidates:
                return max(block_candidates, key=lambda x: title_quality_score(x, "html_context"))

        current = current.parent
        levels_checked += 1

    return ""


def get_link_text_title(link):
    text = normalize_text(link.get_text(" ", strip=True))

    if is_bad_title(text):
        return ""

    return text


def get_raw_html_context_candidates(html_text, doc_link, candidates):
    doc_id = get_last_path_segment(doc_link)

    if not doc_id:
        return

    index = html_text.find(doc_id)

    if index == -1:
        return

    start = max(0, index - 1500)
    end = min(len(html_text), index + 1500)
    context = html_text[start:end]

    json_patterns = [
        r'"title"\s*:\s*"([^"]{4,220})"',
        r'"name"\s*:\s*"([^"]{4,220})"',
        r'"field_title"\s*:\s*"([^"]{4,220})"',
        r'"description"\s*:\s*"([^"]{4,220})"',
        r'"linkText"\s*:\s*"([^"]{4,220})"',
    ]

    for pattern in json_patterns:
        for match in re.findall(pattern, context, flags=re.I):
            add_candidate(candidates, match, "raw_html_json")

    context = re.sub(r"<script.*?</script>", " ", context, flags=re.I | re.S)
    context = re.sub(r"<style.*?</style>", " ", context, flags=re.I | re.S)

    soup = BeautifulSoup(context, "html.parser")
    text = normalize_text(soup.get_text(" ", strip=True))

    chunks = re.split(r"\s{2,}|[\|\n\r\t]+", text)

    for chunk in chunks:
        chunk = normalize_text(chunk)
        if 2 <= len(chunk.split()) <= 28:
            add_candidate(candidates, chunk, "raw_html_context")


def get_html_candidates(page_url, doc_link):
    candidates = []
    html_text = fetch_html(page_url)

    if not html_text:
        return candidates

    soup = BeautifulSoup(html_text, "html.parser")

    for link in soup.find_all("a", href=True):
        full_url = urljoin(page_url, link.get("href", ""))

        if links_match(full_url, doc_link):
            html_title = get_title_from_html_context(link)
            link_text = get_link_text_title(link)

            if "/static-files/" in doc_link.lower():
                add_candidate(candidates, html_title, "html_context_static_file")
                add_candidate(candidates, link_text, "link_text_static_file")
            else:
                add_candidate(candidates, html_title, "html_context")
                add_candidate(candidates, link_text, "link_text")

    get_raw_html_context_candidates(html_text, doc_link, candidates)

    return candidates


def extract_filename_from_content_disposition(header_value):
    if not header_value:
        return ""

    try:
        message = Message()
        message["content-disposition"] = header_value
        filename = message.get_filename()

        if filename:
            filename = unquote(filename)
            filename = os.path.basename(filename)
            filename = re.sub(r"\.(pdf|docx?|pptx?|xlsx?)$", "", filename, flags=re.I)
            return normalize_text(filename)

    except Exception:
        return ""

    return ""


def download_document(doc_link):
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    file_hash = hashlib.md5(doc_link.encode("utf-8")).hexdigest()
    base_path = os.path.join(DOWNLOAD_DIR, file_hash)

    candidates = []

    try:
        response = session.get(doc_link, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        content_disposition = response.headers.get("content-disposition", "")

        header_filename = extract_filename_from_content_disposition(content_disposition)
        add_candidate(candidates, header_filename, "content_disposition")

        content = response.content

        if not content:
            return "", content_type, candidates

        if "pdf" in content_type or content[:4] == b"%PDF":
            file_path = base_path + ".pdf"
        else:
            file_path = base_path + ".bin"

        with open(file_path, "wb") as file:
            file.write(content)

        return file_path, content_type, candidates

    except Exception as error:
        print(f"Download failed for {doc_link}: {error}")
        return "", "", candidates


def get_pdf_metadata_candidates(file_path):
    candidates = []

    if not file_path or not file_path.lower().endswith(".pdf"):
        return candidates

    try:
        reader = PdfReader(file_path)
        metadata = reader.metadata

        if metadata:
            add_candidate(candidates, metadata.title, "pdf_metadata")
            add_candidate(candidates, metadata.subject, "pdf_metadata")

    except Exception as error:
        print(f"PDF metadata failed for {file_path}: {error}")

    return candidates


def get_pdf_first_page_candidates(file_path):
    candidates = []

    if not file_path or not file_path.lower().endswith(".pdf"):
        return candidates

    try:
        doc = fitz.open(file_path)

        if len(doc) == 0:
            return candidates

        page = doc[0]
        page_height = page.rect.height

        data = page.get_text("dict")

        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                parts = []
                sizes = []

                for span in line.get("spans", []):
                    text = span.get("text", "").strip()

                    if text:
                        parts.append(text)
                        sizes.append(span.get("size", 0))

                if not parts:
                    continue

                text = normalize_text(" ".join(parts))
                bbox = line.get("bbox")

                if not bbox:
                    continue

                y_pos = bbox[1]

                if y_pos > page_height * 0.5:
                    continue

                avg_size = sum(sizes) / len(sizes) if sizes else 0

                # Only trust PDF first-page text if it has title signals or large font
                if has_good_signal(text) or avg_size >= 16:
                    add_candidate(candidates, text, "pdf_first_page")

    except Exception as error:
        print(f"PDF first page failed for {file_path}: {error}")

    return candidates[:15]


def choose_best_candidate(candidates):
    valid = []

    seen = set()

    for candidate in candidates:
        title = normalize_text(candidate.get("title", ""))
        source = candidate.get("source", "unknown")
        score = candidate.get("score", title_quality_score(title, source))

        if is_bad_title(title) or score <= 0:
            continue

        key = title.lower()

        if key in seen:
            continue

        seen.add(key)

        valid.append({
            "title": title,
            "source": source,
            "score": score,
        })

    if not valid:
        return {
            "title": "",
            "source": "",
            "score": 0,
            "top_candidates": []
        }

    valid.sort(key=lambda x: x["score"], reverse=True)

    return {
        "title": valid[0]["title"],
        "source": valid[0]["source"],
        "score": valid[0]["score"],
        "top_candidates": valid[:5]
    }


def extract_title_for_row(page_url, doc_link):
    candidates = []

    # URL title
    url_title = clean_title_from_url(doc_link)
    add_candidate(candidates, url_title, "url")

    # HTML context title
    candidates.extend(get_html_candidates(page_url, doc_link))

    # Download + content-disposition + PDF title
    file_path, content_type, header_candidates = download_document(doc_link)
    candidates.extend(header_candidates)

    if file_path and file_path.lower().endswith(".pdf"):
        candidates.extend(get_pdf_metadata_candidates(file_path))
        candidates.extend(get_pdf_first_page_candidates(file_path))

    best = choose_best_candidate(candidates)

    return {
        "document_title": best["title"],
        "document_title_source": best["source"],
        "confidence_score": best["score"],
        "content_type": content_type,
        "candidate_count": len(candidates),
        "top_candidates": best["top_candidates"]
    }


def main():
    os.makedirs("data/output", exist_ok=True)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, dtype=str)

    required_columns = {"url", "doc_link"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in input file: {missing}")

    results = []
    total = len(df)

    for index, row in df.iterrows():
        page_url = normalize_text(row.get("url", ""))
        doc_link = normalize_text(row.get("doc_link", ""))

        print(f"\n[{index + 1}/{total}] Processing: {doc_link}")

        if not page_url or not doc_link:
            results.append({
                "url": page_url,
                "doc_link": doc_link,
                "document_title": "",
                "document_title_source": "",
                "confidence_score": 0,
                "content_type": "",
                "candidate_count": 0,
                "status": "missing_url_or_doc_link",
                "top_candidates": "[]"
            })
            continue

        try:
            result = extract_title_for_row(page_url, doc_link)

            print(
                f"Title: {result['document_title']} | "
                f"Source: {result['document_title_source']} | "
                f"Score: {result['confidence_score']}"
            )

            results.append({
                "url": page_url,
                "doc_link": doc_link,
                "document_title": result["document_title"],
                "document_title_source": result["document_title_source"],
                "confidence_score": result["confidence_score"],
                "content_type": result["content_type"],
                "candidate_count": result["candidate_count"],
                "status": "success",
                "top_candidates": result["top_candidates"]
            })

        except Exception as error:
            print(f"Failed: {error}")

            results.append({
                "url": page_url,
                "doc_link": doc_link,
                "document_title": "",
                "document_title_source": "",
                "confidence_score": 0,
                "content_type": "",
                "candidate_count": 0,
                "status": f"error: {error}",
                "top_candidates": "[]"
            })

    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

    print("\n✅ EXTRACTION COMPLETE")
    print(f"✅ Input rows: {total}")
    print(f"✅ Output saved to: {OUTPUT_FILE}")
    print(f"✅ Titles found: {(output_df['document_title'].astype(str).str.strip() != '').sum()}/{total}")


if __name__ == "__main__":
    main()
