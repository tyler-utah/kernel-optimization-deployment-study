"""C1: MLSys 2025 and ASPLOS 2025 paper census inputs.

mlsys    - parse the official MLSys 2025 proceedings index and abstract pages; download PDFs
asplos   - parse the official ASPLOS 2025 program; resolve DOIs/abstracts (Semantic Scholar,
           Crossref, OpenAlex); locate open full text (arXiv or open PDF)
text     - extract full text with PyMuPDF
dossiers - build coding dossiers (abstract, introduction/contributions, evaluation summary, links)
"""

from __future__ import annotations

import html as htmlmod
import json
import re
import sys
import time
import urllib.parse
import urllib.request

from bs4 import BeautifulSoup

from common import BATCHES, CACHE, DATA, atomic_json, read_json, write_jsonl

PAP = CACHE / "papers"
UA = {"User-Agent": "Mozilla/5.0 (research census; mailto:tsorensen@microsoft.com)"}


def fetch(url: str, binary: bool = False, tries: int = 4):
    last = None
    for i in range(tries):
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90)
            b = r.read()
            return b if binary else b.decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            last = e
            if "404" in str(e) or "403" in str(e):
                break
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"{url}: {last}")


def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


# ------------------------------------------------------------------ MLSys

def mlsys() -> None:
    base = "https://proceedings.mlsys.org"
    index = fetch(base + "/paper_files/paper/2025")
    soup = BeautifulSoup(index, "html.parser")
    papers = []
    for li in soup.find_all("li"):
        a = li.find("a", href=True)
        if not a or "/paper/2025/hash/" not in a["href"]:
            continue
        authors = li.find("i")
        papers.append({"title": a.get_text(strip=True), "url": base + a["href"],
                       "authors": authors.get_text(strip=True) if authors else ""})
    out = []
    for i, p in enumerate(papers):
        pid = f"mlsys25-{i + 1:03d}"
        cache = PAP / "meta" / f"{pid}.json"
        if cache.exists():
            out.append(read_json(cache))
            continue
        page = fetch(p["url"])
        s = BeautifulSoup(page, "html.parser")
        abstract = ""
        h4 = [h for h in s.find_all(["h2", "h3", "h4"]) if "abstract" in h.get_text().lower()]
        if h4:
            nxt = h4[0].find_next(["p", "div"])
            abstract = nxt.get_text(" ", strip=True) if nxt else ""
        links = {a.get_text(strip=True): urllib.parse.urljoin(base, a["href"]) for a in s.find_all("a", href=True)}
        pdf = next((v for k, v in links.items() if k.lower() == "paper" or v.endswith("-Paper-Conference.pdf")), "")
        supp = [v for k, v in links.items() if "supplemental" in k.lower() or "Supplemental" in v]
        rec = {"id": pid, "venue": "MLSys 2025", "title": p["title"], "authors": p["authors"],
               "url": p["url"], "abstract": abstract, "pdf_url": pdf, "supplemental": supp,
               "doi": "", "arxiv": "", "fulltext_source": "proceedings_pdf" if pdf else ""}
        atomic_json(cache, rec)
        out.append(rec)
        time.sleep(0.5)
    atomic_json(PAP / "mlsys2025-papers.json", out)
    print("mlsys papers", len(out), "with abstract", sum(bool(x["abstract"]) for x in out))


# ------------------------------------------------------------------ ASPLOS

def s2_bulk() -> list[dict]:
    cache = PAP / "s2-asplos-2025.json"
    if cache.exists():
        return read_json(cache)
    rows, token = [], None
    while True:
        q = {"venue": "ASPLOS", "year": "2024-2025",
             "fields": "title,externalIds,venue,abstract,openAccessPdf,publicationDate,authors"}
        if token:
            q["token"] = token
        d = json.loads(fetch("https://api.semanticscholar.org/graph/v1/paper/search/bulk?" + urllib.parse.urlencode(q)))
        rows += d.get("data", [])
        token = d.get("token")
        if not token:
            break
        time.sleep(1.5)
    atomic_json(cache, rows)
    return rows


def crossref_title(title: str) -> dict | None:
    q = {"query.bibliographic": title, "filter": "prefix:10.1145", "rows": 5,
         "select": "DOI,title,container-title,abstract,author,published"}
    d = json.loads(fetch("https://api.crossref.org/works?" + urllib.parse.urlencode(q)))
    for it in d["message"]["items"]:
        if norm((it.get("title") or [""])[0]) == norm(title):
            return it
    return None


def openalex_abstract(doi: str) -> str:
    try:
        d = json.loads(fetch("https://api.openalex.org/works/doi:" + doi))
    except Exception:  # noqa: BLE001
        return ""
    inv = d.get("abstract_inverted_index") or {}
    pos = sorted((i, w) for w, idxs in inv.items() for i in idxs)
    return " ".join(w for _, w in pos)


def arxiv_search(title: str) -> dict | None:
    t = re.sub(r"[^A-Za-z0-9 ]+", " ", title)
    words = [w for w in t.split() if len(w) > 2][:12]
    q = "ti:" + "+AND+ti:".join(urllib.parse.quote(w) for w in words)
    x = fetch(f"https://export.arxiv.org/api/query?search_query={q}&max_results=5")
    time.sleep(3.2)
    for entry in re.findall(r"<entry>(.*?)</entry>", x, re.S):
        et = htmlmod.unescape(re.sub(r"\s+", " ", re.search(r"<title>(.*?)</title>", entry, re.S).group(1)))
        if norm(et) == norm(title) or (len(norm(title)) > 25 and norm(title) in norm(et)) or (len(norm(et)) > 25 and norm(et) in norm(title)):
            aid = re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>", entry).group(1)
            pub = re.search(r"<published>([^<]+)</published>", entry).group(1)
            return {"arxiv": re.sub(r"v\d+$", "", aid), "arxiv_title": et, "arxiv_published": pub}
    return None


def asplos() -> None:
    # The live /asplos2025/ URL now serves the ASPLOS 2026 program; use the archived 2025 page.
    src = PAP / "asplos2025-program-wayback.html"
    if not src.exists():
        src.write_text(fetch("https://web.archive.org/web/20250426171456/https://www.asplos-conference.org/asplos2025/program/"), encoding="utf-8")
    html = src.read_text(encoding="utf-8")
    assert "ASPLOS 2025" in html[:5000], "wrong program page"
    soup = BeautifulSoup(html, "html.parser")
    prog = []
    for div in soup.select(".paper"):
        t = div.select_one(".paper-title")
        a = div.select_one(".paper-authors")
        if t:
            prog.append({"title": re.sub(r"\s+", " ", t.get_text(" ", strip=True)),
                         "authors": re.sub(r"\s+", " ", a.get_text(" ", strip=True)) if a else ""})
    seen, uniq = set(), []
    for p in prog:
        if norm(p["title"]) not in seen:
            seen.add(norm(p["title"]))
            uniq.append(p)
    s2 = {norm(x["title"]): x for x in s2_bulk()}
    out = []
    for i, p in enumerate(uniq):
        pid = f"asplos25-{i + 1:03d}"
        cache = PAP / "meta" / f"{pid}.json"
        if cache.exists():
            out.append(read_json(cache))
            continue
        rec = {"id": pid, "venue": "ASPLOS 2025", "title": p["title"], "authors": p["authors"],
               "url": "", "abstract": "", "pdf_url": "", "supplemental": [], "doi": "", "arxiv": "",
               "fulltext_source": ""}
        hit = s2.get(norm(p["title"]))
        if hit:
            rec["doi"] = (hit.get("externalIds") or {}).get("DOI", "")
            rec["arxiv"] = (hit.get("externalIds") or {}).get("ArXiv", "") or ""
            rec["abstract"] = hit.get("abstract") or ""
            rec["s2_paper_id"] = hit.get("paperId")
            rec["publication_date"] = hit.get("publicationDate")
        if not rec["doi"]:
            try:
                cr = crossref_title(p["title"])
            except Exception:  # noqa: BLE001
                cr = None
            if cr:
                rec["doi"] = cr["DOI"]
                rec["container"] = (cr.get("container-title") or [""])[0]
                if not rec["abstract"] and cr.get("abstract"):
                    rec["abstract"] = re.sub(r"<[^>]+>", " ", cr["abstract"]).strip()
        if rec["doi"] and not rec["abstract"]:
            rec["abstract"] = openalex_abstract(rec["doi"])
        if rec["doi"]:
            rec["url"] = "https://doi.org/" + rec["doi"]
        if not rec["arxiv"]:
            try:
                ax = arxiv_search(p["title"])
            except Exception:  # noqa: BLE001
                ax = None
            if ax:
                rec.update(ax)
        if rec["arxiv"]:
            rec["pdf_url"] = f"https://arxiv.org/pdf/{rec['arxiv']}"
            rec["fulltext_source"] = "arxiv"
        rec["volume"] = ("Vol1" if rec["doi"].startswith("10.1145/3669940") else
                         "Vol2" if rec["doi"].startswith("10.1145/3676641") else
                         "other:" + rec["doi"][:15] if rec["doi"] else "unresolved")
        atomic_json(cache, rec)
        out.append(rec)
        print(pid, rec["volume"], bool(rec["abstract"]), rec["arxiv"], p["title"][:60], flush=True)
    atomic_json(PAP / "asplos2025-papers.json", out)
    print("asplos program papers", len(out))


def download_and_extract(recs: list[dict]) -> None:
    import fitz  # PyMuPDF
    for r in recs:
        txt = PAP / "text" / f"{r['id']}.txt"
        if txt.exists() or not r.get("pdf_url"):
            continue
        pdf = PAP / "pdf" / f"{r['id']}.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        if not pdf.exists():
            try:
                pdf.write_bytes(fetch(r["pdf_url"], binary=True))
            except Exception as e:  # noqa: BLE001
                print("pdf fail", r["id"], e)
                continue
            time.sleep(1.0 if "arxiv" in r["pdf_url"] else 0.3)
        try:
            doc = fitz.open(pdf)
            text = "\n".join(page.get_text() for page in doc)
            links = sorted({l.get("uri") for page in doc for l in page.get_links() if l.get("uri")})
        except Exception as e:  # noqa: BLE001
            print("extract fail", r["id"], e)
            continue
        txt.parent.mkdir(parents=True, exist_ok=True)
        txt.write_text(text, encoding="utf-8")
        atomic_json(PAP / "links" / f"{r['id']}.json", links)
        print("text", r["id"], len(text), flush=True)


def all_records() -> list[dict]:
    asplos = [r for r in read_json(PAP / "asplos2025-papers.json", []) if r.get("in_official_2025_proceedings", True)]
    return read_json(PAP / "mlsys2025-papers.json", []) + asplos


def section(text: str, start_pats: list[str], stop_pats: list[str], limit: int) -> str:
    low = text
    start = None
    for pat in start_pats:
        m = re.search(pat, low, re.I | re.M)
        if m:
            start = m.start()
            break
    if start is None:
        return ""
    chunk = text[start:start + limit * 3]
    for pat in stop_pats:
        m = re.search(pat, chunk[200:], re.I | re.M)
        if m:
            chunk = chunk[:200 + m.start()]
            break
    return chunk[:limit]


def dossiers(per_batch: int = 12) -> None:
    items = []
    for r in all_records():
        txt_path = PAP / "text" / f"{r['id']}.txt"
        text = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""
        text = re.sub(r"-\n(?=[a-z])", "", text)
        text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
        intro = section(text, [r"^\s*1\.?\s*Introduction\s*$", r"^\s*I\.?\s*INTRODUCTION", r"Introduction\n"],
                        [r"^\s*2\.?\s+[A-Z][A-Za-z ]{3,40}$", r"^\s*II\.?\s+[A-Z]"], 7000)
        evalsec = section(text, [r"^\s*\d\.?\s*(?:Evaluation|Experiments?|Experimental (?:Setup|Evaluation|Results)|Results|Performance Evaluation)\s*$",
                                 r"^\s*[IVX]+\.?\s*(?:EVALUATION|EXPERIMENTS?)"],
                          [r"^\s*\d\.?\s+(?:Related Work|Discussion|Conclusion|Limitations)", r"^\s*[IVX]+\.?\s*(?:RELATED|CONCLUSION)"], 5000)
        concl = section(text, [r"^\s*\d+\.?\s*Conclusions?\s*$", r"^\s*[IVX]+\.?\s*CONCLUSIONS?"], [r"^\s*(?:References|Acknowledg)"], 1500)
        links = read_json(PAP / "links" / f"{r['id']}.json", []) or []
        url_text = sorted(set(re.findall(r"https?://(?:github\.com|gitlab\.com|zenodo\.org|doi\.org/10\.5281|huggingface\.co)/[^\s)\]}>,;]+", text)))
        badges = sorted(set(m.lower() for m in re.findall(r"artifacts? (?:available|evaluated|functional|reusable)|results? (?:reproduced|replicated)|artifact appendix|artifact evaluation", text, re.I)))
        prod = [m.group(0) for m in re.finditer(r"[^.\n]{0,160}\b(?:deployed|in production|production (?:cluster|system|environment|service|traffic)|serving (?:millions|billions))\b[^.\n]{0,160}", text, re.I)][:6]
        versions = [m.group(0) for m in re.finditer(r"[^.\n]{0,60}\b(?:vLLM|SGLang|TensorRT-LLM|FlashInfer|FlashAttention(?:-\d)?|PyTorch|CUTLASS|cuBLAS|Triton|DeepSpeed|Megatron|llama\.cpp|TVM)\b[^.\n]{0,20}\b(?:v?\d+\.\d+(?:\.\d+)?)[^.\n]{0,40}", text)][:10]
        hw = sorted(set(re.findall(r"\b(?:A100|H100|H200|H800|H20|B200|GB200|V100|A10G?|L4|L40S?|T4|RTX ?\d{4}|MI\d{3}X?|MI250X?|TPU ?v?\d\w*|Gaudi ?\d?|Jetson \w+|Apple M\d|Graviton\d?|Xeon|EPYC|Ascend ?\d+\w*)\b", text)))
        items.append({
            "id": r["id"], "venue": r["venue"], "title": r["title"], "authors": r["authors"],
            "url": r["url"] or r.get("pdf_url", ""), "doi": r.get("doi", ""), "arxiv": r.get("arxiv", ""),
            "fulltext_available": bool(text), "fulltext_source": r.get("fulltext_source", ""),
            "abstract": r.get("abstract", "")[:3000],
            "introduction_and_contributions": intro,
            "evaluation_excerpt": evalsec,
            "conclusion_excerpt": concl,
            "code_links": (url_text + [l for l in links if re.search(r"github|gitlab|zenodo|huggingface", l or "")])[:15],
            "artifact_badge_text": badges,
            "production_claim_snippets": prod,
            "baseline_version_snippets": versions,
            "hardware_mentions": hw[:25],
        })
    items.sort(key=lambda x: x["id"])
    atomic_json(DATA / "paper-dossiers.json", items)
    out = BATCHES / "c1_census"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(items), per_batch):
        write_jsonl(out / f"batch-{i // per_batch:03d}-input.jsonl", items[i:i + per_batch])
    print("dossiers", len(items), "fulltext", sum(x["fulltext_available"] for x in items),
          "batches", (len(items) + per_batch - 1) // per_batch)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "mlsys":
        mlsys()
    elif cmd == "asplos":
        asplos()
    elif cmd == "text":
        download_and_extract(all_records())
    elif cmd == "dossiers":
        dossiers()
