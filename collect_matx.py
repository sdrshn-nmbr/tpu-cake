# /// script
# dependencies = ["beautifulsoup4==4.15.0", "markdownify==1.2.2", "youtube-transcript-api==1.2.3"]
# ///

from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from markdownify import markdownify
from youtube_transcript_api import YouTubeTranscriptApi


ROOT = Path(__file__).parent / "matx"

PAGES = {
    "research-index": "https://matx.com/research",
    "matx-one": "https://matx.com/research/series_b",
    "future-leakage": "https://matx.com/research/leaky_quantization",
    "rules-derive": "https://matx.com/research/rules_derive",
    "speculative-decoding-nsa": "https://matx.com/research/sd_nsa",
    "spire": "https://matx.com/research/sd",
    "prioritize-values": "https://matx.com/research/smva",
    "lifetime-llm-cost": "https://matx.com/research/lifetime_llm_cost",
    "seqax": "https://matx.com/research/seqax",
}

TRANSCRIPTS = {
    "chip-design-bottom-up": "https://www.dwarkesh.com/p/reiner-pope-2",
    "llm-training-and-serving": "https://gist.githubusercontent.com/dwarkeshsp/79100f0fdeed69d76241903bb0604dbe/raw",
    "stripe-transformer-chips": "https://cheekypint.substack.com/p/reiner-pope-of-matx-on-accelerating",
}

PAPERS = {
    "spire-2504.06419": "https://arxiv.org/pdf/2504.06419",
}

CODE_SNAPSHOTS = {
    "seqax-main-b418a2d": "https://github.com/MatX-inc/seqax/archive/b418a2d9059a1bfcff801d22b7088cc444257703.zip",
    "seqax-nsa-e2c8151": "https://github.com/MatX-inc/seqax/archive/e2c8151532c12569385375f3177f212dc96fa7ac.zip",
    "seqax-smva-7fe01d0": "https://github.com/MatX-inc/seqax/archive/7fe01d0a443f5f15504653e180515f5112a9c281.zip",
    "seqax-spire-a0656fa": "https://github.com/MatX-inc/seqax/archive/a0656fa3cbdb952a877307dcd0a5ca71d5187006.zip",
}

VIDEOS = {
    "oIk3R-sMX5o": "chip-design-bottom-up",
    "xmkSf5IS-zw": "llm-training-and-serving",
    "qvrdCpLPbuQ": "stripe-transformer-chips",
    "gm3parYIMqA": "silicon-for-llms",
    "zrMYIhmuXEo": "hardware-software-codesign",
}

VIDEO_METADATA_FIELDS = (
    "id",
    "title",
    "description",
    "channel",
    "channel_url",
    "uploader",
    "upload_date",
    "timestamp",
    "duration",
    "duration_string",
    "webpage_url",
    "chapters",
)

def fetch(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "inkle-research-collector/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(destination)


def html_to_markdown(source: Path, destination: Path) -> None:
    soup = BeautifulSoup(source.read_bytes(), "html.parser")
    content = soup.find("article") or soup.find("main") or soup.body or soup
    for element in content.select("script, style, noscript, svg"):
        element.decompose()
    rendered = markdownify(str(content), heading_style="ATX", bullets="-")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered.strip() + "\n")


def collect_pages(records: list[dict[str, str]]) -> None:
    for name, url in PAGES.items():
        html_path = ROOT / "raw" / "pages" / f"{name}.html"
        text_path = ROOT / "text" / "pages" / f"{name}.md"
        fetch(url, html_path)
        html_to_markdown(html_path, text_path)
        records.extend(
            (
                {"kind": "page-html", "source": url, "path": str(html_path.relative_to(ROOT))},
                {"kind": "page-text", "source": url, "path": str(text_path.relative_to(ROOT))},
            )
        )


def collect_transcripts(records: list[dict[str, str]]) -> None:
    for name, url in TRANSCRIPTS.items():
        path = ROOT / "text" / "transcripts" / f"{name}.md"
        if "gist.githubusercontent.com" in url:
            fetch(url, path)
        else:
            raw_path = ROOT / "raw" / "transcripts" / f"{name}.html"
            fetch(url, raw_path)
            html_to_markdown(raw_path, path)
            records.append(
                {"kind": "transcript-html", "source": url, "path": str(raw_path.relative_to(ROOT))}
            )
        records.append({"kind": "official-transcript", "source": url, "path": str(path.relative_to(ROOT))})


def collect_papers(records: list[dict[str, str]]) -> None:
    for name, url in PAPERS.items():
        path = ROOT / "raw" / "papers" / f"{name}.pdf"
        fetch(url, path)
        records.append({"kind": "paper", "source": url, "path": str(path.relative_to(ROOT))})
        text_path = ROOT / "text" / "papers" / f"{name}.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["pdftotext", "-layout", str(path), str(text_path)], check=True)
        records.append({"kind": "paper-text", "source": url, "path": str(text_path.relative_to(ROOT))})


def collect_code(records: list[dict[str, str]]) -> None:
    inventory_url = "https://api.github.com/orgs/MatX-inc/repos?per_page=100&sort=full_name"
    inventory_path = ROOT / "raw" / "code" / "github-repositories.json"
    fetch(inventory_url, inventory_path)
    records.append(
        {"kind": "code-inventory", "source": inventory_url, "path": str(inventory_path.relative_to(ROOT))}
    )
    for name, url in CODE_SNAPSHOTS.items():
        path = ROOT / "raw" / "code" / f"{name}.zip"
        fetch(url, path)
        records.append({"kind": "code-snapshot", "source": url, "path": str(path.relative_to(ROOT))})


def yt_dlp_json(url: str) -> dict[str, Any]:
    result = subprocess.run(
        ["uvx", "--from", "yt-dlp", "yt-dlp", "--dump-single-json", "--skip-download", url],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def seconds_to_timestamp(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def collect_clean_transcript(video_id: str, destination: Path) -> None:
    transcript = YouTubeTranscriptApi().fetch(video_id, languages=("en", "en-US"))
    rendered = [f"[{seconds_to_timestamp(item.start)}] {item.text}" for item in transcript]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n\n".join(rendered) + "\n")


def collect_videos(records: list[dict[str, str]]) -> None:
    video_dir = ROOT / "raw" / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    for video_id, name in VIDEOS.items():
        url = f"https://www.youtube.com/watch?v={video_id}"
        metadata = yt_dlp_json(url)
        selected = {field: metadata.get(field) for field in VIDEO_METADATA_FIELDS}
        selected["subtitle_languages"] = sorted(metadata.get("subtitles", {}))
        selected["automatic_caption_languages"] = sorted(metadata.get("automatic_captions", {}))
        metadata_path = video_dir / f"{name}.metadata.json"
        metadata_path.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
        output_template = str(video_dir / f"{name}.%(language)s.%(ext)s")
        subprocess.run(
            [
                "uvx",
                "--from",
                "yt-dlp",
                "yt-dlp",
                "--skip-download",
                "--force-overwrites",
                "--write-subs",
                "--write-auto-subs",
                "--sub-langs",
                "en-orig,en",
                "--sub-format",
                "vtt",
                "--output",
                output_template,
                url,
            ],
            check=True,
        )
        records.append({"kind": "video-metadata", "source": url, "path": str(metadata_path.relative_to(ROOT))})
        for caption_path in sorted(video_dir.glob(f"{name}.*.vtt")):
            records.append(
                {"kind": "video-caption", "source": url, "path": str(caption_path.relative_to(ROOT))}
            )
        transcript_path = ROOT / "text" / "transcripts" / f"{name}-youtube.md"
        collect_clean_transcript(video_id, transcript_path)
        records.append(
            {"kind": "derived-transcript", "source": url, "path": str(transcript_path.relative_to(ROOT))}
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(records: list[dict[str, str]]) -> None:
    for record in records:
        path = ROOT / record["path"]
        record["bytes"] = str(path.stat().st_size)
        record["sha256"] = sha256(path)
    manifest = {"schema_version": 1, "records": sorted(records, key=lambda item: item["path"])}
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    records: list[dict[str, str]] = []
    collect_pages(records)
    collect_transcripts(records)
    collect_papers(records)
    collect_code(records)
    collect_videos(records)
    write_manifest(records)
    print(f"collected {len(records)} artifacts under {ROOT}")


if __name__ == "__main__":
    main()
