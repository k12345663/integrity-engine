"""End-to-end robustness on malformed / unusual inputs ("hidden edge cases").

Builds a throw-away corpus and runs the whole pipeline once. Needs the ML
models (downloaded on first use); AI detection is disabled to keep it quick.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "sample_submissions"


def _ffmpeg():
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    import pymupdf

    from integrity_engine.config import Config, project_root
    from integrity_engine.pipeline import IntegrityPipeline

    corpus = tmp_path_factory.mktemp("corpus")
    (corpus / "nested").mkdir()
    shutil.copy(SAMPLES / "sub_01_aarav.pdf", corpus / "original.pdf")
    shutil.copy(SAMPLES / "sub_01_aarav.pdf", corpus / "nested" / "DUPLICATE.PDF")  # same bytes, upper-case ext
    shutil.copy(SAMPLES / "sub_09_vikram.docx", corpus / "independent.docx")

    # image-only PDF (scanned document without a text layer)
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 50, 50), False)
    pix.clear_with(200)
    page.insert_image(page.rect, pixmap=pix)
    doc.save(corpus / "scanned.pdf")

    # password-protected PDF
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Secret text that must not be read.")
    doc.save(corpus / "locked.pdf", encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")

    # MP4 whose audio track exists but is pure silence
    subprocess.run(
        [
            _ffmpeg(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:r=10",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=16000:cl=mono",
            "-t",
            "5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(corpus / "muted.mp4"),
        ],
        check=True,
    )
    (corpus / "truncated.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 100)
    (corpus / "slides.pptx").write_bytes(b"not supported")
    (corpus / "notes.txt").write_text("Short plain text submission about scooters.\n\nSecond paragraph here.")

    cfg = Config()
    cfg.ai.enabled = False
    cfg.external.reference_dir = str(project_root() / "reference_corpus")
    out = tmp_path_factory.mktemp("reports")
    reports = IntegrityPipeline(cfg).run(str(corpus), str(out))
    return {r["document_id"]: r for r in reports}, out


def test_pipeline_completes_and_reports_every_file(run):
    reports, out = run
    assert len(reports) == 9
    assert (out / "summary.md").exists() and (out / "summary.json").exists()
    for doc_id in reports:
        assert (out / f"{doc_id.replace('/', '__')}.json").exists()


@pytest.mark.parametrize(
    "name, needle",
    [
        ("scanned.pdf", "no text layer"),
        ("locked.pdf", "password"),
        ("muted.mp4", "silent"),
        ("truncated.mp4", "corrupted"),
        ("slides.pptx", "Unsupported"),
    ],
)
def test_bad_files_are_flagged_with_reason(run, name, needle):
    reports, _ = run
    r = reports[name]
    assert r["status"] in ("flagged", "skipped")
    assert r["total_integrity_risk"] == "UNASSESSED"
    assert needle.lower() in " ".join(r["warnings"]).lower()


def test_identical_files_match_fully_but_never_themselves(run):
    reports, _ = run
    r = reports["original.pdf"]
    assert r["score_details"]["per_source_peer"].get("nested/DUPLICATE.PDF", 0) > 90
    assert "original.pdf" not in r["score_details"]["per_source_peer"]  # self-match excluded


def test_independent_document_stays_low(run):
    reports, _ = run
    assert reports["independent.docx"]["total_integrity_risk"] == "LOW"
