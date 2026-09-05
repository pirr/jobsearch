import pytest

from jobsearch.candidate.archive import archive_profile
from jobsearch.candidate.extractor import extract_profile
from jobsearch.candidate.factory import ReaderFactory
from jobsearch.candidate.profile import CandidateProfile, Constraint
from jobsearch.candidate.readers.base import ReaderError
from jobsearch.matching.llm.fake import FakeLLM


def test_factory_dispatches_by_extension(tmp_path):
    f = tmp_path / "cv.md"
    f.write_text("Jane Doe, Python engineer")
    assert "Python engineer" in ReaderFactory().read(f)


def test_factory_rejects_unsupported_format(tmp_path):
    f = tmp_path / "cv.odt"
    f.write_text("x")
    with pytest.raises(ReaderError, match="no reader"):
        ReaderFactory().read(f)


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(ReaderError, match="does not exist"):
        ReaderFactory().read(tmp_path / "nope.pdf")


def test_fabricated_evidence_is_demoted_to_soft():
    """The core protection for CV-inferred constraints.

    Constraints are inferred, so the model sometimes invents a quote. An
    unverifiable quote must never become a hard filter -- that would silently
    delete good matches.
    """
    cv = "Jane Doe. Senior Python Engineer in Berlin. I am looking for remote-only roles."
    llm = FakeLLM([{
        "name": "Jane Doe",
        "constraints": [
            {"kind": "remote", "value": "remote only", "stated": True,
             "evidence": "I am looking for remote-only roles"},
            {"kind": "visa", "value": "needs sponsorship", "stated": True,
             "evidence": "I require visa sponsorship"},  # never said
        ],
    }])
    profile = extract_profile(llm, cv)
    kinds = {c.kind: c for c in profile.constraints}
    assert kinds["remote"].is_hard is True
    assert kinds["visa"].is_hard is False
    assert kinds["visa"].evidence == ""


def test_stated_without_evidence_is_not_hard():
    c = Constraint(kind="remote", value="remote", stated=True, evidence="")
    assert c.is_hard is False


def test_profile_hash_tracks_content_not_filename():
    a = CandidateProfile(skills=["Python"], source_files=["a.pdf"])
    b = CandidateProfile(skills=["Python"], source_files=["b.pdf"])
    c = CandidateProfile(skills=["Python", "Go"], source_files=["a.pdf"])
    assert a.hash() == b.hash()   # same person, different filename
    assert a.hash() != c.hash()   # edited profile re-rates


def test_empty_documents_are_refused():
    with pytest.raises(ValueError):
        extract_profile(FakeLLM(), "   ")


def test_roundtrip_through_disk(tmp_path):
    p = CandidateProfile(skills=["Python"], constraints=[Constraint(kind="remote", value="yes")])
    path = tmp_path / "profile.json"
    p.save(path)
    assert CandidateProfile.load(path).hash() == p.hash()


def test_archive_profile_copies_sources_and_writes_json(tmp_path, profile):
    cv = tmp_path / "resume.pdf"
    cv.write_bytes(b"%PDF-1.4 fake")
    profile.source_files = [str(cv)]
    storage_root = tmp_path / "profile_storage"

    profile_id, dest_dir = archive_profile(storage_root, profile)

    assert dest_dir == storage_root / profile_id
    assert (dest_dir / "resume.pdf").read_bytes() == b"%PDF-1.4 fake"
    assert CandidateProfile.load(dest_dir / "profile.json").hash() == profile.hash()


def test_archive_profile_ids_never_collide(tmp_path, profile):
    cv = tmp_path / "resume.pdf"
    cv.write_text("x")
    profile.source_files = [str(cv)]
    storage_root = tmp_path / "profile_storage"

    id1, _ = archive_profile(storage_root, profile)
    id2, _ = archive_profile(storage_root, profile)
    assert id1 != id2


def test_archive_profile_with_no_source_files_falls_back_to_a_generic_slug(tmp_path, profile):
    profile.source_files = []
    profile_id, dest_dir = archive_profile(tmp_path / "profile_storage", profile)
    assert profile_id.endswith("-profile")
    assert (dest_dir / "profile.json").exists()
