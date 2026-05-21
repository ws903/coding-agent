import subprocess

import pytest

from agent.git_ops import GitOps


@pytest.fixture
def git_repo(tmp_path):
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(tmp_path),
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(tmp_path),
        capture_output=True,
    )
    (tmp_path / "readme.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=str(tmp_path),
        capture_output=True,
    )
    return tmp_path


def test_is_repo(git_repo):
    ops = GitOps(git_repo)
    assert ops.is_repo() is True


def test_is_not_repo(tmp_path):
    ops = GitOps(tmp_path)
    assert ops.is_repo() is False


def test_snapshot_creates_commit(git_repo):
    ops = GitOps(git_repo)
    (git_repo / "new.txt").write_text("content")
    sha = ops.snapshot("test snapshot")
    assert sha is not None
    assert len(sha) == 40


def test_snapshot_no_changes_returns_head(git_repo):
    ops = GitOps(git_repo)
    sha = ops.snapshot("no changes")
    assert sha is not None
    assert len(sha) == 40


def test_snapshot_not_a_repo(tmp_path):
    ops = GitOps(tmp_path)
    assert ops.snapshot() is None


def test_rollback_restores_state(git_repo):
    ops = GitOps(git_repo)
    original_sha = ops.snapshot("before")
    (git_repo / "new.txt").write_text("added")
    ops.snapshot("after adding file")
    assert (git_repo / "new.txt").exists()
    ops.rollback(original_sha)
    assert not (git_repo / "new.txt").exists()


def test_rollback_not_a_repo(tmp_path):
    ops = GitOps(tmp_path)
    assert ops.rollback("abc123") is False


def test_rollback_invalid_sha(git_repo):
    ops = GitOps(git_repo)
    assert ops.rollback("0000000000000000000000000000000000000000") is False


def test_diff_since(git_repo):
    ops = GitOps(git_repo)
    sha = ops.snapshot("baseline")
    (git_repo / "changed.txt").write_text("hello")
    ops.snapshot("add file")
    diff = ops.diff_since(sha)
    assert "changed.txt" in diff


def test_current_branch(git_repo):
    ops = GitOps(git_repo)
    branch = ops.current_branch()
    assert branch in ("main", "master")
