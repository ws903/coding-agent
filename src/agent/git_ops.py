import subprocess
from pathlib import Path


class GitOps:
    def __init__(self, project_root: Path):
        self.root = project_root.resolve()

    def is_repo(self) -> bool:
        return (self.root / ".git").exists()

    def snapshot(self, message: str = "agent: snapshot") -> str | None:
        if not self.is_repo():
            return None
        self._run("git", "add", "-A")
        status = self._run("git", "status", "--porcelain")
        if not status.strip():
            return self._head_sha()
        self._run("git", "commit", "-m", message, "--allow-empty-message")
        return self._head_sha()

    def rollback(self, commit_sha: str) -> bool:
        if not self.is_repo():
            return False
        try:
            self._run("git", "reset", "--hard", commit_sha)
            return True
        except subprocess.CalledProcessError:
            return False

    def diff_since(self, commit_sha: str) -> str:
        try:
            return self._run("git", "diff", commit_sha, "--stat")
        except subprocess.CalledProcessError:
            return ""

    def current_branch(self) -> str:
        try:
            return self._run("git", "branch", "--show-current").strip()
        except subprocess.CalledProcessError:
            return ""

    def _head_sha(self) -> str:
        return self._run("git", "rev-parse", "HEAD").strip()

    def _run(self, *args: str) -> str:
        result = subprocess.run(
            args,
            cwd=str(self.root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, args, result.stdout, result.stderr
            )
        return result.stdout
