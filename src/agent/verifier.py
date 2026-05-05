from agent.models import VerificationResult
from agent.sandbox import Sandbox


class Verifier:
    def __init__(self, sandbox: Sandbox, commands: list[str] | None = None):
        self.sandbox = sandbox
        self.commands = commands or []

    def run(self, step_command: str | None = None) -> VerificationResult:
        all_commands = list(self.commands)
        if step_command:
            all_commands.append(step_command)

        if not all_commands:
            return VerificationResult(passed=True, details=[])

        details = []
        for cmd in all_commands:
            result = self.sandbox.run_command(cmd)
            details.append(result)

        passed = all(r.exit_code == 0 for r in details)
        return VerificationResult(passed=passed, details=details)
