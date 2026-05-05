# tests/test_cli.py


from agent.cli import build_orchestrator, parse_args


def test_parse_args_interactive():
    args = parse_args([])
    assert args.auto is False
    assert args.task is None


def test_parse_args_autonomous():
    args = parse_args(["--auto", "--task", "Fix the bug"])
    assert args.auto is True
    assert args.task == "Fix the bug"


def test_parse_args_max_steps():
    args = parse_args(["--auto", "--task", "Fix", "--max-steps", "10"])
    assert args.max_steps == 10


def test_parse_args_custom_model():
    args = parse_args(["--model", "phi4:14b"])
    assert args.model == "phi4:14b"


def test_parse_args_custom_base_url():
    args = parse_args(["--base-url", "http://localhost:5000/v1"])
    assert args.base_url == "http://localhost:5000/v1"


def test_parse_args_project_dir():
    args = parse_args(["--project", "/tmp/myproject"])
    assert args.project == "/tmp/myproject"


def test_parse_args_step_mode():
    args = parse_args(["--step"])
    assert args.step is True


def test_build_orchestrator(tmp_path):
    orch = build_orchestrator(
        project_root=tmp_path,
        base_url="http://localhost:11434/v1",
        model="qwen3:14b",
    )
    assert orch is not None
    assert orch.project_root == tmp_path
