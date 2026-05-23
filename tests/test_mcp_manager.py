import json

import pytest

from agent.extensions.mcp import (
    MCPManager,
    _parse_qualified_name,
    _qualified_name,
    load_mcp_config,
)


def test_load_mcp_config_returns_empty_when_missing(tmp_path):
    assert load_mcp_config(tmp_path) == {}


def test_load_mcp_config_reads_valid_json(tmp_path):
    config = {
        "mcpServers": {
            "fs": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            }
        }
    }
    (tmp_path / ".mcp.json").write_text(json.dumps(config))
    loaded = load_mcp_config(tmp_path)
    assert loaded == config


def test_load_mcp_config_returns_empty_on_invalid_json(tmp_path):
    (tmp_path / ".mcp.json").write_text("{not valid json")
    assert load_mcp_config(tmp_path) == {}


def test_qualified_name_roundtrip():
    name = _qualified_name("filesystem", "read_file")
    assert name == "mcp__filesystem__read_file"
    parsed = _parse_qualified_name(name)
    assert parsed == ("filesystem", "read_file")


def test_parse_qualified_name_rejects_non_mcp():
    assert _parse_qualified_name("read_file") is None


def test_parse_qualified_name_rejects_malformed():
    assert _parse_qualified_name("mcp__no_tool") is None
    assert _parse_qualified_name("mcp____") is None


def test_manager_with_no_config_does_nothing():
    manager = MCPManager({})
    assert manager.tools == []
    assert manager.connected_servers == []


def test_manager_owns_only_prefixed_names():
    manager = MCPManager({})
    assert manager.owns("mcp__fs__read")
    assert not manager.owns("read_file")


@pytest.mark.asyncio
async def test_manager_close_is_safe_when_never_connected():
    manager = MCPManager({})
    await manager.close()  # should not raise


@pytest.mark.asyncio
async def test_manager_call_with_unknown_server_returns_error():
    manager = MCPManager({})
    result = await manager.call("mcp__missing__tool", {})
    assert "not connected" in result.lower()


@pytest.mark.asyncio
async def test_manager_call_with_malformed_name_returns_error():
    manager = MCPManager({})
    result = await manager.call("mcp__bad", {})
    assert "malformed" in result.lower()
