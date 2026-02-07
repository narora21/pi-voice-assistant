import pytest

from src.tools.base import ToolDefinition, ToolParameter
from src.tools.builtin.device_control import DeviceControlTool
from src.tools.registry import ToolRegistry


@pytest.fixture
def registry():
    r = ToolRegistry()
    r.register(DeviceControlTool())
    return r


def test_register_and_get(registry):
    tool = registry.get("device_control")
    assert tool is not None
    assert tool.definition.name == "device_control"


def test_get_missing():
    r = ToolRegistry()
    assert r.get("nonexistent") is None


def test_duplicate_register():
    r = ToolRegistry()
    r.register(DeviceControlTool())
    with pytest.raises(ValueError, match="already registered"):
        r.register(DeviceControlTool())


def test_list_tools(registry):
    tools = registry.list_tools()
    assert len(tools) == 1
    assert tools[0].name == "device_control"


def test_to_ollama_tools(registry):
    ollama_tools = registry.to_ollama_tools()
    assert len(ollama_tools) == 1

    func = ollama_tools[0]
    assert func["type"] == "function"
    assert func["function"]["name"] == "device_control"
    assert "properties" in func["function"]["parameters"]
    assert "device_name" in func["function"]["parameters"]["properties"]
    assert "action" in func["function"]["parameters"]["properties"]
    assert "device_name" in func["function"]["parameters"]["required"]
    assert "action" in func["function"]["parameters"]["required"]

    # Check enum conversion
    action_prop = func["function"]["parameters"]["properties"]["action"]
    assert action_prop["enum"] == ["on", "off"]


@pytest.mark.asyncio
async def test_call_tool_success(registry):
    result = await registry.call_tool("device_control", {
        "device_name": "living_room_light",
        "action": "on",
    })
    assert "OK" in result
    assert "living_room_light" in result


@pytest.mark.asyncio
async def test_call_tool_unknown(registry):
    result = await registry.call_tool("nonexistent", {})
    assert "Error" in result
    assert "Unknown tool" in result


@pytest.mark.asyncio
async def test_call_tool_invalid_action(registry):
    result = await registry.call_tool("device_control", {
        "device_name": "lamp",
        "action": "explode",
    })
    assert "Error" in result
    assert "Invalid action" in result


@pytest.mark.asyncio
async def test_call_tool_missing_device(registry):
    result = await registry.call_tool("device_control", {"action": "on"})
    assert "Error" in result
