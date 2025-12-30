"""
Agentic Handler - Interprets user requests and executes desktop agent tools
Integrates with Ollama LLM to understand natural language commands and translate them to actions
"""

import json
import logging
from typing import Any, Dict, List, Optional

from aurora.agents.desktop_agent import desktop_agent
from aurora.core.Generation import ollama_manager

logger = logging.getLogger(__name__)


class AgenticHandler:
    """Handler for agentic AI operations with tool calling"""

    def __init__(self):
        """Initialize agentic handler"""
        self.desktop_agent = desktop_agent
        self.tool_registry = self._build_tool_registry()
        logger.info("Agentic Handler initialized with desktop control capabilities")

    def _build_tool_registry(self) -> Dict[str, Any]:
        """Build registry of available tools"""
        registry = {}

        # Register all desktop agent methods
        tools = [
            # Mouse operations
            "mouse_click",
            "mouse_move",
            "mouse_drag",
            "mouse_scroll",
            # Keyboard operations
            "type_text",
            "press_key",
            # Screen operations
            "take_screenshot",
            "locate_on_screen",
            # OCR & GUI Recognition operations
            "ocr_screen",
            "find_text_on_screen",
            "click_text",
            "recognize_gui_elements",
            # Application operations
            "open_application",
            "close_application",
            "list_windows",
            "switch_window",
            "get_window_info",
            # File operations
            "read_file",
            "write_file",
            "list_directory",
            "search_files",
            # System operations
            "run_command",
            "get_system_info",
            # Utility operations
            "wait",
        ]

        for tool_name in tools:
            if hasattr(self.desktop_agent, tool_name):
                registry[tool_name] = getattr(self.desktop_agent, tool_name)

        logger.info(f"Registered {len(registry)} tools")
        return registry

    def get_system_prompt(self) -> str:
        """Get system prompt for agentic AI"""
        tools_description = self._format_tools_for_prompt()

        system_prompt = f"""You are an advanced AI assistant with desktop control capabilities. You can help users automate tasks on their Windows computer.

Available Tools:
{tools_description}

When the user asks you to perform a task:
1. Analyze the request and determine which tools are needed
2. Plan the sequence of actions
3. Execute the tools step by step
4. Report the results clearly

Tool Response Format:
To use a tool, respond with a JSON object:
{{
    "thought": "Your reasoning about what to do",
    "tool": "tool_name",
    "parameters": {{
        "param1": "value1",
        "param2": "value2"
    }}
}}

For multi-step tasks, explain your plan first, then execute each step.

Safety Guidelines:
- Always confirm destructive actions (delete, close, etc.)
- Be careful with system commands
- Respect file permissions
- Don't execute potentially harmful commands

Example interactions:
User: "Open Notepad and type Hello World"
Assistant: I'll help you with that. Let me:
1. First, open Notepad
2. Then type "Hello World"

{{
    "thought": "Opening Notepad application",
    "tool": "open_application",
    "parameters": {{"app_name": "notepad"}}
}}

Remember: Always be helpful, safe, and explain what you're doing!
"""
        return system_prompt

    def _format_tools_for_prompt(self) -> str:
        """Format tool descriptions for the prompt"""
        tools = self.desktop_agent.get_available_tools()

        formatted = []
        for tool in tools:
            params = ", ".join([f"{k}: {v}" for k, v in tool.get("parameters", {}).items()])
            formatted.append(f"- {tool['name']}({params}): {tool['description']}")

        return "\n".join(formatted)

    def process_user_request(self, user_input: str, model: str = None) -> str:
        """Process user request with agentic capabilities"""
        print(f"🔍 DEBUG [agentic_handler]: process_user_request called with: '{user_input}'")
        try:
            # Get system prompt
            system_prompt = self.get_system_prompt()
            print(
                f"🔍 DEBUG [agentic_handler]: System prompt generated ({len(system_prompt)} chars)"
            )

            # Create context with system prompt
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ]

            # Get AI response
            print(f"🔍 DEBUG [agentic_handler]: Calling Ollama with model: {model or 'llama3.2'}")
            response = ollama_manager.client.chat(model=model or "llama3.2", messages=messages)
            print(f"🔍 DEBUG [agentic_handler]: Got Ollama response")

            ai_response = response["message"]["content"]

            # Check if response contains tool call
            tool_call = self._extract_tool_call(ai_response)

            if tool_call:
                # Execute tool
                result = self._execute_tool(tool_call)

                # Get AI to interpret results
                follow_up = f"Tool execution result: {json.dumps(result)}\n\nPlease summarize this for the user."
                messages.append({"role": "assistant", "content": ai_response})
                messages.append({"role": "user", "content": follow_up})

                final_response = ollama_manager.client.chat(
                    model=model or "llama3.2", messages=messages
                )

                return final_response["message"]["content"]
            else:
                # No tool call, return AI response
                return ai_response

        except Exception as e:
            logger.error(f"Error processing agentic request: {e}")
            return f"Error: {str(e)}"

    def _extract_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract tool call from AI response"""
        try:
            # Look for JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1

            if start != -1 and end > start:
                json_str = response[start:end]
                tool_call = json.loads(json_str)

                # Validate tool call structure
                if "tool" in tool_call and "parameters" in tool_call:
                    return tool_call

            return None
        except json.JSONDecodeError:
            return None
        except Exception as e:
            logger.error(f"Error extracting tool call: {e}")
            return None

    def _execute_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call"""
        try:
            tool_name = tool_call["tool"]
            parameters = tool_call.get("parameters", {})

            if tool_name not in self.tool_registry:
                return {"success": False, "error": f"Tool '{tool_name}' not found"}

            # Execute tool
            tool_func = self.tool_registry[tool_name]
            result = tool_func(**parameters)

            logger.info(f"Executed tool '{tool_name}' with result: {result.get('success')}")
            return result

        except Exception as e:
            logger.error(f"Error executing tool: {e}")
            return {"success": False, "error": str(e)}

    def execute_workflow(self, workflow: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a workflow of multiple tool calls"""
        results = []

        for step in workflow:
            result = self._execute_tool(step)
            results.append({"step": step, "result": result})

            # Stop on failure unless configured to continue
            if not result.get("success") and not step.get("continue_on_error"):
                break

        return results

    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool"""
        tools = self.desktop_agent.get_available_tools()
        for tool in tools:
            if tool["name"] == tool_name:
                return tool
        return None

    def list_available_tools(self) -> List[str]:
        """List all available tool names"""
        return list(self.tool_registry.keys())


# Global instance
agentic_handler = AgenticHandler()


# Helper functions for integration
def is_desktop_control_request(user_input: str) -> bool:
    """Check if user input is requesting desktop control"""
    control_keywords = [
        "open",
        "close",
        "click",
        "type",
        "press",
        "screenshot",
        "window",
        "application",
        "file",
        "folder",
        "directory",
        "command",
        "run",
        "execute",
        "mouse",
        "keyboard",
        "move",
        "search",
        "find",
        "list",
        "show",
        "read",
        "write",
        "create",
    ]

    user_input_lower = user_input.lower()
    return any(keyword in user_input_lower for keyword in control_keywords)


def handle_agentic_request(user_input: str, model: str = None) -> str:
    """Main entry point for agentic requests"""
    print(
        f"🔍 DEBUG [agentic_handler]: handle_agentic_request called with input: '{user_input}', model: '{model}'"
    )
    logger.info(f"Handling agentic request: {user_input}")
    result = agentic_handler.process_user_request(user_input, model)
    print(f"🔍 DEBUG [agentic_handler]: Result: {result[:200] if result else 'None'}...")
    return result
