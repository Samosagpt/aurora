import os
import streamlit.components.v1 as components
from typing import List, Dict, Any, Optional, Union

# Create a _RELEASE constant. We'll set this to False while we're developing
# the component, and True when we're ready to package and distribute it.
_RELEASE = True

if not _RELEASE:
    _component_func = components.declare_component(
        "streamlit_markdown_select",
        url="http://localhost:3001",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component(
        "streamlit_markdown_select", path=build_dir
    )

def markdown_select(
    options: List[Dict[str, Any]],
    key: Optional[str] = None,
    placeholder: str = "Select an option...",
    default_code_color: str = "#00cc88",
    max_height: str = "300px",
    disabled: bool = False,
    default_value: Optional[str] = None,
) -> Optional[str]:
    """
    Create a markdown select component with code blocks and buttons.
    
    Parameters
    ----------
    options : List[Dict[str, Any]]
        List of option dictionaries. Each option should have:
        - value (str): The value to return when selected
        - label (str): Display text for the option
        - code_block (Dict, optional): Code block configuration with:
            - code (str): The code to display
            - language (str, optional): Programming language
            - color (str, optional): Hex color for the code text
        - buttons (List[Dict], optional): List of button configurations with:
            - text (str): Button text
            - variant (str, optional): 'primary', 'secondary', or 'danger'
            - action (str): Action identifier for the button
        - disabled (bool, optional): Whether this option is disabled
    
    key : str, optional
        An optional key that uniquely identifies this component.
        
    placeholder : str, default "Select an option..."
        Placeholder text when no option is selected.
        
    default_code_color : str, default "#00cc88"
        Default color for code blocks (Streamlit green).
        
    max_height : str, default "300px"
        Maximum height of the dropdown.
        
    disabled : bool, default False
        Whether the entire component is disabled.
        
    default_value : str, optional
        Default selected value.
    
    Returns
    -------
    str or None
        The selected option value, or None if nothing is selected.
    
    Examples
    --------
    >>> options = [
    ...     {
    ...         "value": "python",
    ...         "label": "Python Script",
    ...         "code_block": {
    ...             "code": "print('Hello World')",
    ...             "language": "python",
    ...             "color": "#3776ab"
    ...         },
    ...         "buttons": [
    ...             {"text": "Run", "variant": "primary", "action": "run"},
    ...             {"text": "Edit", "variant": "secondary", "action": "edit"}
    ...         ]
    ...     },
    ...     {
    ...         "value": "javascript",
    ...         "label": "JavaScript Function",
    ...         "code_block": {
    ...             "code": "console.log('Hello World')",
    ...             "language": "javascript",
    ...             "color": "#f7df1e"
    ...         }
    ...     }
    ... ]
    >>> 
    >>> selected = markdown_select(options, key="my_select")
    >>> if selected:
    ...     st.write(f"You selected: {selected}")
    """
    
    component_value = _component_func(
        options=options,
        placeholder=placeholder,
        defaultCodeColor=default_code_color,
        maxHeight=max_height,
        disabled=disabled,
        defaultValue=default_value,
        key=key,
    )
    
    return component_value

# Predefined language colors for convenience
LANGUAGE_COLORS = {
    "javascript": "#f7df1e",
    "typescript": "#3178c6", 
    "python": "#3776ab",
    "java": "#ed8b00",
    "cpp": "#00599c",
    "csharp": "#239120",
    "php": "#777bb4",
    "ruby": "#cc342d",
    "go": "#00add8",
    "rust": "#000000",
    "swift": "#fa7343",
    "kotlin": "#7f52ff",
    "html": "#e34c26",
    "css": "#1572b6",
    "scss": "#cf649a",
    "json": "#000000",
    "xml": "#0060ac",
    "sql": "#336791",
    "bash": "#4eaa25",
    "powershell": "#012456",
    "markdown": "#083fa1",
    "yaml": "#cc1018",
    "dockerfile": "#384d54",
    "streamlit": "#00cc88",
}

def create_option(
    value: str,
    label: str,
    code: Optional[str] = None,
    language: Optional[str] = None,
    code_color: Optional[str] = None,
    buttons: Optional[List[Dict[str, str]]] = None,
    disabled: bool = False,
) -> Dict[str, Any]:
    """
    Helper function to create an option dictionary.
    
    Parameters
    ----------
    value : str
        The value to return when selected.
    label : str
        Display text for the option.
    code : str, optional
        Code to display in the code block.
    language : str, optional
        Programming language for syntax highlighting.
    code_color : str, optional
        Hex color for the code text. If not provided and language is specified,
        will use predefined color for that language.
    buttons : List[Dict[str, str]], optional
        List of button configurations.
    disabled : bool, default False
        Whether this option is disabled.
        
    Returns
    -------
    Dict[str, Any]
        Option dictionary ready for use with markdown_select.
    """
    option = {
        "value": value,
        "label": label,
        "disabled": disabled,
    }
    
    if code is not None:
        code_block = {"code": code}
        if language:
            code_block["language"] = language
        if code_color:
            code_block["color"] = code_color
        elif language and language in LANGUAGE_COLORS:
            code_block["color"] = LANGUAGE_COLORS[language]
        option["code_block"] = code_block
    
    if buttons:
        option["buttons"] = buttons
        
    return option