import os
import streamlit.components.v1 as components
from typing import List, Dict, Any, Optional

# Create a _RELEASE constant. We'll set this to False while we're developing
# the component, and True when we're ready to package and distribute it.
_RELEASE = True

if not _RELEASE:
    _component_func = components.declare_component(
        "streamlit_navbar",
        url="http://localhost:3002",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component(
        "streamlit_navbar", path=build_dir
    )

def navbar(
    items: List[Dict[str, Any]],
    key: Optional[str] = None,
    logo_text: str = "🌅 Aurora",
    selected: Optional[str] = None,
    sticky: bool = True,
    style: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """
    Create a navigation bar component.
    
    Parameters
    ----------
    items : List[Dict[str, Any]]
        List of navigation items. Each item should have:
        - id (str): Unique identifier for the item
        - label (str): Display text for the item
        - icon (str, optional): Icon/emoji for the item
        - disabled (bool, optional): Whether this item is disabled
    
    key : str, optional
        An optional key that uniquely identifies this component.
        
    logo_text : str, default "🌅 Aurora"
        Text to display as logo/brand.
        
    selected : str, optional
        Currently selected item id.
        
    sticky : bool, default True
        Whether the navbar should stick to the top.
        
    style : Dict[str, str], optional
        Custom CSS styles for the navbar.
    
    Returns
    -------
    str or None
        The selected item id, or None if nothing is selected.
    """
    
    default_style = {
        "background": "#0E1117",
        "color": "white",
        "padding": "0.75rem 1rem",
        "border-bottom": "1px solid rgba(255, 255, 255, 0.1)",
        "box-shadow": "0 1px 3px rgba(0,0,0,0.3), 0 1px 2px rgba(0,0,0,0.5)",
        "z-index": "1000"
    }
    
    if style:
        default_style.update(style)
    
    component_value = _component_func(
        items=items,
        logoText=logo_text,
        selected=selected,
        sticky=sticky,
        style=default_style,
        key=key,
        height=80,
    )
    
    return component_value

def create_nav_item(
    id: str,
    label: str,
    icon: Optional[str] = None,
    disabled: bool = False,
) -> Dict[str, Any]:
    """
    Helper function to create a navigation item dictionary.
    
    Parameters
    ----------
    id : str
        Unique identifier for the item.
    label : str
        Display text for the item.
    icon : str, optional
        Icon/emoji for the item.
    disabled : bool, default False
        Whether this item is disabled.
        
    Returns
    -------
    Dict[str, Any]
        Navigation item dictionary ready for use with navbar.
    """
    item = {
        "id": id,
        "label": label,
        "disabled": disabled,
    }
    
    if icon:
        item["icon"] = icon
        
    return item
