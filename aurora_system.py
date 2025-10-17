"""
AURORA System Configuration and Identity Manager
Handles the core identity, capabilities, and team information for AURORA
"""

import json
import os
from typing import Dict, List, Optional, Any

class AuroraSystem:
    """Manages AURORA's identity, capabilities, and team information"""
    
    def __init__(self, config_path: str = "aurora_config.json"):
        """Initialize AURORA system with configuration"""
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load AURORA configuration from JSON file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default AURORA configuration"""
        return {
            "project": {
                "name": "AURORA",
                "full_form": "Agentic Unified multi-model Reasoning Orchestrator for Rapid One-shot Assistance",
                "version": "1.0",
                "creators": ["Aurora contributors"]
            }
        }
    
    def get_identity(self) -> str:
        """Get AURORA's full identity string"""
        project = self.config.get("project", {})
        name = project.get("name", "AURORA")
        full_form = project.get("full_form", "")
        creators = ", ".join(project.get("creators", []))
        
        return f"{name} ({full_form}) - Created by {creators}"
    
    def get_system_prompt(self) -> str:
        """Get the complete system prompt for AURORA"""
        return self.config.get("project", {}).get("system_prompt", "")
    
    def get_capabilities(self) -> Dict[str, List[str]]:
        """Get AURORA's capabilities"""
        return self.config.get("project", {}).get("capabilities", {})
    
    def get_team_info(self) -> List[Dict[str, Any]]:
        """Get information about AURORA's development team"""
        return self.config.get("team", {}).get("core_members", [])
    
    def get_team_member(self, member_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific team member"""
        members = self.get_team_info()
        for member in members:
            if member.get("id") == member_id:
                return member
        return None
    
    def get_formatting_guidelines(self) -> Dict[str, Any]:
        """Get formatting guidelines for responses"""
        return self.config.get("project", {}).get("formatting_guidelines", {})
    
    def get_policies(self) -> Dict[str, str]:
        """Get AURORA's operational policies"""
        return self.config.get("project", {}).get("policies", {})
    
    def get_version(self) -> str:
        """Get AURORA version"""
        return self.config.get("project", {}).get("version", "1.0")
    
    def get_attribution(self) -> str:
        """Get attribution string"""
        return self.config.get("project", {}).get("attribution", "")
    
    def format_team_display(self) -> str:
        """Format team information for display"""
        team_members = self.get_team_info()
        output = "## AURORA Development Team\n\n"
        
        for member in team_members:
            output += f"### {member.get('name', 'Unknown')}\n"
            output += f"**Roles:** {', '.join(member.get('roles', []))}\n\n"
            output += "**Responsibilities:**\n"
            for resp in member.get('responsibilities', []):
                output += f"- {resp}\n"
            output += "\n**Skills:**\n"
            for skill in member.get('skills', []):
                output += f"- {skill}\n"
            output += "\n"
        
        return output
    
    def format_capabilities_display(self) -> str:
        """Format capabilities for display"""
        capabilities = self.get_capabilities()
        output = "## AURORA Capabilities\n\n"
        
        if "modalities" in capabilities:
            output += "**Modalities:**\n"
            for mod in capabilities["modalities"]:
                output += f"- {mod}\n"
            output += "\n"
        
        if "skills" in capabilities:
            output += "**Skills:**\n"
            for skill in capabilities["skills"]:
                output += f"- {skill.replace('_', ' ').title()}\n"
            output += "\n"
        
        return output
    
    def format_about_display(self) -> str:
        """Format complete about information for display"""
        project = self.config.get("project", {})
        
        output = f"# {project.get('name', 'AURORA')}\n\n"
        output += f"**{project.get('full_form', '')}**\n\n"
        output += f"Version: {project.get('version', '1.0')}\n\n"
        output += f"{project.get('attribution', '')}\n\n"
        output += "---\n\n"
        output += self.format_capabilities_display()
        output += "---\n\n"
        output += self.format_team_display()
        output += "---\n\n"
        
        # Add policies
        policies = self.get_policies()
        if policies:
            output += "## Operational Policies\n\n"
            for policy_name, policy_text in policies.items():
                output += f"**{policy_name.title()}:** {policy_text}\n\n"
        
        return output


# Global instance
_aurora_system = None

def get_aurora_system() -> AuroraSystem:
    """Get or create the global AURORA system instance"""
    global _aurora_system
    if _aurora_system is None:
        _aurora_system = AuroraSystem()
    return _aurora_system


if __name__ == "__main__":
    # Test the system
    aurora = AuroraSystem()
    print(aurora.get_identity())
    print("\n" + "="*60 + "\n")
    print(aurora.format_about_display())
