"""
User Preferences Manager for Aurora
Handles loading, saving, and managing user preferences across all components.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


class UserPreferencesManager:
    """Manages user preferences with automatic saving and loading"""

    def __init__(self, preferences_file: str = None):
        """Initialize preferences manager"""
        if preferences_file is None:
            base_dir = Path(__file__).parent.absolute()
            logs_dir = base_dir / "logs"
            logs_dir.mkdir(exist_ok=True)
            preferences_file = logs_dir / "user_preferences.json"

        self.preferences_file = preferences_file
        self.preferences = self._load_preferences()

    def _load_preferences(self) -> Dict[str, Any]:
        """Load preferences from file or create default"""
        try:
            if os.path.exists(self.preferences_file):
                with open(self.preferences_file, "r", encoding="utf-8") as f:
                    prefs = json.load(f)
                    # Migrate old preferences format if needed
                    return self._migrate_preferences(prefs)
            else:
                return self._get_default_preferences()
        except Exception as e:
            print(f"Error loading preferences: {e}")
            return self._get_default_preferences()

    def _get_default_preferences(self) -> Dict[str, Any]:
        """Get default preferences structure"""
        return {
            "chat": {
                "preferred_model": None,
                "preferred_streaming": True,
                "last_used_model": None,
                "enable_speech": False,
                "tts_engine": "pyttsx3",
            },
            "image_generation": {
                "preferred_num_images": 2,
                "preferred_resolution": [512, 512],
                "preferred_steps": 20,
                "preferred_guidance": 7.5,
                "last_used_model": None,
                "preferred_seed": None,
                "use_safety_checker": True,
            },
            "video_generation": {
                "preferred_frames": 16,
                "preferred_fps": 8,
                "preferred_resolution": [512, 512],
                "preferred_steps": 20,
                "preferred_guidance": 7.5,
                "last_used_model": None,
            },
            "general": {
                "save_preferences": True,
                "last_updated": time.time(),
                "auto_load_last_used": True,
                "theme": "auto",
            },
        }

    def _migrate_preferences(self, prefs: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate old preferences to new format"""
        default_prefs = self._get_default_preferences()

        # Merge with defaults to ensure all keys exist
        for category in default_prefs:
            if category not in prefs:
                prefs[category] = default_prefs[category]
            else:
                # Merge missing keys within category
                for key in default_prefs[category]:
                    if key not in prefs[category]:
                        prefs[category][key] = default_prefs[category][key]

        return prefs

    def save_preferences(self) -> bool:
        """Save preferences to file"""
        try:
            # Update timestamp
            self.preferences["general"]["last_updated"] = time.time()

            # Ensure directory exists
            os.makedirs(os.path.dirname(self.preferences_file), exist_ok=True)

            # Save to file
            with open(self.preferences_file, "w", encoding="utf-8") as f:
                json.dump(self.preferences, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"Error saving preferences: {e}")
            return False

    def get_preference(self, category: str, key: str, default: Any = None) -> Any:
        """Get a specific preference value"""
        try:
            return self.preferences.get(category, {}).get(key, default)
        except Exception:
            return default

    def set_preference(self, category: str, key: str, value: Any, auto_save: bool = True) -> bool:
        """Set a specific preference value"""
        try:
            if category not in self.preferences:
                self.preferences[category] = {}

            self.preferences[category][key] = value

            if auto_save:
                return self.save_preferences()
            return True
        except Exception as e:
            print(f"Error setting preference: {e}")
            return False

    def get_category_preferences(self, category: str) -> Dict[str, Any]:
        """Get all preferences for a category"""
        return self.preferences.get(category, {})

    def set_category_preferences(
        self, category: str, preferences: Dict[str, Any], auto_save: bool = True
    ) -> bool:
        """Set multiple preferences for a category"""
        try:
            if category not in self.preferences:
                self.preferences[category] = {}

            self.preferences[category].update(preferences)

            if auto_save:
                return self.save_preferences()
            return True
        except Exception as e:
            print(f"Error setting category preferences: {e}")
            return False

    def should_save_preferences(self) -> bool:
        """Check if preferences should be saved"""
        return self.get_preference("general", "save_preferences", True)

    def should_auto_load_last_used(self) -> bool:
        """Check if last used settings should be auto-loaded"""
        return self.get_preference("general", "auto_load_last_used", True)

    # Chat-specific methods
    def get_last_used_chat_model(self) -> Optional[str]:
        """Get the last used chat model"""
        return self.get_preference("chat", "last_used_model")

    def set_last_used_chat_model(self, model: str) -> bool:
        """Set the last used chat model"""
        return self.set_preference("chat", "last_used_model", model)

    def get_preferred_chat_model(self) -> Optional[str]:
        """Get the preferred chat model"""
        return self.get_preference("chat", "preferred_model")

    def set_preferred_chat_model(self, model: str) -> bool:
        """Set the preferred chat model"""
        return self.set_preference("chat", "preferred_model", model)

    def get_chat_streaming_preference(self) -> bool:
        """Get streaming preference for chat"""
        return self.get_preference("chat", "preferred_streaming", True)

    def set_chat_streaming_preference(self, enabled: bool) -> bool:
        """Set streaming preference for chat"""
        return self.set_preference("chat", "preferred_streaming", enabled)

    # Image generation methods
    def get_last_used_image_model(self) -> Optional[str]:
        """Get the last used image model"""
        return self.get_preference("image_generation", "last_used_model")

    def set_last_used_image_model(self, model: str) -> bool:
        """Set the last used image model"""
        return self.set_preference("image_generation", "last_used_model", model)

    def get_image_generation_defaults(self) -> Dict[str, Any]:
        """Get default image generation settings"""
        return {
            "num_images": self.get_preference("image_generation", "preferred_num_images", 2),
            "resolution": self.get_preference(
                "image_generation", "preferred_resolution", [512, 512]
            ),
            "steps": self.get_preference("image_generation", "preferred_steps", 20),
            "guidance": self.get_preference("image_generation", "preferred_guidance", 7.5),
            "seed": self.get_preference("image_generation", "preferred_seed"),
            "use_safety_checker": self.get_preference(
                "image_generation", "use_safety_checker", True
            ),
        }

    def save_image_generation_defaults(self, settings: Dict[str, Any]) -> bool:
        """Save image generation defaults"""
        preferences = {}
        if "num_images" in settings:
            preferences["preferred_num_images"] = settings["num_images"]
        if "resolution" in settings:
            preferences["preferred_resolution"] = settings["resolution"]
        if "steps" in settings:
            preferences["preferred_steps"] = settings["steps"]
        if "guidance" in settings:
            preferences["preferred_guidance"] = settings["guidance"]
        if "seed" in settings:
            preferences["preferred_seed"] = settings["seed"]
        if "use_safety_checker" in settings:
            preferences["use_safety_checker"] = settings["use_safety_checker"]

        return self.set_category_preferences("image_generation", preferences)

    # Video generation methods
    def get_last_used_video_model(self) -> Optional[str]:
        """Get the last used video model"""
        return self.get_preference("video_generation", "last_used_model")

    def set_last_used_video_model(self, model: str) -> bool:
        """Set the last used video model"""
        return self.set_preference("video_generation", "last_used_model", model)

    def get_video_generation_defaults(self) -> Dict[str, Any]:
        """Get default video generation settings"""
        return {
            "frames": self.get_preference("video_generation", "preferred_frames", 16),
            "fps": self.get_preference("video_generation", "preferred_fps", 8),
            "resolution": self.get_preference(
                "video_generation", "preferred_resolution", [512, 512]
            ),
            "steps": self.get_preference("video_generation", "preferred_steps", 20),
            "guidance": self.get_preference("video_generation", "preferred_guidance", 7.5),
        }

    def save_video_generation_defaults(self, settings: Dict[str, Any]) -> bool:
        """Save video generation defaults"""
        preferences = {}
        if "frames" in settings:
            preferences["preferred_frames"] = settings["frames"]
        if "fps" in settings:
            preferences["preferred_fps"] = settings["fps"]
        if "resolution" in settings:
            preferences["preferred_resolution"] = settings["resolution"]
        if "steps" in settings:
            preferences["preferred_steps"] = settings["steps"]
        if "guidance" in settings:
            preferences["preferred_guidance"] = settings["guidance"]

        return self.set_category_preferences("video_generation", preferences)

    def reset_preferences(self, category: str = None) -> bool:
        """Reset preferences to defaults"""
        try:
            if category:
                # Reset specific category
                default_prefs = self._get_default_preferences()
                if category in default_prefs:
                    self.preferences[category] = default_prefs[category]
            else:
                # Reset all preferences
                self.preferences = self._get_default_preferences()

            return self.save_preferences()
        except Exception as e:
            print(f"Error resetting preferences: {e}")
            return False

    def export_preferences(self) -> str:
        """Export preferences as JSON string"""
        try:
            return json.dumps(self.preferences, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error exporting preferences: {e}")
            return "{}"

    def import_preferences(self, json_str: str) -> bool:
        """Import preferences from JSON string"""
        try:
            imported_prefs = json.loads(json_str)
            # Migrate and validate
            self.preferences = self._migrate_preferences(imported_prefs)
            return self.save_preferences()
        except Exception as e:
            print(f"Error importing preferences: {e}")
            return False

    def get_all_preferences(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all preferences for all categories

        Returns:
            Dict containing all preferences organized by category
        """
        return dict(self.preferences)

    def _save_to_string(self) -> str:
        """
        Save preferences to JSON string for export

        Returns:
            JSON string representation of preferences
        """
        prefs_copy = dict(self.preferences)
        prefs_copy["general"]["last_updated"] = time.time()
        return json.dumps(prefs_copy, indent=2, ensure_ascii=False)


# Global instance
_preferences_manager = None


def get_preferences_manager() -> UserPreferencesManager:
    """Get the global preferences manager instance"""
    global _preferences_manager
    if _preferences_manager is None:
        _preferences_manager = UserPreferencesManager()
    return _preferences_manager


# Convenience functions
def get_preference(category: str, key: str, default: Any = None) -> Any:
    """Get a preference value"""
    return get_preferences_manager().get_preference(category, key, default)


def set_preference(category: str, key: str, value: Any) -> bool:
    """Set a preference value"""
    return get_preferences_manager().set_preference(category, key, value)


def save_preferences() -> bool:
    """Save all preferences"""
    return get_preferences_manager().save_preferences()
