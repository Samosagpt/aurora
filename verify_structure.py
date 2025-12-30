"""
Verification script to test the new package structure
"""

import sys


def test_import(module_path, description):
    """Test importing a module and print result"""
    try:
        parts = module_path.split(".")
        if len(parts) > 1:
            # Complex import
            module = __import__(module_path, fromlist=[parts[-1]])
        else:
            module = __import__(module_path)
        print(f"✅ {description}: {module_path}")
        return True
    except Exception as e:
        print(f"❌ {description}: {module_path} - {str(e)}")
        return False


def main():
    print("=" * 80)
    print("AURORA PACKAGE STRUCTURE VERIFICATION")
    print("=" * 80)
    print()

    tests = [
        # Core modules
        ("aurora.core.Generation", "Core: Generation"),
        ("aurora.core.logmanagement", "Core: Log Management"),
        ("aurora.core.config", "Core: Config"),
        ("aurora.core.aurora_system", "Core: Aurora System"),
        # Agents
        ("aurora.agents.agentic_handler", "Agents: Agentic Handler"),
        ("aurora.agents.vision_agent", "Agents: Vision Agent"),
        ("aurora.agents.desktop_agent", "Agents: Desktop Agent"),
        # Handlers
        ("aurora.handlers.attachment_handler", "Handlers: Attachment"),
        ("aurora.handlers.prompthandler", "Handlers: Prompt"),
        ("aurora.handlers.rag_handler", "Handlers: RAG"),
        ("aurora.handlers.error_handler", "Handlers: Error"),
        # Models
        ("aurora.models.image_model_manager", "Models: Image Manager"),
        ("aurora.models.video_model_manager", "Models: Video Manager"),
        # Utils
        ("aurora.utils.user_preferences", "Utils: User Preferences"),
        ("aurora.utils.hardware_optimizer", "Utils: Hardware Optimizer"),
        ("aurora.utils.ws", "Utils: Web Services"),
        # Audio
        ("aurora.audio.offline_sr_whisper", "Audio: Speech Recognition"),
        ("aurora.audio.offline_text2speech", "Audio: Text-to-Speech"),
        # UI
        ("aurora.ui.streamlit_app", "UI: Streamlit App"),
        ("aurora.ui.image_gen", "UI: Image Generation"),
        ("aurora.ui.video_gen", "UI: Video Generation"),
        # Security
        ("aurora.security.security", "Security: Security Manager"),
        ("aurora.security.health_check", "Security: Health Check"),
    ]

    passed = 0
    failed = 0

    for module_path, description in tests:
        if test_import(module_path, description):
            passed += 1
        else:
            failed += 1

    print()
    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 80)

    if failed == 0:
        print("🎉 All imports successful! Package structure is correct.")
        return 0
    else:
        print(f"⚠️  {failed} imports failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
