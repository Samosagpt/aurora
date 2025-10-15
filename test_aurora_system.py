"""
Test script for AURORA system configuration
"""

from aurora_system import AuroraSystem, get_aurora_system

def test_aurora_system():
    """Test AURORA system configuration loading and display"""
    
    print("="*80)
    print("AURORA SYSTEM CONFIGURATION TEST")
    print("="*80)
    print()
    
    # Initialize AURORA system
    aurora = AuroraSystem()
    
    # Test identity
    print("1. IDENTITY:")
    print("-" * 80)
    print(aurora.get_identity())
    print()
    
    # Test version
    print("2. VERSION:")
    print("-" * 80)
    print(f"Version: {aurora.get_version()}")
    print(f"Attribution: {aurora.get_attribution()}")
    print()
    
    # Test capabilities
    print("3. CAPABILITIES:")
    print("-" * 80)
    capabilities = aurora.get_capabilities()
    print(f"Modalities: {', '.join(capabilities.get('modalities', []))}")
    print(f"Skills: {', '.join(capabilities.get('skills', []))}")
    print()
    
    # Test team info
    print("4. TEAM INFORMATION:")
    print("-" * 80)
    team_members = aurora.get_team_info()
    for member in team_members:
        print(f"\n{member['name']}:")
        print(f"  Roles: {', '.join(member['roles'])}")
        print(f"  Skills: {', '.join(member['skills'])}")
    print()
    
    # Test policies
    print("5. OPERATIONAL POLICIES:")
    print("-" * 80)
    policies = aurora.get_policies()
    for policy_name, policy_text in policies.items():
        print(f"{policy_name.title()}: {policy_text}")
    print()
    
    # Test system prompt
    print("6. SYSTEM PROMPT (first 200 chars):")
    print("-" * 80)
    system_prompt = aurora.get_system_prompt()
    print(system_prompt[:200] + "...")
    print()
    
    # Test formatted displays
    print("7. FORMATTED DISPLAYS:")
    print("-" * 80)
    print("\n--- CAPABILITIES DISPLAY ---")
    print(aurora.format_capabilities_display())
    
    print("\n--- TEAM DISPLAY ---")
    print(aurora.format_team_display())
    
    print("\n--- FULL ABOUT DISPLAY (first 500 chars) ---")
    about = aurora.format_about_display()
    print(about[:500] + "...")
    print()
    
    # Test global instance
    print("8. GLOBAL INSTANCE TEST:")
    print("-" * 80)
    aurora_global = get_aurora_system()
    print(f"Global instance identity: {aurora_global.get_identity()}")
    print()
    
    print("="*80)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
    print("="*80)


if __name__ == "__main__":
    test_aurora_system()
