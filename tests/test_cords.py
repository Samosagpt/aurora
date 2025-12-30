"""
Coordinate Finder Tool
======================
Click anywhere on screen to get the exact pixel coordinates.
Press ESC to exit.

Usage:
1. Run this script
2. Open WhatsApp Web in your browser
3. Click on the search box - note the coordinates
4. Type a contact name in search
5. Click on the first contact result - note the coordinates
6. Click on the message input box - note the coordinates
7. Press ESC when done

The coordinates will be displayed with percentages relative to your screen size.
"""

import sys
import time

import pyautogui
from pynput import keyboard, mouse

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

print("=" * 70)
print("🎯 COORDINATE FINDER TOOL")
print("=" * 70)
print(f"\n🖥️  Screen Resolution: {screen_width} x {screen_height}")
print("\n📋 Instructions:")
print("   1. Open WhatsApp Web in your browser")
print("   2. Click on any element to see its coordinates")
print("   3. Note the coordinates for:")
print("      - Search box (top of left sidebar)")
print("      - First contact result (after typing in search)")
print("      - Message input box (bottom of chat)")
print("   4. Press ESC to exit")
print("\n" + "=" * 70)
print("\n⏳ Waiting for clicks... (Press ESC to exit)\n")

# Track if we should exit
exit_flag = False


def on_click(x, y, button, pressed):
    """Called when mouse is clicked"""
    if pressed and not exit_flag:
        # Calculate percentages
        x_percent = (x / screen_width) * 100
        y_percent = (y / screen_height) * 100

        # Calculate decimal for code
        x_decimal = x / screen_width
        y_decimal = y / screen_height

        print(f"\n🖱️  CLICK DETECTED:")
        print(f"   Absolute: ({x}, {y})")
        print(f"   Percent:  ({x_percent:.1f}%, {y_percent:.1f}%)")
        print(f"   For code: x = int(screen_width * {x_decimal:.3f})")
        print(f"             y = int(screen_height * {y_decimal:.3f})")
        print("-" * 70)


def on_press(key):
    """Called when keyboard key is pressed"""
    global exit_flag
    try:
        if key == keyboard.Key.esc:
            exit_flag = True
            print("\n\n✅ Exiting... Use these coordinates in vision_agent.py!")
            return False  # Stop listener
    except:
        pass


# Start mouse listener in non-blocking mode
mouse_listener = mouse.Listener(on_click=on_click)
mouse_listener.start()

# Start keyboard listener (blocking)
with keyboard.Listener(on_press=on_press) as kb_listener:
    kb_listener.join()

# Stop mouse listener
mouse_listener.stop()

print("\n" + "=" * 70)
print("📝 TO UPDATE COORDINATES IN CODE:")
print("=" * 70)
print(
    """
Edit vision_agent.py around line 451-460 and update:

search_x = int(screen_width * YOUR_X_VALUE)
search_y = int(screen_height * YOUR_Y_VALUE)

contact_x = int(screen_width * YOUR_X_VALUE)
contact_y = int(screen_height * YOUR_Y_VALUE)

input_x = int(screen_width * YOUR_X_VALUE)
input_y = int(screen_height * YOUR_Y_VALUE)
"""
)
print("=" * 70)
