import subprocess
import os
import sys

RESPONSES = {
    'who are you': "I am Aurora, your virtual assistant. Built by Tejaji, Karthikeyan and Shyam",
    'are you fine': "I am doing fine and always at your service.",
    'help': "I can search Wikipedia, open websites, play music, and more!",
    'introduce yourself': "__LAUNCH_PRESENTATION__",  # Special marker for presentation
}


def get_pretrained_response(query):
    key = query.lower().strip()
    response = RESPONSES.get(key)
    
    # Handle presentation launch
    if response == "__LAUNCH_PRESENTATION__":
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            lol_path = os.path.join(script_dir, "lol.py")
            
            # Check if lol.py exists
            if not os.path.exists(lol_path):
                return " "
            
            # Launch the presentation script in a new process
            print(f"[Aurora] Launching presentation from: {lol_path}")
            subprocess.Popen([sys.executable, lol_path], 
                           cwd=script_dir,
                           creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0)
            
            return " "
        except Exception as e:
            return f"Error launching presentation: {str(e)}"
    
    return response