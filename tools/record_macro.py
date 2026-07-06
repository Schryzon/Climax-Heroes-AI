import time
import sys

# Attempt to import pynput, install if missing
try:
    from pynput import keyboard
except ImportError:
    print("Installing pynput library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pynput"])
    from pynput import keyboard

print("=" * 60)
print("             KEYBOARD MACRO RECORDER")
print("=" * 60)
print("Instructions:")
print("1. Click on the emulator window to focus it.")
print("2. Perform your menu inputs (approx 4 taps).")
print("3. When finished, press the 'ESC' key to stop recording.")
print("=" * 60)
print("Recording starting in 3 seconds... Switch to emulator now!")
time.sleep(3.0)
print("\n[RECORDING STARTED] Press keys now...")

recorded_events = []
start_time = time.time()
last_time = start_time

def on_press(key):
    global last_time
    curr_time = time.time()
    delay = curr_time - last_time
    last_time = curr_time
    
    try:
        key_name = key.char
    except AttributeError:
        key_name = key.name
        
    recorded_events.append((key_name, delay))
    print(f"  Captured tap: '{key_name}' (after {delay:.2f} seconds)")
    
    if key_name == 'esc':
        # Stop listener
        return False

# Collect events until released
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()

print("\n" + "=" * 60)
print("             RECORDED MACRO SCRIPT")
print("=" * 60)
print("Add this sequence to your environment init/reset code:")
print("-" * 60)

for key, delay in recorded_events:
    if key == 'esc':
        continue
    # Format special keys to Xbox / controller layout mapping equivalents if needed
    print(f"time.sleep({delay:.2f})")
    print(f"self._press_key('{key}')  # maps to gamepad button")

print("=" * 60)
