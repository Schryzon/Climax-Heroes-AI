import time
import sys
import subprocess

# Install pygame if missing
try:
    import pygame
except ImportError:
    print("Installing pygame library for joystick input recording...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame"])
    import pygame

pygame.init()
pygame.joystick.init()

# Detect joysticks
joystick_count = pygame.joystick.get_count()
if joystick_count == 0:
    print("=" * 60)
    print("ERROR: No gamepad, arcade stick, or controller detected!")
    print("Please make sure your stick is plugged in and recognized by Windows.")
    print("=" * 60)
    sys.exit(1)

# Initialize first joystick
joystick = pygame.joystick.Joystick(0)
joystick.init()

print("=" * 60)
print(f"Detected Controller: {joystick.get_name()}")
print("=" * 60)
print("Instructions:")
print("1. Perform your menu inputs on your stick (approx 4 inputs).")
print("2. Hold down button 0 (usually A or Cross) for 3 seconds to STOP recording.")
print("=" * 60)
print("Recording started! Press stick inputs now...")

recorded_events = []
start_time = time.time()
last_time = start_time

running = True
clock = pygame.time.Clock()

button_hold_start = None

while running:
    pygame.event.pump()
    curr_time = time.time()
    
    # 1. Check D-pad (Hats)
    for i in range(joystick.get_numhats()):
        hat_val = joystick.get_hat(i)
        if hat_val != (0, 0):
            # Format direction
            dx, dy = hat_val
            direction = ""
            if dx == -1: direction = "Left"
            elif dx == 1: direction = "Right"
            if dy == -1: direction = "Down"
            elif dy == 1: direction = "Up"
            
            delay = curr_time - last_time
            last_time = curr_time
            recorded_events.append((f"Dpad_{direction}", delay))
            print(f"  Captured: D-pad {direction} (after {delay:.2f}s)")
            # Wait for button release/neutral to prevent duplicate hits
            while joystick.get_hat(i) != (0, 0):
                pygame.event.pump()
                time.sleep(0.05)
                
    # 2. Check Buttons
    for i in range(joystick.get_numbuttons()):
        if joystick.get_button(i):
            delay = curr_time - last_time
            last_time = curr_time
            recorded_events.append((f"Button_{i}", delay))
            print(f"  Captured: Button {i} (after {delay:.2f}s)")
            
            # Check for hold-to-exit on button 0
            if i == 0:
                button_hold_start = time.time()
                while joystick.get_button(0):
                    pygame.event.pump()
                    if time.time() - button_hold_start > 3.0:
                        print("\n[STOPPING] Detected 3-second button hold. Stopping recording...")
                        running = False
                        break
                    time.sleep(0.05)
            else:
                while joystick.get_button(i):
                    pygame.event.pump()
                    time.sleep(0.05)
                    
    clock.tick(60)

# Remove the final button 0 event used for exiting
if recorded_events and recorded_events[-1][0] == "Button_0":
    recorded_events.pop()

print("\n" + "=" * 60)
print("             RECORDED JOYSTICK MACRO SCRIPT")
print("=" * 60)
print("Here is the exact sequence of your stick presses:")
print("-" * 60)

for event_name, delay in recorded_events:
    print(f"time.sleep({delay:.2f})")
    print(f"press_joystick('{event_name}')")

print("=" * 60)
pygame.quit()
