import time
import sys

print("Checking if vgamepad is importable...")
try:
    import vgamepad as vg
    print("vgamepad imported successfully!")
except ImportError:
    print("Error: vgamepad library is not installed in this Python environment.")
    sys.exit(1)

print("Initializing virtual Xbox 360 gamepad...")
try:
    gamepad = vg.VX360Gamepad()
    print("Virtual gamepad successfully created!")
    print("Simulating pressing 'A' button for 1 second...")
    gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    gamepad.update()
    time.sleep(1.0)
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    gamepad.update()
    print("Released button. Gamepad test succeeded!")
except Exception as e:
    print(f"Error creating virtual gamepad device: {e}")
    print("\nTroubleshooting:")
    print("1. Ensure you have the ViGEmBus driver installed on your machine.")
    print("2. You can download and install it manually from: https://github.com/ViGEm/ViGEmBus/releases")
    sys.exit(1)
