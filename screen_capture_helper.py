import pygetwindow as gw
import mss
import cv2
import numpy as np
import time

def find_game_window(window_title_keyword="仮面ライダー"):
    print(f"Searching for windows containing '{window_title_keyword}'...")
    windows = gw.getWindowsWithTitle(window_title_keyword)
    if not windows:
        # Try lowercase check
        all_titles = gw.getAllTitles()
        windows = [gw.getWindowsWithTitle(t)[0] for t in all_titles if window_title_keyword.lower() in t.lower()]
        
    if windows:
        win = windows[0]
        print(f"Found window: '{win.title}'")
        print(f"Position: Left={win.left}, Top={win.top}, Width={win.width}, Height={win.height}")
        return win
    else:
        print(f"No window found matching keyword '{window_title_keyword}'.")
        print("Available window titles:")
        for t in gw.getAllTitles()[:20]: # show first 20
            if t: print(f" - {t}")
        return None

def test_capture(window_title_keyword="仮面ライダー", output_filename="game_capture_test.png"):
    win = find_game_window(window_title_keyword)
    if not win:
        print("Could not capture: Game window not found. Make sure the emulator is running and visible.")
        return False
        
    # Bring window to focus if possible
    try:
        win.activate()
        time.sleep(0.5)
    except Exception as e:
        print(f"Could not focus window automatically (non-fatal): {e}")

    # Capture the exact window region
    region = {
        "top": win.top,
        "left": win.left,
        "width": win.width,
        "height": win.height
    }
    
    print(f"Capturing region: {region}...")
    with mss.mss() as sct:
        img = np.array(sct.grab(region))
        
        # Save screenshot
        cv2.imwrite(output_filename, img)
        print(f"Successfully captured and saved to '{output_filename}'!")
        print(f"Image shape: {img.shape}")
        
        # Draw placeholder regions where health bars are expected
        # Typically HP bars are in the top ~15% of the screen
        h, w, _ = img.shape
        cv2.rectangle(img, (int(w*0.05), int(h*0.05)), (int(w*0.45), int(h*0.12)), (0, 255, 0), 2) # P1 HP
        cv2.rectangle(img, (int(w*0.55), int(h*0.05)), (int(w*0.95), int(h*0.12)), (0, 0, 255), 2) # P2 HP
        
        annotated_filename = "game_capture_annotated.png"
        cv2.imwrite(annotated_filename, img)
        print(f"Saved annotated preview with HP bounding boxes to '{annotated_filename}'!")
        return True

if __name__ == "__main__":
    # Install pygetwindow if missing
    try:
        import pygetwindow
    except ImportError:
        print("Installing pygetwindow library for window detection...")
        import subprocess
        subprocess.run(["pip", "install", "pygetwindow", "pyrect"])
        
    # Run test
    keywords = ["仮面ライダー", "PCSX2", "Dolphin", "Climax Heroes"]
    success = False
    for kw in keywords:
        print(f"Trying to find window with keyword: '{kw}'...")
        if test_capture(kw):
            success = True
            break
    if not success:
        print("Failed to find any game window.")

