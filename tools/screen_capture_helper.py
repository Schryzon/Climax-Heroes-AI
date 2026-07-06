import ctypes
# Make Python DPI-aware on Windows to prevent display scaling from cropping screen capture
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2) # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

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
        h, w, _ = img.shape
        # Top HP Bars
        p1_x1, p1_x2 = int(w * 0.220), int(w * 0.445)
        p2_x1, p2_x2 = int(w * 0.555), int(w * 0.780)
        y1, y2 = int(h * 0.087), int(h * 0.113)
        
        # Bottom Rider Gauges
        p1_rg_x1, p1_rg_x2 = int(w * 0.220), int(w * 0.380)
        p2_rg_x1, p2_rg_x2 = int(w * 0.620), int(w * 0.780)
        rg_y1, rg_y2 = int(h * 0.884), int(h * 0.921)
        
        # Guard Gauges (below HP bars, closer to the center)
        p1_gg_x1, p1_gg_x2 = int(w * 0.220), int(w * 0.336)
        p2_gg_x1, p2_gg_x2 = int(w * 0.664), int(w * 0.780)
        gg_y1, gg_y2 = int(h * 0.120), int(h * 0.138)
        
        # Calculate HP estimates
        p1_crop = img[y1:y2, p1_x1:p1_x2]
        p2_crop = img[y1:y2, p2_x1:p2_x2]
        
        def estimate_hp(crop, is_p1=True):
            if crop.size == 0:
                return 0.0
            # Convert to HSV to isolate the green color of the health bar
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Check which columns contain green pixels
            col_has_green = np.any(mask > 0, axis=0)
            cols = len(col_has_green)
            green_cols = 0
            
            # Health is anchored at the portraits:
            # P1 HP bar drains from right-to-left (portrait is on the left)
            # P2 HP bar drains from left-to-right (portrait is on the right)
            if is_p1:
                # Count consecutive green columns starting from the left (portrait side)
                for val in col_has_green:
                    if val:
                        green_cols += 1
                    else:
                        break
            else:
                # Count consecutive green columns starting from the right (portrait side)
                for val in reversed(col_has_green):
                    if val:
                        green_cols += 1
                    else:
                        break
                        
            return (green_cols / cols) * 100.0 if cols > 0 else 0.0

        p1_hp = estimate_hp(p1_crop, is_p1=True)
        p2_hp = estimate_hp(p2_crop, is_p1=False)
        print(f"Estimated P1 HP: {p1_hp:.1f}%")
        print(f"Estimated P2 HP: {p2_hp:.1f}%")
        
        # Draw the target boxes on the preview image (using 4-channel colors for BGRA)
        cv2.rectangle(img, (p1_x1, y1), (p1_x2, y2), (0, 255, 0, 255), 2) # P1 HP Bounding Box (Green)
        cv2.rectangle(img, (p2_x1, y1), (p2_x2, y2), (0, 0, 255, 255), 2) # P2 HP Bounding Box (Red)
        
        cv2.rectangle(img, (p1_rg_x1, rg_y1), (p1_rg_x2, rg_y2), (255, 0, 0, 255), 2) # P1 Rider Gauge Box (Blue)
        cv2.rectangle(img, (p2_rg_x1, rg_y1), (p2_rg_x2, rg_y2), (255, 0, 0, 255), 2) # P2 Rider Gauge Box (Blue)
        
        cv2.rectangle(img, (p1_gg_x1, gg_y1), (p1_gg_x2, gg_y2), (0, 255, 255, 255), 2) # P1 Guard Gauge Box (Yellow)
        cv2.rectangle(img, (p2_gg_x1, gg_y1), (p2_gg_x2, gg_y2), (0, 255, 255, 255), 2) # P2 Guard Gauge Box (Yellow)
        
        annotated_filename = "game_capture_annotated.png"
        cv2.imwrite(annotated_filename, img)
        print(f"Saved annotated preview with HP, Rider & Guard Gauge bounding boxes to '{annotated_filename}'!")
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

