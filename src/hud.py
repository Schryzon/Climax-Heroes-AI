import cv2
import numpy as np

class Hud_Parser:
    def read_hps(self, bgr):
        h, w, _ = bgr.shape
        p1_x1, p1_x2 = int(w * 0.220), int(w * 0.445)
        p2_x1, p2_x2 = int(w * 0.555), int(w * 0.780)
        y1, y2 = int(h * 0.087), int(h * 0.113)
        
        p1_crop = cv2.cvtColor(bgr[y1:y2, p1_x1:p1_x2], cv2.COLOR_BGR2HSV)
        p2_crop = cv2.cvtColor(bgr[y1:y2, p2_x1:p2_x2], cv2.COLOR_BGR2HSV)
        
        return (self._estimate_layered_hp(p1_crop), 
                self._estimate_layered_hp(p2_crop))

    def _estimate_layered_hp(self, hsv_crop):
        if hsv_crop.size == 0:
            return 0.0
            
        # Color ranges (hsv_crop is already in HSV space):
        # Green (Stack 3)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv_crop, lower_green, upper_green)
        
        # Yellow (Stack 2)
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv_crop, lower_yellow, upper_yellow)
        
        # Red (Stack 1 - red wraps around 0 and 180 in HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_red1 = cv2.inRange(hsv_crop, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_crop, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        def get_fraction(mask):
            col_has_active = np.any(mask > 0, axis=0)
            active_cols = np.sum(col_has_active)
            cols = len(col_has_active)
            # Calibrated padding divisor (since green bar doesn't stretch 100% of the box)
            max_active_fraction = 0.945
            pct = (active_cols / (cols * max_active_fraction)) * 100.0
            return min(100.0, pct)

        # Check stack levels sequentially (Green -> Yellow -> Red)
        green_pct = get_fraction(mask_green)
        if green_pct > 5.0: # 5% buffer to avoid boundary noise
            return 200.0 + green_pct
            
        yellow_pct = get_fraction(mask_yellow)
        if yellow_pct > 5.0:
            return 100.0 + yellow_pct
            
        red_pct = get_fraction(mask_red)
        if red_pct > 5.0:
            return red_pct
            
        return 0.0

    def read_guard_gauges(self, bgr):
        h, w, _ = bgr.shape
        p1_x1, p1_x2 = int(w * 0.220), int(w * 0.336)
        p2_x1, p2_x2 = int(w * 0.664), int(w * 0.780)
        y1, y2 = int(h * 0.120), int(h * 0.138)
        
        p1_crop = cv2.cvtColor(bgr[y1:y2, p1_x1:p1_x2], cv2.COLOR_BGR2HSV)
        p2_crop = cv2.cvtColor(bgr[y1:y2, p2_x1:p2_x2], cv2.COLOR_BGR2HSV)
        
        return (self._estimate_guard_gauge_color(p1_crop), 
                self._estimate_guard_gauge_color(p2_crop))

    def _estimate_guard_gauge_color(self, hsv_crop):
        if hsv_crop.size == 0:
            return 0.0
            
        # Strip vertical margins to avoid HUD borders or screen capture marking outlines
        h_crop = hsv_crop.shape[0]
        margin = max(1, min(3, int(h_crop * 0.15)))
        crop_stripped = hsv_crop[margin:-margin, :]
        
        if crop_stripped.size == 0:
            return 0.0
            
        # Guard bar normal colors (Green, Yellow, Blue).
        # We explicitly restrict the Hue range to 15-165 to exclude the broken guard's flashing Red (0-15 and 165-180).
        # This keeps a broken guard reading stably at 0.0 rather than pulsing/bouncing wildy.
        lower_color = np.array([15, 100, 100])
        upper_color = np.array([165, 255, 255])
        mask = cv2.inRange(crop_stripped, lower_color, upper_color)
            
        col_has_pixels = np.any(mask > 0, axis=0)
        active_cols = np.sum(col_has_pixels)
        cols = len(col_has_pixels)
        
        # Guard bars fill about 90% of their crop bounding boxes when full
        max_active_fraction = 0.90
        pct = (active_cols / (cols * max_active_fraction)) * 100.0
        return min(100.0, pct)

    def read_rider_gauges(self, bgr):
        h, w, _ = bgr.shape
        p1_x1, p1_x2 = int(w * 0.220), int(w * 0.380)
        p2_x1, p2_x2 = int(w * 0.620), int(w * 0.780)
        y1, y2 = int(h * 0.884), int(h * 0.921)
        
        p1_crop = cv2.cvtColor(bgr[y1:y2, p1_x1:p1_x2], cv2.COLOR_BGR2HSV)
        p2_crop = cv2.cvtColor(bgr[y1:y2, p2_x1:p2_x2], cv2.COLOR_BGR2HSV)
        
        return (self._estimate_gauge_color(p1_crop), 
                self._estimate_gauge_color(p2_crop))

    def _estimate_gauge_color(self, hsv_crop):
        if hsv_crop.size == 0:
            return 0.0
            
        # Strip vertical margins to avoid HUD borders or screen capture marking outlines
        h_crop = hsv_crop.shape[0]
        margin = max(1, min(3, int(h_crop * 0.15)))
        crop_stripped = hsv_crop[margin:-margin, :]
        
        if crop_stripped.size == 0:
            return 0.0
            
        # Saturation-based threshold (captures highly saturated colors: Red, Blue, Green, Yellow, etc.)
        # and filters out metallic gray/silver casing and black/white boundaries.
        lower_color = np.array([0, 100, 100])
        upper_color = np.array([180, 255, 255])
        mask = cv2.inRange(crop_stripped, lower_color, upper_color)
            
        col_has_pixels = np.any(mask > 0, axis=0)
        active_cols = np.sum(col_has_pixels)
        cols = len(col_has_pixels)
        
        # Guard/Rider bars fill about 90% of their crop bounding boxes when full
        max_active_fraction = 0.90
        pct = (active_cols / (cols * max_active_fraction)) * 100.0
        return min(100.0, pct)

    def read_combo_count(self, bgr):
        # Optional: OCR / template-matching for combo count later
        return 0

    def read_rounds_won(self, bgr):
        h, w, _ = bgr.shape
        p1_x1, p1_x2 = int(w * 0.441), int(w * 0.472)
        p2_x1, p2_x2 = int(w * 0.528), int(w * 0.559)
        y1, y2 = int(h * 0.137), int(h * 0.212)
        
        p1_crop_bgr = bgr[y1:y2, p1_x1:p1_x2]
        p2_crop_bgr = bgr[y1:y2, p2_x1:p2_x2]
        
        if p1_crop_bgr.size == 0 or p2_crop_bgr.size == 0:
            return 0, 0
            
        p1_crop = cv2.cvtColor(p1_crop_bgr, cv2.COLOR_BGR2HSV)
        p2_crop = cv2.cvtColor(p2_crop_bgr, cv2.COLOR_BGR2HSV)
        
        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([45, 255, 255])
        
        mask_p1 = cv2.inRange(p1_crop, lower_yellow, upper_yellow)
        mask_p2 = cv2.inRange(p2_crop, lower_yellow, upper_yellow)
        
        p1_yellow_pixels = np.sum(mask_p1 > 0)
        p2_yellow_pixels = np.sum(mask_p2 > 0)
        
        p1_rounds = 0
        if p1_yellow_pixels >= 350:
            p1_rounds = 2
        elif p1_yellow_pixels >= 25:
            p1_rounds = 1
            
        p2_rounds = 0
        if p2_yellow_pixels >= 350:
            p2_rounds = 2
        elif p2_yellow_pixels >= 25:
            p2_rounds = 1
            
        return p1_rounds, p2_rounds

    def is_timer_infinite(self, bgr):
        h, w, _ = bgr.shape
        t_x1, t_x2 = int(w * 0.468), int(w * 0.531)
        t_y1, t_y2 = int(h * 0.087), int(h * 0.156)
        
        timer_crop_bgr = bgr[t_y1:t_y2, t_x1:t_x2]
        if timer_crop_bgr.size == 0:
            return False
            
        timer_crop = cv2.cvtColor(timer_crop_bgr, cv2.COLOR_BGR2HSV)
        # Mint green HSV range for the infinity symbol
        lower_mint = np.array([40, 30, 150])
        upper_mint = np.array([80, 180, 255])
        mask = cv2.inRange(timer_crop, lower_mint, upper_mint)
        
        return np.sum(mask > 0) > 100

    def is_survival_game_over(self, bgr):
        h, w, _ = bgr.shape
        # Crop the bottom banner area dynamically: y = [0.60, 0.85], x = [0.10, 0.90]
        y1, y2 = int(h * 0.60), int(h * 0.85)
        x1, x2 = int(w * 0.10), int(w * 0.90)
        banner_area = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        
        # Teal HSV range for the banner background
        lower_teal = np.array([85, 100, 100])
        upper_teal = np.array([115, 255, 255])
        mask_teal = cv2.inRange(banner_area, lower_teal, upper_teal)
        
        # Crop the Proceed button pill area on the right dynamically: y = [0.66, 0.80], x = [0.68, 0.86]
        py1, py2 = int(h * 0.66), int(h * 0.80)
        px1, px2 = int(w * 0.68), int(w * 0.86)
        proceed_hsv = cv2.cvtColor(bgr[py1:py2, px1:px2], cv2.COLOR_BGR2HSV)
        
        # Red HSV range for the Circle button prompt inside the pill
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask_red1 = cv2.inRange(proceed_hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(proceed_hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        return np.sum(mask_teal > 0) > 30000 and np.sum(mask_red > 0) > 100

    def is_vs_splash_screen(self, bgr):
        h, w, _ = bgr.shape
        
        # Crop the upper sky region of the screen (y = 10% to 45%) where the artificial green backdrop boxes live.
        sky_hsv = cv2.cvtColor(bgr[int(h * 0.10):int(h * 0.45), int(w * 0.05):int(w * 0.95)], cv2.COLOR_BGR2HSV)
        
        # Artificial backdrop green box: Hue: 40-75, Sat: 50-255, Val: 50-255
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(sky_hsv, lower_green, upper_green)
        
        # Orange/red background canvas (Hue: 10-30, Sat: 100-255, Val: 100-255)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([30, 255, 255])
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        
        return np.sum(mask_green > 0) > 20000 and np.sum(mask_orange > 0) > 40000

    def is_now_loading_screen(self, bgr):
        h, w, _ = bgr.shape
        
        # Crop around the pink card symbol on the bottom black bar:
        c_x1, c_x2 = int(w * 0.50), int(w * 0.60)
        c_y1, c_y2 = int(h * 0.78), int(h * 0.94)
        
        card_crop_bgr = bgr[c_y1:c_y2, c_x1:c_x2]
        if card_crop_bgr.size == 0:
            return False
            
        card_crop = cv2.cvtColor(card_crop_bgr, cv2.COLOR_BGR2HSV)
        lower_pink = np.array([140, 80, 80])
        upper_pink = np.array([170, 255, 255])
        mask_pink = cv2.inRange(card_crop, lower_pink, upper_pink)
        
        return np.sum(mask_pink > 0) > 100
