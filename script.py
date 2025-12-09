import cv2
import numpy as np
import os
import glob


def analyze_checkerboard_failure(image_path, board_size=(6, 9)):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read {image_path}")
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---------------------------------------------------------
    # שלב 1: ניסיון זיהוי רגיל
    # ---------------------------------------------------------
    ret, corners = cv2.findChessboardCorners(gray, board_size,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH +
                                             cv2.CALIB_CB_FAST_CHECK +
                                             cv2.CALIB_CB_NORMALIZE_IMAGE)

    status = "SUCCESS" if ret else "FAILED"
    color = (0, 255, 0) if ret else (0, 0, 255)
    print(f"Analysis for {os.path.basename(image_path)}: {status}")

    # ---------------------------------------------------------
    # שלב 2: "ראיית הרובוט" (Adaptive Threshold)
    # ---------------------------------------------------------
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 29, 2)
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # ---------------------------------------------------------
    # שלב 3: זיהוי פינות "טיפש" (Harris Corners)
    # ---------------------------------------------------------
    corners_raw = cv2.goodFeaturesToTrack(gray, maxCorners=2000, qualityLevel=0.01, minDistance=10)
    raw_corners_img = img.copy()
    if corners_raw is not None:
        corners_raw = np.int0(corners_raw)
        for i in corners_raw:
            x, y = i.ravel()
            # === שינוי גודל: רדיוס 8 במקום 3 ===
            cv2.circle(raw_corners_img, (x, y), 8, (0, 0, 255), -1)

    # ---------------------------------------------------------
    # תצוגה
    # ---------------------------------------------------------
    cv2.putText(img, f"Status: {status}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

    scale = 0.15

    h, w = img.shape[:2]
    small_size = (int(w * scale), int(h * scale))

    img_small = cv2.resize(img, small_size)
    thresh_small = cv2.resize(thresh_color, small_size)
    raw_small = cv2.resize(raw_corners_img, small_size)

    top_row = np.hstack((img_small, thresh_small))

    cv2.imshow(f"Analysis (Original | Threshold) - {os.path.basename(image_path)}", top_row)
    cv2.imshow(f"Raw Corners Detected (BIG DOTS) - {os.path.basename(image_path)}", raw_small)

    print("Press any key for next image, or 'q' to quit...")
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return key == ord('q')


# --- הרצה על תיקייה ---
folder = 'images'
extensions = ['*.jpg', '*.png', '*.jpeg', '*.JPG']

files = []
for ext in extensions:
    files.extend(glob.glob(os.path.join(folder, ext)))

if not files:
    print(f"No images found in '{folder}' folder.")
else:
    print(f"Starting analysis on {len(files)} images...")
    for f in files:
        if analyze_checkerboard_failure(f, board_size=(6, 9)):
            break