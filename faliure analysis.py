import cv2 as cv
import numpy as np

# --- הגדרות ---
img_path = "images/image.jpg"  # <--- שנה לשם הקובץ הבעייתי שלך


# ---------------

def check_checkerboard_dims(path):
    print(f"Checking image: {path}")
    img = cv.imread(path)
    if img is None:
        print("Error: Could not read image.")
        return

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # ניסיון לשפר קונטרסט (למקרה שהתאורה גרועה)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    print("Brute-forcing dimensions (rows x cols)...")

    found_any = False

    # לולאה שמנסה הכל מ-3x3 ועד 12x12
    for r in range(3, 13):
        for c in range(3, 13):
            # שימוש בדגלים מתירניים יותר לבדיקה
            flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

            ret, corners = cv.findChessboardCorners(gray, (r, c), flags)

            if ret:
                print(f"\n✅ SUCCESS! Found pattern: {r} x {c} (Inner Corners)")
                print(f"   (This usually means the board is {r + 1} x {c + 1} squares)")

                # ציור התוצאה
                cv.drawChessboardCorners(img, (r, c), corners, ret)

                # הקטנה לתצוגה
                h, w = img.shape[:2]
                scale = 600 / h
                dim = (int(w * scale), int(h * scale))
                resized = cv.resize(img, dim)

                cv.imshow(f"Match: {r}x{c}", resized)
                cv.waitKey(0)
                found_any = True
                # אנחנו לא עוצרים, אולי הוא ימצא עוד קומבינציות (לפעמים מוצאים תת-לוח)

    if not found_any:
        print("\n❌ FAILED. Tried all combinations from 3x3 to 12x12.")
        print("Possible reasons:")
        print("1. Not enough white border (padding) around the black squares.")
        print("2. Extreme lens distortion (fisheye).")
        print("3. Poor lighting / Reflection on the board.")

    cv.destroyAllWindows()


if __name__ == "__main__":
    check_checkerboard_dims(img_path)