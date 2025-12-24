import cv2
import numpy as np
import glob
import os
import json
import matplotlib.pyplot as plt
import shutil
import pickle  # <--- חובה עבור ה-CACHE

# ==========================================
#              הגדרות משתמש
# ==========================================

# מצבי עבודה:
# "CALIBRATE" - חישוב המודל (K, D) מתיקיית images ושמירה ל-JSON
# "VALIDATE"  - בדיקת דיוק המודל הקיים על תמונות חדשות בתיקיית validate
# "TEST"      - תיקון תמונה בודדת (undistort)
OPERATION_MODE = "CALIBRATE"

# הגדרות תצוגה וקבצים
DISPLAY_SCALE = 0.3
TEST_IMAGE_NAME = "test"
CALIB_FILE = 'fisheye_calib_data.json'
CACHE_FILE = 'corners_cache_tracked.pkl'  # <--- שם קובץ ה-CACHE
CHECKERBOARD = (6, 9)


# ==========================================

def save_calibration_data(K, D, img_shape, filename=CALIB_FILE):
    data = {
        'K': K.tolist(),
        'D': D.tolist(),
        'img_shape': img_shape
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\n[V] Calibration data saved to '{filename}'")

def load_calibration_data(filename=CALIB_FILE):
    if not os.path.exists(filename):
        print(f"[ERROR] File '{filename}' not found. Run CALIBRATE first.")
        return None, None, None

    with open(filename, 'r') as f:
        data = json.load(f)

    K = np.array(data['K'])
    D = np.array(data['D'])
    img_shape = tuple(data['img_shape'])
    print(f"[V] Loaded calibration data from '{filename}'")
    return K, D, img_shape

def plot_coverage_matplotlib(imgPtsList, imgShape, title='Coverage Map'):
    width = imgShape[0]
    height = imgShape[1]
    all_x = []
    all_y = []
    for corners in imgPtsList:
        x_coords = corners[:, 0, 0]
        y_coords = corners[:, 0, 1]
        all_x.extend(x_coords)
        all_y.extend(y_coords)

    plt.figure(figsize=(10, 6))
    plt.plot([0, width, width, 0, 0], [0, 0, height, height, 0], 'r--', linewidth=2, label='Sensor Borders')
    plt.scatter(all_x, all_y, s=2, c='blue', alpha=0.5, label='Detected Corners')
    plt.xlim(-100, width + 100)
    plt.ylim(height + 100, -100)
    plt.title(f'{title} ({width}x{height})')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    print(">> Displaying Map... Close graph to continue.")
    plt.show()

def get_images_and_points(folder_name):
    """
    גרסה משופרת עם CACHE ודיאגנוסטיקה.
    """
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    # 1. ניסיון לטעון CACHE קיים
    corners_cache = {}
    if os.path.exists(CACHE_FILE):
        print(f"[Info] Loading cache from '{CACHE_FILE}'...")
        try:
            with open(CACHE_FILE, 'rb') as f:
                corners_cache = pickle.load(f)
            print(f"[Info] Loaded {len(corners_cache)} entries from cache.")
        except Exception as e:
            print(f"[Warning] Failed to load cache: {e}")
            corners_cache = {}

    cache_updated = False  # דגל לבדוק אם צריך לשמור מחדש בסוף

    # יצירת תיקיית כישלונות
    failed_dir = os.path.join(folder_name, 'failed_detection')
    if not os.path.exists(failed_dir):
        os.makedirs(failed_dir)

    # קובץ לוג לשמירת הסיבות
    log_file_path = os.path.join(failed_dir, "failures_log.txt")
    with open(log_file_path, "w") as f:
        f.write("Failure Analysis Log:\n=====================\n")

    # הכנת נקודות 3D
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    # טעינת תמונות
    exts = ['*.jpg', '*.png', '*.jpeg', '*.JPG']
    images = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(folder_name, ext)))
    images = list(set(images))

    if not images:
        return None, None, None, []

    print(f"Scanning {len(images)} unique images in '{folder_name}'...")
    img_shape = None
    valid_images = []

    # === כלי עבודה ===
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    kernel_dilate = np.ones((3, 3), np.uint8)

    # --- פונקציית עזר לאבחון כישלון ---
    def analyze_fail_reason(image_gray):
        mean_val = np.mean(image_gray)
        std_val = np.std(image_gray)
        reason = "Unknown"
        if mean_val < 40:
            reason = "Too Dark (Under-exposed)"
        elif mean_val > 220:
            reason = "Too Bright (Over-exposed)"
        elif std_val < 20:
            reason = "Low Contrast (Murky water/Fog)"
        else:
            reason = "Geometry/Occlusion (Crop, Rope, Blur, or Angle)"
        return reason, mean_val, std_val

    for fname in images:
        fname_key = os.path.basename(fname)  # המפתח ב-Cache הוא שם הקובץ בלבד

        # === בדיקה ב-CACHE ===
        if fname_key in corners_cache:
            cache_data = corners_cache[fname_key]

            # --- התיקון כאן: חילוץ המערך מתוך המילון אם צריך ---
            if isinstance(cache_data, dict) and 'corners' in cache_data:
                # המידע הגיע מסקריפט הסינון (מילון)
                cached_corners = cache_data['corners']
            else:
                # המידע הגיע מהסקריפט הזה (מערך ישיר)
                cached_corners = cache_data
            # ----------------------------------------------------

            objpoints.append(objp)
            imgpoints.append(cached_corners)  # עכשיו זה בטוח numpy array
            valid_images.append(fname_key)

            if img_shape is None:
                temp_img = cv2.imread(fname)
                if temp_img is not None:
                    img_shape = temp_img.shape[:2][::-1]

            print(f"[Cache] Loaded: {fname_key}")
            continue
        # =====================

        # אם הגענו לכאן - התמונה לא ב-Cache, צריך לחשב
        img = cv2.imread(fname)
        if img is None: continue

        gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray_orig.shape[::-1]

        found = False
        corners = None

        # === Pipeline Attempts ===
        norm_img = cv2.normalize(gray_orig, None, 0, 255, cv2.NORM_MINMAX)
        clahe_img = clahe.apply(gray_orig)
        dilated_img = cv2.dilate(clahe_img, kernel_dilate, iterations=1)
        _, binary_img = cv2.threshold(gray_orig, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        morph_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel_dilate)

        attempts = [
            ("Original", gray_orig),
            ("Normalize", norm_img),
            ("CLAHE", clahe_img),
            ("Dilate", dilated_img),
            ("Binary Morph", morph_img)
        ]

        find_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE

        for method_name, processed_img in attempts:
            ret, corners = cv2.findChessboardCornersSB(processed_img, CHECKERBOARD, find_flags)

            if ret:
                print(f"[New] Found: {fname_key} | Method: {method_name}")
                objpoints.append(objp)

                refine_src = gray_orig
                if method_name in ["Binary Morph", "Dilate"]:
                    refine_src = clahe_img

                corners2 = cv2.cornerSubPix(refine_src, corners, (11, 11), (-1, -1), subpix_criteria)
                imgpoints.append(corners2)
                valid_images.append(fname_key)

                # שמירה ל-CACHE
                corners_cache[fname_key] = corners2
                cache_updated = True

                found = True
                break

        if not found:
            reason_str, mean_v, std_v = analyze_fail_reason(gray_orig)
            msg = f"[-] FAILED: {fname_key} | Suspect: {reason_str}"
            print(msg)
            with open(log_file_path, "a") as f:
                f.write(f"{fname_key}: {reason_str}\n")
            try:
                dst_path = os.path.join(failed_dir, fname_key)
                shutil.move(fname, dst_path)
            except Exception:
                pass

    # שמירת Cache מעודכן לדיסק בסוף הריצה
    if cache_updated:
        print(f"[Info] Saving updated cache to '{CACHE_FILE}'...")
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(corners_cache, f)

    return objpoints, imgpoints, img_shape, valid_images

def run_calibration():
    objpoints, imgpoints, img_shape, _ = get_images_and_points('images')

    if not objpoints:
        print("No valid images found (check folder or cache).")
        return

    plot_coverage_matplotlib(imgpoints, img_shape, "Calibration Coverage")

    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC  + cv2.fisheye.CALIB_FIX_SKEW + cv2.fisheye.CALIB_CHECK_COND

    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

    print(f"\nCalibrating on {N_OK} images...")
    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints, imgpoints, img_shape, K, D, rvecs, tvecs, calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

    print(f"--------------------------------")
    print(f"RMS Error: {rms:.4f} px")
    print(f"--------------------------------")
    save_calibration_data(K, D, img_shape)

def run_validation():
    K, D, expected_shape = load_calibration_data()
    if K is None: return

    val_folder = 'validate'
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)
        return

    # גם בוולידציה נשתמש באותה פונקציה (היא תשתמש ב-Cache שלה או תייצר חדש)
    # שימו לב: אם רוצים Cache נפרד לוולידציה, צריך לשנות את שם הקובץ הגלובלי,
    # אבל בד"כ בוולידציה התמונות מעטות ורצות מהר.
    objpoints, imgpoints, img_shape, filenames = get_images_and_points(val_folder)

    if not objpoints:
        print(f"No valid images found in '{val_folder}'.")
        return

    flags = cv2.fisheye.CALIB_USE_INTRINSIC_GUESS | cv2.fisheye.CALIB_FIX_INTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    N_OK = len(objpoints)
    rvecs_init = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs_init = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

    print(f"\nValidating on {N_OK} images...")
    rms, _, _, rvecs_new, tvecs_new = cv2.fisheye.calibrate(
        objpoints, imgpoints, img_shape, K, D, rvecs_init, tvecs_init, flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

    print(f"Validation RMS: {rms:.4f} px")
    # ... (המשך קוד הוולידציה נשאר זהה) ...

def run_test_image():
    # ... (קוד הטסט נשאר זהה) ...
    K, D, dim = load_calibration_data()
    if K is None: return

    potential_files = glob.glob(f"{TEST_IMAGE_NAME}.*")
    valid_files = [f for f in potential_files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if not valid_files: return

    img = cv2.imread(valid_files[0])
    h, w = img.shape[:2]
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=1.0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    cv2.imwrite("result_undistort.jpg", undistorted)
    print("Saved result_undistort.jpg")


# ==========================================
#                   MAIN
# ==========================================
if __name__ == "__main__":
    if OPERATION_MODE == "CALIBRATE":
        run_calibration()
    elif OPERATION_MODE == "VALIDATE":
        run_validation()
    elif OPERATION_MODE == "TEST":
        run_test_image()