import cv2
import numpy as np
import glob
import os
import json
import matplotlib.pyplot as plt
import shutil  # <--- הוספנו את הספרייה הזו להעברת קבצים

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
TEST_IMAGE_NAME = "test"  # עבור מצב TEST
CALIB_FILE = 'fisheye_calib_data.json'
CHECKERBOARD = (6, 9)  # (שורות, עמודות) פנימיות


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
    """ פונקציית עזר לטעינת תמונות ומציאת פינות """
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

    # יצירת תיקיית try אם אינה קיימת
    failed_dir = os.path.join(folder_name, 'try')
    if not os.path.exists(failed_dir):
        os.makedirs(failed_dir)
        print(f"[INFO] Created directory for failed images: {failed_dir}")

    # הכנת נקודות 3D
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []

    # תמיכה בפורמטים שונים
    exts = ['*.jpg', '*.png', '*.jpeg']
    images = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(folder_name, ext)))

    if not images:
        return None, None, None, []

    print(f"Scanning {len(images)} images in '{folder_name}'...")
    img_shape = None
    valid_images = []

    for fname in images:
        img = cv2.imread(fname)
        if img is None: continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
            imgpoints.append(corners2)
            valid_images.append(os.path.basename(fname))
            print(f"[+] Found: {os.path.basename(fname)}")
        else:
            # === לוגיקת העברה לתיקיית try ===
            print(f"[-] No corners: {os.path.basename(fname)}")
            try:
                dst_path = os.path.join(failed_dir, os.path.basename(fname))
                shutil.move(fname, dst_path)
                print(f"    -> Moved to: {dst_path}")
            except Exception as e:
                print(f"    -> Error moving file: {e}")
            # ===============================

    return objpoints, imgpoints, img_shape, valid_images


def run_calibration():
    objpoints, imgpoints, img_shape, _ = get_images_and_points('images')

    if not objpoints:
        print("No valid images found in 'images' folder (or all were moved to 'try').")
        return

    plot_coverage_matplotlib(imgpoints, img_shape, "Calibration Coverage")

    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

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
    # 1. טעינת הכיול הקיים
    K, D, expected_shape = load_calibration_data()
    if K is None: return

    # 2. טעינת תמונות מתיקיית VALIDATE
    val_folder = 'validate'
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)
        print(f"[INFO] Created folder '{val_folder}'. Please put test images there.")
        return

    objpoints, imgpoints, img_shape, filenames = get_images_and_points(val_folder)

    if not objpoints:
        print(f"No valid images found in '{val_folder}' folder.")
        return

    # 3. חישוב השגיאה
    flags = cv2.fisheye.CALIB_USE_INTRINSIC_GUESS | cv2.fisheye.CALIB_FIX_INTRINSIC | cv2.fisheye.CALIB_FIX_SKEW

    N_OK = len(objpoints)
    # אתחול רשימות ריקות (לא קריטי כי אנחנו דורסים אותן תכף, אבל למען הסדר הטוב)
    rvecs_init = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs_init = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

    print(f"\nValidating on {N_OK} images using saved K and D...")

    # --- התיקון כאן: קליטת הערכים החוזרים במשתנים חדשים ---
    rms, _, _, rvecs_new, tvecs_new = cv2.fisheye.calibrate(
        objpoints, imgpoints, img_shape, K, D, rvecs_init, tvecs_init, flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

    print(f"========================================")
    print(f"VALIDATION RESULTS")
    print(f"========================================")
    print(f"Total Mean RMS Error: {rms:.4f} px")

    print("\nPer-Image Errors:")

    for i in range(N_OK):
        # שימוש ב-rvecs_new ו-tvecs_new המחושבים
        img_pts_projected, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs_new[i], tvecs_new[i], K, D)

        # המרה כפויה של הצורה
        img_pts_projected = img_pts_projected.reshape(imgpoints[i].shape)

        # חישוב המרחק האוקלידי
        error = cv2.norm(imgpoints[i], img_pts_projected, cv2.NORM_L2) / np.sqrt(len(img_pts_projected))

        # סימון חריגים (מעל 5 פיקסלים נחשב גבוה בוולידציה)
        status = "OK" if error < 5.0 else "HIGH ERROR"
        print(f"  [{i + 1}] {filenames[i]:<25} : {error:.4f} px  \t{status}")

    print(f"========================================")
    plot_coverage_matplotlib(imgpoints, img_shape, "Validation Set Coverage")


def run_test_image():
    K, D, dim = load_calibration_data()
    if K is None: return

    potential_files = glob.glob(f"{TEST_IMAGE_NAME}.*")
    valid_files = [f for f in potential_files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not valid_files:
        print(f"[ERROR] Could not find '{TEST_IMAGE_NAME}'")
        return

    img_path = valid_files[0]
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # חישוב המטריצה החדשה והאופטימלית
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=1.0)

    # --- הוספת ההדפסה של הפרמטרים החדשים ---
    fx_new = new_K[0, 0]
    fy_new = new_K[1, 1]

    print("\n" + "=" * 40)
    print("NEW OPTIMAL CAMERA PARAMETERS")
    print("=" * 40)
    print(f"New fx: {fx_new:.4f} pixels")
    print(f"New fy: {fy_new:.4f} pixels")
    print(f"Average f: {(fx_new + fy_new) / 2:.4f} pixels")  # השתמש בזה לנוסחה
    print("=" * 40 + "\n")
    # -----------------------------------------

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    save_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_modif.jpg"
    cv2.imwrite(save_name, undistorted_img)
    print(f"[V] Saved: {save_name}")

    small_orig = cv2.resize(img, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    small_undist = cv2.resize(undistorted_img, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    cv2.imshow("Result", np.hstack((small_orig, small_undist)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================
#                   MAIN
# ==========================================
if __name__ == "__main__":
    print(f"--- MODE: {OPERATION_MODE} ---")

    if OPERATION_MODE == "CALIBRATE":
        run_calibration()
    elif OPERATION_MODE == "VALIDATE":
        run_validation()
    elif OPERATION_MODE == "TEST":
        run_test_image()
    else:
        print("Invalid Mode.")