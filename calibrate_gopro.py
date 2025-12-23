import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt
import json
import sys
import time

# --- הגדרות הלוח ---
# הקוד ינסה אוטומטית גם 9x6 וגם 6x9.
# הזן כאן את המספרים כפי שספרת (צמתים פנימיים).
N_ROWS = 9
N_COLS = 6
SQUARE_SIZE = 15  # מ"מ


def read_image_safe(path):
    """ קריאת תמונה שתומכת בנתיבים עם עברית """
    try:
        with open(path, "rb") as f:
            bytes_data = bytearray(f.read())
            numpy_array = np.asarray(bytes_data, dtype=np.uint8)
            img = cv.imdecode(numpy_array, cv.IMREAD_COLOR)
            return img
    except Exception as e:
        return None


def resize_for_display(img, target_height=400):
    h, w = img.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv.resize(img, (new_w, new_h))


def get_object_points(rows, cols, square_size):
    """ מייצר את רשת הנקודות התלת-ממדית (0,0,0), (1,0,0)... בהתאם למימדים שנמצאו """
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
    return objp * square_size


def find_corners_robust(img_gray, base_rows, base_cols):
    """
    זיהוי פינות חכם שמנסה גם כיוון רגיל וגם כיוון מסובב (9x6 וגם 6x9).
    מחזיר: (הצלחה, פינות, שם השיטה, המימדים שנמצאו)
    """

    # רשימת המימדים לבדיקה (גם לרוחב וגם לאורך)
    dims_to_try = [(base_rows, base_cols), (base_cols, base_rows)]

    # דגלים לכל שיטה
    sb_flags = cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_EXHAUSTIVE + cv.CALIB_CB_ACCURACY
    std_flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

    for rows, cols in dims_to_try:
        current_dims = (rows, cols)

        # --- שיטה 1: SB (הכי טובה) ---
        try:
            ret, corners = cv.findChessboardCornersSB(img_gray, current_dims, flags=sb_flags)
            if ret: return True, corners, "SB", current_dims
        except:
            pass

        # --- שיטה 2: Padding (אם הלוח בקצה) ---
        border = int(max(img_gray.shape) * 0.05)
        img_padded = cv.copyMakeBorder(img_gray, border, border, border, border, cv.BORDER_CONSTANT, value=255)
        try:
            ret, corners_padded = cv.findChessboardCornersSB(img_padded, current_dims, flags=sb_flags)
            if ret:
                corners_padded[:, 0, 0] -= border
                corners_padded[:, 0, 1] -= border
                return True, corners_padded, "Padding", current_dims
        except:
            pass

        # --- שיטה 3: רגילה (גיבוי) ---
        ret, corners = cv.findChessboardCorners(img_gray, current_dims, flags=std_flags)
        if ret: return True, corners, "Standard", current_dims

    return False, None, "Fail", None


def plot_coverage_matplotlib(imgPtsList, imgShape):
    width = imgShape[0]
    height = imgShape[1]
    all_x = []
    all_y = []
    for corners in imgPtsList:
        all_x.extend(corners[:, 0, 0])
        all_y.extend(corners[:, 0, 1])

    plt.figure(figsize=(10, 6))
    plt.plot([0, width, width, 0, 0], [0, 0, height, height, 0], 'r--', linewidth=2, label="Sensor")
    plt.scatter(all_x, all_y, s=2, c='green', alpha=0.6, label="Corners")
    plt.xlim(-100, width + 100)
    plt.ylim(height + 100, -100)
    plt.title(f'Calibration Coverage Map ({width}x{height})')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    print("\nDisplaying Coverage Map... Close graph to continue.")
    plt.show()


def calibrate(showPics=True, complexity_mode="simple"):
    root = os.getcwd()
    calibrationDir = os.path.join(root, 'images')

    if not os.path.exists(calibrationDir):
        print(f"[ERROR] Folder not found: {calibrationDir}")
        return None, None, 0

    imgPathList = glob.glob(os.path.join(calibrationDir, '*.jpg')) + \
                  glob.glob(os.path.join(calibrationDir, '*.png')) + \
                  glob.glob(os.path.join(calibrationDir, '*.jpeg'))

    total_images = len(imgPathList)
    print(f"Found {total_images} images.")

    if total_images == 0:
        return None, None, 0

    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    worldPtsList = []
    imgPtsList = []
    imgShape = None

    window_name = 'Calibration Scanner'
    if showPics:
        cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)

    print("Scanning images...")
    deleted_count = 0

    for i, curImgPath in enumerate(imgPathList):
        filename = os.path.basename(curImgPath)

        # חיווי התקדמות
        good_count = len(worldPtsList)
        sys.stdout.write(f"\r[Proc] {i + 1}/{total_images} | Good: {good_count} | {filename:<20}")
        sys.stdout.flush()

        imgBGR = read_image_safe(curImgPath)
        if imgBGR is None: continue

        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
        if imgShape is None: imgShape = imgGray.shape[::-1]

        if showPics:
            display_img = imgBGR.copy()
            cv.putText(display_img, "Scanning...", (30, 80), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
            cv.imshow(window_name, resize_for_display(display_img))
            cv.waitKey(1)

        # === החלק החכם: זיהוי פינות גמיש ===
        found, cornersOrg, method, dims_found = find_corners_robust(imgGray, N_ROWS, N_COLS)

        if found:
            # === הצלחה ===
            # יצירת נקודות 3D שמתאימות בדיוק למימדים שנמצאו בתמונה זו (9x6 או 6x9)
            obj_points = get_object_points(dims_found[0], dims_found[1], SQUARE_SIZE)
            worldPtsList.append(obj_points)

            # שיפור דיוק אם השיטה היא סטנדרטית
            if method == "Standard":
                cornersRefined = cv.cornerSubPix(imgGray, cornersOrg, (11, 11), (-1, -1), termCriteria)
            else:
                cornersRefined = cornersOrg

            imgPtsList.append(cornersRefined)

            if showPics:
                display_img = imgBGR.copy()
                cv.drawChessboardCorners(display_img, dims_found, cornersRefined, found)
                # כתיבת מידע ירוק
                msg = f"OK! {dims_found[0]}x{dims_found[1]}"
                cv.putText(display_img, msg, (30, 80), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
                cv.imshow(window_name, resize_for_display(display_img))
                cv.waitKey(50)
        else:
            # === כישלון ומחיקה ===
            if showPics:
                display_img = imgBGR.copy()
                cv.putText(display_img, "FAIL - DELETING", (30, 80), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                cv.imshow(window_name, resize_for_display(display_img))
                cv.waitKey(50)

            try:
                # השהייה קטנטנה לשחרור הקובץ אם הוא נתפס
                time.sleep(0.05)
                os.remove(curImgPath)
                deleted_count += 1
                sys.stdout.write(" -> DELETED")
            except OSError as e:
                sys.stdout.write(f" -> Err: {e}")

    if showPics:
        cv.destroyAllWindows()

    print(f"\n\nDone. Deleted {deleted_count} bad images.")

    n_used_imgs = len(worldPtsList)
    if n_used_imgs < 10:
        print(f"[WARNING] Only {n_used_imgs} good images remain. Need at least 10 for good calibration.")
        return None, None, n_used_imgs

    print("Generating Coverage Map...")
    plot_coverage_matplotlib(imgPtsList, imgShape)

    print(f"\nStarting Calibration ({complexity_mode})...")

    base_flags = 0
    if complexity_mode == "simple":
        calib_flags = base_flags
        print(">> Mode: Standard (K1-K3, P1-P2)")
    elif complexity_mode == "complex":
        calib_flags = base_flags | cv.CALIB_RATIONAL_MODEL
        print(">> Mode: Rational (K1-K6, P1-P2) - For wide angles")
    else:
        calib_flags = base_flags

    t_start = time.time()
    try:
        ret_val, camMatrix, distCoeff, rvecs, tvecs = cv.calibrateCamera(
            worldPtsList, imgPtsList, imgShape, None, None, flags=calib_flags, criteria=termCriteria
        )
    except cv.error as e:
        print(f"\n[ERROR] Calibration Math Failed: {e}")
        return None, None, n_used_imgs

    duration = time.time() - t_start

    print('\n' + '=' * 40)
    print(f'      RESULTS ({complexity_mode})      ')
    print('=' * 40)
    print(f"Images Used:     {n_used_imgs}")
    print(f"Calc Time:       {duration:.2f} sec")
    print(f"Reproj Error:    {ret_val:.4f} px (Goal: < 1.0)")
    print('-' * 40)
    print('Camera Matrix (K):\n', camMatrix)
    print('Distortion Coeffs:\n', distCoeff.ravel())
    print('=' * 40)

    output_data = {
        'mode': complexity_mode,
        'error': float(ret_val),
        'camera_matrix': camMatrix.tolist(),
        'dist_coeff': distCoeff.tolist(),
        'resolution': imgShape
    }

    json_path = os.path.join(root, 'calibration_data.json')
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        print(f"Saved to {json_path}")

    return camMatrix, distCoeff, n_used_imgs


if __name__ == '__main__':
    calibrate(showPics=True, complexity_mode="simple")