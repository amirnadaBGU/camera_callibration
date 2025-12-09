import cv2
import numpy as np
import os
import glob
import json
import shutil
import pickle
import matplotlib.pyplot as plt  # וודא שמותקן: pip install matplotlib

# ==========================================
#              הגדרות משתמש
# ==========================================
CALIB_FILE = 'fisheye_calib_data.json'
IMAGE_FOLDER = 'images'
CACHE_FILE = 'corners_cache_tracked.pkl'
CHECKERBOARD = (6, 9)
WORST_PERCENTAGE = 0.20  # 20%

# הגדרת תיקיות המחיקה
DELETE_ROOT = 'TO_DELETE'  # תיקיית האב
DIR_CENTER = 'CENTER_JUNK'  # תת-תיקייה 1: זבל במרכז
DIR_EDGE = 'EDGE_RARE'  # תת-תיקייה 2: שגיאה בקצוות (חשוב לבדוק!)

# כמה אחוז מהתמונה נחשב "מרכז"? (0.25 אומר שמשאירים רבע שוליים מכל צד)
CENTER_MARGIN = 0.25


# ==========================================

def load_calibration_data(filename):
    if not os.path.exists(filename):
        print(f"[ERROR] File '{filename}' not found.")
        return None, None, None

    with open(filename, 'r') as f:
        data = json.load(f)

    K = np.array(data['K'])
    D = np.array(data['D'])
    img_shape = tuple(data['img_shape'])
    return K, D, img_shape


def get_centroid(corners):
    """ מחשב את מרכז המסה של הנקודות """
    pts = corners.reshape(-1, 2)
    mean_x = np.mean(pts[:, 0])
    mean_y = np.mean(pts[:, 1])
    return mean_x, mean_y


def is_in_center(cx, cy, img_w, img_h):
    """ בודק אם המרכז נופל בתיבה הפנימית של התמונה """
    x_min = img_w * CENTER_MARGIN
    x_max = img_w * (1 - CENTER_MARGIN)
    y_min = img_h * CENTER_MARGIN
    y_max = img_h * (1 - CENTER_MARGIN)

    return (x_min < cx < x_max) and (y_min < cy < y_max)


def main():
    # 1. טעינת המודל
    K, D, expected_shape = load_calibration_data(CALIB_FILE)
    if K is None: return

    img_w, img_h = expected_shape

    # 2. הכנת תיקיות מחיקה מפוצלות
    path_root = os.path.join(IMAGE_FOLDER, DELETE_ROOT)
    path_center = os.path.join(path_root, DIR_CENTER)
    path_edge = os.path.join(path_root, DIR_EDGE)

    for p in [path_root, path_center, path_edge]:
        if not os.path.exists(p):
            os.makedirs(p)

    # 3. הכנת אובייקט תלת-ממדי
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # ==========================================
    #       טעינת מטמון (CACHE)
    # ==========================================
    corners_cache = {}
    if os.path.exists(CACHE_FILE):
        print(f"[Info] Loading cache: '{CACHE_FILE}'")
        try:
            with open(CACHE_FILE, 'rb') as f:
                raw_cache = pickle.load(f)
            for key, val in raw_cache.items():
                if isinstance(val, np.ndarray):
                    corners_cache[key] = {'corners': val, 'last_error': None}
                else:
                    corners_cache[key] = val
            print(f"[Info] Loaded {len(corners_cache)} entries.")
        except:
            corners_cache = {}

    # 4. איסוף תמונות
    images = glob.glob(os.path.join(IMAGE_FOLDER, '*.jpg')) + \
             glob.glob(os.path.join(IMAGE_FOLDER, '*.png'))

    if not images: return
    print(f"Scanning {len(images)} images...")

    valid_objpoints = []
    valid_imgpoints = []
    valid_filenames = []
    cache_needs_saving = False

    sb_flags = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY + cv2.CALIB_CB_NORMALIZE_IMAGE

    for fname in images:
        if DELETE_ROOT in fname: continue

        fname_key = os.path.basename(fname)
        corners = None
        ret = False

        if fname_key in corners_cache:
            corners = corners_cache[fname_key]['corners']
            ret = True
        else:
            img = cv2.imread(fname)
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if gray.shape[::-1] != expected_shape: continue

            ret, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, sb_flags)
            if ret:
                corners_cache[fname_key] = {'corners': corners, 'last_error': None}
                cache_needs_saving = True
                print(f"[New] {fname_key}")

        if ret:
            valid_objpoints.append(objp)
            valid_imgpoints.append(corners)
            valid_filenames.append(fname)

    if cache_needs_saving:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(corners_cache, f)

    if not valid_objpoints:
        print("No valid corners found.")
        return

    # 5. חישוב שגיאות
    flags = cv2.fisheye.CALIB_USE_INTRINSIC_GUESS | cv2.fisheye.CALIB_FIX_INTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    N_OK = len(valid_objpoints)
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

    print("Calculating errors...")
    cv2.fisheye.calibrate(valid_objpoints, valid_imgpoints, expected_shape, K, D, rvecs, tvecs, flags,
                          (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

    errors_list = []
    improved = 0
    worsened = 0

    for i in range(N_OK):
        fname_key = os.path.basename(valid_filenames[i])
        proj, _ = cv2.fisheye.projectPoints(valid_objpoints[i], rvecs[i], tvecs[i], K, D)
        proj = proj.reshape(valid_imgpoints[i].shape)
        err = cv2.norm(valid_imgpoints[i], proj, cv2.NORM_L2) / np.sqrt(len(proj))

        prev_err = corners_cache[fname_key].get('last_error')
        if prev_err:
            if err < prev_err - 0.001:
                improved += 1
            elif err > prev_err + 0.001:
                worsened += 1

        corners_cache[fname_key]['last_error'] = err
        errors_list.append({'file': valid_filenames[i], 'error': err, 'corners': valid_imgpoints[i]})

    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(corners_cache, f)

    print(f"\n--- REPORT ---")
    print(f"Improved: {improved} | Worsened: {worsened}")

    # 6. מיון
    errors_list.sort(key=lambda x: x['error'])

    # חישוב הסף לחיתוך
    cut_index = int(len(errors_list) * (1 - WORST_PERCENTAGE))
    files_to_delete = errors_list[cut_index:]

    if not files_to_delete:
        print("No files to delete based on percentage.")
        return

    threshold_val = files_to_delete[0]['error']
    print(f"Deleting {len(files_to_delete)} worst images (Threshold: {threshold_val:.2f} px)")

    # ==========================================
    #       7. ציור היסטוגרמה (התווסף מחדש)
    # ==========================================
    all_errors = [x['error'] for x in errors_list]
    plt.figure(figsize=(10, 6))
    plt.hist(all_errors, bins=40, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(threshold_val, color='red', linestyle='dashed', linewidth=2,
                label=f'Threshold ({threshold_val:.2f} px)')

    plt.title(f'Reprojection Error Histogram (Total: {len(all_errors)} images)')
    plt.xlabel('Error (pixels)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)

    print("Displaying histogram... Close the window to proceed with moving files.")
    plt.show()  # התוכנית תעצור כאן עד שתסגור את החלון
    # ==========================================

    print("\nSorting into folders based on location...")

    moved_center = 0
    moved_edge = 0

    for item in files_to_delete:
        src = item['file']
        filename = os.path.basename(src)

        cx, cy = get_centroid(item['corners'])

        if is_in_center(cx, cy, img_w, img_h):
            dst_folder = path_center
            moved_center += 1
            type_tag = "CENTER"
        else:
            dst_folder = path_edge
            moved_edge += 1
            type_tag = "EDGE/RARE"

        dst = os.path.join(dst_folder, filename)

        try:
            shutil.move(src, dst)
            print(f" [{type_tag}] Error: {item['error']:.2f}px -> {filename}")
        except Exception as e:
            print(f" [Err] {e}")

    print(f"\nDone.")
    print(f"Moved to {DIR_CENTER}: {moved_center}")
    print(f"Moved to {DIR_EDGE}:   {moved_edge} ( <-- Check these! )")


if __name__ == "__main__":
    main()