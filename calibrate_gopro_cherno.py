import numpy as np
import cv2 as cv
import glob
import os
import sys
import json
import matplotlib.pyplot as plt

# ==========================================================
#   הגדרות
# ==========================================================
SQUARES_X = 6
SQUARES_Y = 8
SQUARE_LENGTH = 23.0
MARKER_LENGTH = 22.34
ARUCO_DICT = cv.aruco.DICT_4X4_50


def get_marker_corners_3d():
    obj_points = []
    ids = []
    marker_id = 0
    for y in range(SQUARES_Y):
        for x in range(SQUARES_X):
            if (x + y) % 2 == 1:
                center_x = x * SQUARE_LENGTH + SQUARE_LENGTH / 2
                center_y = y * SQUARE_LENGTH + SQUARE_LENGTH / 2
                half = MARKER_LENGTH / 2
                # Z=0, סדר: TL, TR, BR, BL
                corners = np.array([
                    [center_x - half, center_y - half, 0],
                    [center_x + half, center_y - half, 0],
                    [center_x + half, center_y + half, 0],
                    [center_x - half, center_y + half, 0]
                ], dtype=np.float32)
                obj_points.append(corners)
                ids.append(marker_id)
                marker_id += 1
    return obj_points, np.array(ids)


def run_constrained_calibration():
    root = os.getcwd()
    calib_dir = os.path.join(root, 'images')
    images = glob.glob(os.path.join(calib_dir, '*.jpg')) + \
             glob.glob(os.path.join(calib_dir, '*.png')) + \
             glob.glob(os.path.join(calib_dir, '*.jpeg'))

    if not images:
        print("No images found.")
        return

    print("Generating 3D Model...")
    board_obj_corners_list, board_ids = get_marker_corners_3d()

    aruco_dict = cv.aruco.getPredefinedDictionary(ARUCO_DICT)
    params = cv.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX

    detector = cv.aruco.ArucoDetector(aruco_dict, params)

    all_img_points = []
    all_obj_points = []
    img_shape = None

    print(f"Scanning {len(images)} images...")

    for i, path in enumerate(images):
        img = cv.imread(path)
        if img is None: continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if img_shape is None: img_shape = gray.shape[::-1]

        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 6:
            frame_obj = []
            frame_img = []
            flat_ids = ids.flatten()

            for idx, marker_id in enumerate(flat_ids):
                match = np.where(board_ids == marker_id)[0]
                if len(match) > 0:
                    frame_obj.append(board_obj_corners_list[match[0]])
                    frame_img.append(corners[idx])

            if len(frame_obj) > 4:
                # המרה למבנה שטוח ש-calibrateCamera אוהב
                current_img = np.concatenate(frame_img, axis=0).reshape(-1, 2).astype(np.float32)
                current_obj = np.concatenate(frame_obj, axis=0).reshape(-1, 3).astype(np.float32)

                all_img_points.append(current_img)
                all_obj_points.append(current_obj)
                sys.stdout.write(".")
            else:
                sys.stdout.write("x")
        else:
            sys.stdout.write("x")
        sys.stdout.flush()

    print(f"\nCollected data from {len(all_img_points)} valid frames.")

    if len(all_img_points) < 3:
        print("Not enough data.")
        return

    # === שלב 1: ניחוש ראשוני ===
    print("\n[Step 1] Initializing Camera Matrix...")
    camera_matrix_init = cv.initCameraMatrix2D(all_obj_points, all_img_points, img_shape)
    print("Initial Guess:\n", camera_matrix_init)

    # === שלב 2: כיול מרוסן (Constrained Calibration) ===
    print("\n[Step 2] Running Constrained Calibration...")

    # אלו ה"אזיקים" שאנחנו שמים על האלגוריתם:
    flags = (
            cv.CALIB_USE_INTRINSIC_GUESS |  # תתחיל מהניחוש הטוב שמצאנו
            cv.CALIB_FIX_ASPECT_RATIO |  # אל תעוות את הפיקסלים (fx=fy)
            cv.CALIB_FIX_PRINCIPAL_POINT |  # אל תזיז את המרכז
            cv.CALIB_ZERO_TANGENT_DIST |  # אין עיוות הרכבה (פחות משתנים לחישוב)
            cv.CALIB_RATIONAL_MODEL  # כן תשתמש במודל מתקדם לעדשה רחבה (K4, K5, K6)
    )

    try:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            all_obj_points, all_img_points, img_shape, camera_matrix_init, None, flags=flags
        )

        print(f"\n=== SUCCESS ===")
        print(f"Reprojection Error: {ret:.4f} px")
        print("Camera Matrix:\n", mtx)
        print("Dist Coeffs:\n", dist.ravel())

        # שמירה
        data = {
            "camera_matrix": mtx.tolist(),
            "dist_coeff": dist.tolist(),
            "error": ret,
            "resolution": img_shape
        }
        with open("calibration_constrained.json", "w") as f:
            json.dump(data, f, indent=4)
            print("Saved to calibration_constrained.json")

        # גרף כיסוי
        all_x, all_y = [], []
        for p in all_img_points:
            all_x.extend(p[:, 0])
            all_y.extend(p[:, 1])

        plt.figure(figsize=(10, 6))
        plt.scatter(all_x, all_y, s=1, alpha=0.5)
        plt.title(f"Final Coverage (Err: {ret:.2f}px)")
        plt.gca().invert_yaxis()
        plt.show()

    except Exception as e:
        print(f"\n[ERROR] Calibration Failed: {e}")


if __name__ == "__main__":
    run_constrained_calibration()