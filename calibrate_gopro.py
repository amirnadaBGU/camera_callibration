import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt
import json


def plot_coverage_matplotlib(imgPtsList, imgShape):
    """
    פונקציה עזר: מציירת את מפת הכיסוי עם צירים וסקאלה מדויקת
    """
    width = imgShape[0]
    height = imgShape[1]

    # איסוף כל נקודות ה-X וה-Y לרשימה אחת שטוחה לצורך הציור
    all_x = []
    all_y = []

    for corners in imgPtsList:
        # המרת המערך המורכב לרשימה פשוטה של קואורדינטות
        x_coords = corners[:, 0, 0]
        y_coords = corners[:, 0, 1]
        all_x.extend(x_coords)
        all_y.extend(y_coords)

    # יצירת הגרף
    plt.figure(figsize=(12, 7))

    # 1. ציור מסגרת החיישן (הגבולות האמיתיים של המצלמה) - קו מקווקו אדום
    # סוגר מלבן: (0,0) -> (width,0) -> (width,height) -> (0,height) -> (0,0)
    plt.plot([0, width, width, 0, 0], [0, 0, height, height, 0], 'r--', linewidth=2,
             label='Sensor Borders (Full Frame)')

    # 2. ציור הנקודות שנאספו - נקודות ירוקות
    plt.scatter(all_x, all_y, s=2, c='green', alpha=0.6, label='Detected Corners')

    # הגדרות תצוגה
    plt.xlim(-100, width + 100)  # קצת מרווח בצדדים כדי לראות את המסגרת
    plt.ylim(height + 100, -100)  # הופך את ציר Y (כי בתמונות 0 זה למעלה)

    plt.title(f'Calibration Coverage Map ({width}x{height})')
    plt.xlabel('Width (Pixels)')
    plt.ylabel('Height (Pixels)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)

    # הצגה למשתמש
    print("Displaying Coverage Map... Close the graph window to continue calibration.")
    plt.show()


def calibrate(showPics=True):
    # --- הגדרות נתיב ---
    root = os.getcwd()
    # הערה: וודא שזה הנתיב הנכון לתמונות שלך!
    calibrationDir = os.path.join(root, 'demoImages', 'calibration')

    # חיפוש תמונות
    imgPathList = glob.glob(os.path.join(calibrationDir, '*.jpg')) + \
                  glob.glob(os.path.join(calibrationDir, '*.png'))

    print(f"Found {len(imgPathList)} images.")

    # --- הגדרות הלוח ---
    nRows = 9
    nCols = 6
    squareSize = 15  # מ"מ

    # קריטריונים לדיוק
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # הכנת נקודות העולם
    worldPtsCur = np.zeros((nRows * nCols, 3), np.float32)
    worldPtsCur[:, :2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1, 2)
    worldPtsCur = worldPtsCur * squareSize

    worldPtsList = []
    imgPtsList = []

    imgShape = None

    # --- שלב 1: איסוף הנקודות ---
    print("Scanning images for corners...")
    for curImgPath in imgPathList:
        imgBGR = cv.imread(curImgPath)
        if imgBGR is None:
            continue

        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
        imgShape = imgGray.shape[::-1]

        flags_find = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK
        cornersFound, cornersOrg = cv.findChessboardCorners(imgGray, (nRows, nCols), flags_find)

        if cornersFound == True:
            # print(f". Found in {os.path.basename(curImgPath)}") # אפשר להחזיר אם רוצים פירוט
            worldPtsList.append(worldPtsCur)
            cornersRefined = cv.cornerSubPix(imgGray, cornersOrg, (11, 11), (-1, -1), termCriteria)
            imgPtsList.append(cornersRefined)

            if showPics:
                cv.drawChessboardCorners(imgBGR, (nRows, nCols), cornersRefined, cornersFound)
                display_img = cv.resize(imgBGR, (0, 0), fx=0.2, fy=0.2)
                cv.imshow('Scanning...', display_img)
                cv.waitKey(10)
        else:
            print(f"X Corners NOT found in {os.path.basename(curImgPath)}")

    cv.destroyAllWindows()

    if len(worldPtsList) < 10:
        print("Not enough good images for calibration! Need at least 10.")
        return None, None

    # --- שלב 2: יצירת מפת כיסוי (הגרף החדש) ---
    print("\nGenerating Coverage Map Plot...")
    plot_coverage_matplotlib(imgPtsList, imgShape)

    print("\nStarting Calibration (This may take a moment)...")

    # --- שלב 3: כיול ---
    # משתמשים רק בנעילת מרכז, בלי מודל רציונלי כדי למנוע "התפוצצות" בקצוות
    calib_flags = cv.CALIB_FIX_PRINCIPAL_POINT

    repError, camMatrix, distCoeff, rvecs, tvecs = cv.calibrateCamera(
        worldPtsList, imgPtsList, imgShape, None, None, flags=calib_flags, criteria=termCriteria
    )

    print('\n--- Results ---')
    print(f"Reprojection Error: {repError:.4f} pixels (Target: < 1.5)")
    print('\nCamera Matrix:\n', camMatrix)
    print('\nDistortion Coeff:\n', distCoeff.ravel())

    # Save to JSON
    curFolder = os.path.dirname(os.path.abspath(__file__))
    jsonPath = os.path.join(curFolder, 'gopro_calibration.json')
    jsonData = {
        'repError': float(repError),
        'camMatrix': camMatrix.tolist(),
        'distCoeff': distCoeff.tolist(),
        'resolution': imgShape
    }
    with open(jsonPath, 'w') as f:
        json.dump(jsonData, f, indent=4)
        print(f"\nSaved to {jsonPath}")

    return camMatrix, distCoeff


def runCalibration():
    # showPics=False מזרז, המפה תוצג בכל מקרה בסוף
    calibrate(showPics=False)


if __name__ == '__main__':
    runCalibration()