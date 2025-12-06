import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt
import json

def calibrate(showPics=True):
    # Read Image
    root = os.getcwd()
    calibrationDir = os.path.join(root,'demoImages//calibration')
    imgPathList = glob.glob(os.path.join(calibrationDir, '*.jpg')) + glob.glob(os.path.join(calibrationDir, '*.png'))

    print(f"Found {len(imgPathList)} images.")
    # Initialize
    nRows = 9
    nCols = 6
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,30,0.001)
    worldPtsCur = np.zeros((nRows*nCols,3), np.float32)
    worldPtsCur[:,:2] = np.mgrid[0:nRows,0:nCols].T.reshape(-1,2)
    worldPtsList = []
    imgPtsList = []

    # Find Corners
    for curImgPath in imgPathList:
        imgBGR = cv.imread(curImgPath)
        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
        cornersFound, cornersOrg = cv.findChessboardCorners(imgGray,(nRows,nCols), None)

        if cornersFound == True:
            worldPtsList.append(worldPtsCur)
            cornersRefined = cv.cornerSubPix(imgGray,cornersOrg,(11,11),(-1,-1),termCriteria)
            imgPtsList.append(cornersRefined)

            if showPics:
                cv.drawChessboardCorners(imgBGR,(nRows,nCols),cornersRefined,cornersFound)
                cv.imshow('Chessboard', imgBGR)
                cv.waitKey(500)
    cv.destroyAllWindows()

    # Calibrate
    calib_flags = cv.CALIB_FIX_PRINCIPAL_POINT + cv.CALIB_RATIONAL_MODEL
    repError, camMatrix, distCoeff, rvecs, tvecs = cv.calibrateCamera(
        worldPtsList, imgPtsList, imgShape, None, None, flags=calib_flags, criteria=termCriteria
    )
    # Plot each chessboard in a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        R, _ = cv.Rodrigues(rvec)
        points_3d = np.dot(R, worldPtsList[i].T).T + tvec.T
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], label=f'Image {i+1}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('3D Chessboard Positions')
    plt.show()


    print('Camera Matrix:\n',camMatrix)
    print('Distortion Coeff:\n',distCoeff)
    print("Reproj Error (pixels): {:.4f}".format(repError))

    # Save Calibration Parameters (later video)
    curFolder = os.path.dirname(os.path.abspath(__file__))
    paramPath = os.path.join(curFolder,'calibration.npz')
    np.savez(paramPath,
        repError=repError,
        camMatrix=camMatrix,
        distCoeff=distCoeff,
        rvecs=rvecs,
        tvecs=tvecs)

    # Save to JSON
    jsonPath = os.path.join(curFolder, 'calibration.json')
    jsonData = {
        'repError': float(repError),
        'camMatrix': camMatrix.tolist(),
        'distCoeff': distCoeff.tolist(),
        'rvecs': [r.tolist() for r in rvecs],
        'tvecs': [t.tolist() for t in tvecs]
    }
    with open(jsonPath, 'w') as f:
        json.dump(jsonData, f, indent=4)

    return camMatrix,distCoeff

def removeDistortion(camMatrix,distCoeff):
    root = os.getcwd()
    imgPath = os.path.join(root,'demoImages//distortion2.png')
    img = cv.imread(imgPath)

    height,width = img.shape[:2]
    camMatrixNew,roi = cv.getOptimalNewCameraMatrix(camMatrix,distCoeff,(width,height),1,(width,height))
    imgUndist = cv.undistort(img,camMatrix,distCoeff,None,camMatrixNew)

    # Draw Line to See Distortion Change
    cv.line(img,(1769,103),(1780,922),(255,255,255),2)
    cv.line(imgUndist,(1769,103),(1780,922),(255,255,255),2)

    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(imgUndist)
    plt.show()

def runCalibration():
    calibrate(showPics=True)

def runRemoveDistortion():
    camMatrix,distCoeff = calibrate(showPics=False)
    removeDistortion(camMatrix,distCoeff)

if __name__ == '__main__':
    # runCalibration()
    runRemoveDistortion()