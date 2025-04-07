import cv2
import numpy as np

# 설정
video_path = "video.mp4"
frame_interval = 10
checkerboard_size = (10, 7)  # 내부 교차점 개수

# 체커보드 실제 월드 좌표 (단위: mm 단위 넣고 싶으면 여기서 곱하면 됨)
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

# 누적 포인트 저장소
objpoints = []  # 3D
imgpoints = []  # 2D

# 체커보드 찾기 파라미터
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

# 영상 읽기
cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_cb, corners = cv2.findChessboardCorners(gray, checkerboard_size, flags)

        print(f"[{frame_count}] Checkerboard found? {ret_cb}")

        if ret_cb:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)

            # 디버깅 시각화 (선택)
            vis = frame.copy()
            cv2.drawChessboardCorners(vis, checkerboard_size, corners2, ret_cb)
            cv2.imshow('Detected', vis)
            cv2.waitKey(50)

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]

    print("\n=== Calibration Results ===")
    print(f"Reprojection RMSE: {ret:.6f}")
    print(f"Focal Length (fx, fy): {fx:.6f}, {fy:.6f}")
    print(f"Principal Point (cx, cy): {cx:.6f}, {cy:.6f}")
    print("Camera Matrix:\n", mtx)
    print("Distortion Coefficients (k1, k2, p1, p2, k3, ...):\n", dist.ravel())
else:
    print("❌ No valid checkerboard frames found.")
