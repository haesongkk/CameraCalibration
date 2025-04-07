import cv2
import numpy as np
import os

# --- 설정 ---
video_path = "video.mp4"
output_dir = "output"
frame_save_dir = os.path.join(output_dir, "undistorted_frames_every_10")
os.makedirs(frame_save_dir, exist_ok=True)

checkerboard_size = (10, 7)
frame_interval = 10
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

# 체커보드 3D 좌표 생성
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

# 영상 읽기
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

frame_idx = 0
frame_for_calibration = []

print("🔍 체커보드 코너 검출 중...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % frame_interval == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_cb, corners = cv2.findChessboardCorners(gray, checkerboard_size, flags)
        print(f"[{frame_idx}] Checkerboard found? {ret_cb}")

        if ret_cb:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)

            frame_for_calibration.append(gray.shape[::-1])  # 마지막 해상도 저장용

    frame_idx += 1


cap.release()
# --- 캘리브레이션 수행 ---
print("\n📸 카메라 캘리브레이션 수행 중...")
if len(objpoints) > 0:
    image_size = frame_for_calibration[-1]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]

    print("\n=== Calibration Results ===")
    print(f"Reprojection RMSE: {ret:.6f}")
    print(f"Focal Length (fx, fy): {fx:.6f}, {fy:.6f}")
    print(f"Principal Point (cx, cy): {cx:.6f}, {cy:.6f}")
    print("Camera Matrix:\n", mtx)
    print("Distortion Coefficients:\n", dist.ravel())

    # --- 두 번째 순회: 전체 영상 보정 + 10프레임마다 이미지 저장 ---
    cap = cv2.VideoCapture(video_path)
    video_writer = cv2.VideoWriter(
        os.path.join(output_dir, "undistorted_output_full.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
    
    frame_save_dir = os.path.join(output_dir, "undistorted_frames_every_10")
    original_save_dir = os.path.join(output_dir, "original_frames_every_10")
    os.makedirs(frame_save_dir, exist_ok=True)
    os.makedirs(original_save_dir, exist_ok=True)
    
    frame_idx = 0
    saved_idx = 0
    
    print("\n🎞️ 전체 영상 보정 및 프레임 저장 중...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    
        undistorted = cv2.undistort(frame, mtx, dist)
        video_writer.write(undistorted)
        side_by_side_dir = os.path.join(output_dir, "side_by_side_frames_every_10")
        os.makedirs(side_by_side_dir, exist_ok=True)



        if frame_idx % frame_interval == 0:
            # 🔹 원본 저장
            original_path = os.path.join(original_save_dir, f"frame_{saved_idx:03}.jpg")
            cv2.imwrite(original_path, frame)

            # 🔹 보정본 저장
            undistorted_path = os.path.join(frame_save_dir, f"frame_{saved_idx:03}.jpg")
            cv2.imwrite(undistorted_path, undistorted)

            # 🔹 나란히 비교 이미지 저장
            side_by_side = cv2.hconcat([frame, undistorted])
            side_path = os.path.join(side_by_side_dir, f"frame_{saved_idx:03}.jpg")
            cv2.imwrite(side_path, side_by_side)

            saved_idx += 1
    
    cap.release()
    video_writer.release()

    print(f"\n✅ 보정된 전체 영상: {output_dir}/undistorted_output_full.mp4")
    print(f"🖼️ 10프레임 간격 보정 이미지 저장됨: {frame_save_dir}")

else:
    print("❌ 유효한 체커보드 프레임이 없어서 캘리브레이션 실패.")
