import cv2
import os
import glob
import sys


def calculate_blur_score(image):
    """
    מחשב את רמת החדות של התמונה באמצעות שונות הלפלסיאן.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score


def extract_frames(video_path, output_folder, time_interval_sec, blur_threshold):
    # יצירת תיקיית הפלט אם אינה קיימת
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"[INFO] Created folder: {output_folder}")

    # --- שינוי 1: חילוץ שם הוידאו הנקי לשם הקובץ ---
    video_filename = os.path.basename(video_path)  # למשל: video1.mp4
    video_name_clean = os.path.splitext(video_filename)[0]  # למשל: video1
    # -----------------------------------------------

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        return

    # --- נתונים לחישוב אחוזים ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n[INFO] Processing: {video_filename}")
    print(f"[INFO] FPS: {fps:.2f} | Total Frames: {total_frames}")

    saved_count = 0
    last_saved_time = -time_interval_sec

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # --- הדפסת חיווי התקדמות ---
        if total_frames > 0:
            progress_percent = (current_frame_num / total_frames) * 100
        else:
            progress_percent = 0

        sys.stdout.write(
            f"\r[RUNNING] {video_name_clean} | Frame: {current_frame_num}/{total_frames} ({progress_percent:.1f}%) | Saved: {saved_count}")
        sys.stdout.flush()

        # בדיקה אם עבר מספיק זמן
        if current_time_sec - last_saved_time >= time_interval_sec:
            blur_score = calculate_blur_score(frame)

            if blur_score > blur_threshold:
                # --- שינוי 2: הוספת שם הוידאו לשם הקובץ ---
                filename = f"{output_folder}/{video_name_clean}_frame_{saved_count:04d}_{current_time_sec:.2f}s.jpg"
                # ------------------------------------------

                cv2.imwrite(filename, frame)

                # יורדים שורה כדי לא לדרוס את הסטטוס
                print(f"\n[SAVED] {filename} (Blur Score: {blur_score:.2f})")

                last_saved_time = current_time_sec
                saved_count += 1

    cap.release()
    print(f"\n[DONE] Finished {video_filename}. Extracted {saved_count} images.")


# --- הגדרות משתמש ---

# מחפש את כל קבצי ה-MP4 בתיקייה הנוכחית
mp4_files = glob.glob("*.mp4")

if not mp4_files:
    print("[ERROR] No .mp4 file found in the current directory.")
else:
    OUTPUT_DIR = "images"
    TIME_INTERVAL = 0.5  # כל כמה שניות לדגום
    BLUR_THRESHOLD = 5  # סף החדות

    print(f"Found {len(mp4_files)} videos. Starting process...")

    # --- שינוי 3: לולאה שרצה על כל הוידאוים בתיקייה ---
    for video_file in mp4_files:
        extract_frames(video_file, OUTPUT_DIR, TIME_INTERVAL, BLUR_THRESHOLD)

    print("\n[ALL DONE] Processed all videos.")