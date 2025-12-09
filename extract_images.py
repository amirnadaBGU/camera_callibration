import cv2
import os
import glob
import sys  # הוספתי את זה כדי לאפשר הדפסה באותה שורה


def calculate_blur_score(image):
    """
    מחשב את רמת החדות של התמונה באמצעות שונות הלפלסיאן.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score


def extract_frames(video_path, output_folder, time_interval_sec, blur_threshold):
    # יצירת תיקיית הפלט
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"[INFO] Created folder: {output_folder}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        return

    # --- נתונים לחישוב אחוזים ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # סך כל הפריימים בווידאו
    print(f"[INFO] Processing {video_path}")
    print(f"[INFO] FPS: {fps:.2f} | Total Frames: {total_frames}")

    saved_count = 0
    last_saved_time = -time_interval_sec

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # משיכת מספר הפריים הנוכחי
        current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # חישוב הזמן הנוכחי
        current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # --- הדפסת חיווי התקדמות (באותה שורה) ---
        progress_percent = (current_frame_num / total_frames) * 100
        # \r מחזיר את הסמן לתחילת השורה כדי לעדכן במקום לרדת שורה
        sys.stdout.write(
            f"\r[RUNNING] Frame: {current_frame_num}/{total_frames} ({progress_percent:.1f}%) | Saved: {saved_count}")
        sys.stdout.flush()

        # בדיקה אם עבר מספיק זמן
        if current_time_sec - last_saved_time >= time_interval_sec:
            blur_score = calculate_blur_score(frame)

            if blur_score > blur_threshold:
                filename = f"{output_folder}/frame_{saved_count:04d}_{current_time_sec:.2f}s_score_{int(blur_score)}.jpg"
                cv2.imwrite(filename, frame)

                # יורדים שורה (\n) כדי לא לדרוס את שורת הסטטוס עם הודעת השמירה
                print(f"\n[SAVED] {filename} (Blur Score: {blur_score:.2f})")

                last_saved_time = current_time_sec
                saved_count += 1

    cap.release()
    print(f"\n[DONE] Extracted {saved_count} images to '{output_folder}'")


# --- הגדרות משתמש ---

mp4_files = glob.glob("*.mp4")
if not mp4_files:
    print("[ERROR] No .mp4 file found in the current directory.")
else:
    VIDEO_FILE = mp4_files[0]
    OUTPUT_DIR = "images"
    TIME_INTERVAL = 0.33  # כל כמה שניות לדגום
    BLUR_THRESHOLD = 5 # סף החדות

    extract_frames(VIDEO_FILE, OUTPUT_DIR, TIME_INTERVAL, BLUR_THRESHOLD)