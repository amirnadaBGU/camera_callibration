import os


def delete_copy_files(folder_name="Images"):
    # בדיקה שהתיקייה קיימת
    if not os.path.exists(folder_name):
        print(f"[Error] The folder '{folder_name}' does not exist.")
        return

    print(f"Scanning folder: {folder_name}...")

    files = os.listdir(folder_name)
    deleted_count = 0

    for filename in files:
        # פיצול השם והסיומת (למשל: 'image - Copy' ו-'.jpg')
        name, ext = os.path.splitext(filename)

        # בדיקה אם השם נגמר בביטוי המבוקש
        if name.endswith(" - Copy"):
            full_path = os.path.join(folder_name, filename)

            try:
                os.remove(full_path)
                print(f"[Deleted] {filename}")
                deleted_count += 1
            except Exception as e:
                print(f"[Error] Could not delete {filename}: {e}")

    if deleted_count == 0:
        print("No files with ' - Copy' found.")
    else:
        print(f"Done. Deleted {deleted_count} files.")


if __name__ == "__main__":
    # הרצת הפונקציה
    delete_copy_files()