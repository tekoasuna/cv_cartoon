import sys
import traceback

print("测试1: import cv2...")
import cv2
print("  OK")

print("测试2: import mediapipe...")
import mediapipe as mp
print("  OK")

print("测试3: 打开摄像头...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("  失败：无法打开摄像头")
    sys.exit(1)
print("  OK")

print("测试4: 读取一帧...")
ret, frame = cap.read()
if not ret:
    print("  失败：无法读取帧")
    sys.exit(1)
print(f"  OK, shape={frame.shape}")

print("测试5: cv2.imshow...")
cv2.imshow("Test", frame)
print("  OK, 等待按键...")

key = cv2.waitKey(5000) & 0xFF
print(f"  按键: {key}")

cap.release()
cv2.destroyAllWindows()

print("\n所有测试通过！")
