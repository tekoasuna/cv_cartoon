"""
实时卡通/手绘风格滤镜 + 手势火焰特效
"""

import cv2
import numpy as np
import os
import sys
import traceback

os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'


# ========== 卡通效果 ==========

def fast_color_quantize(image, k=8):
    quantized = np.zeros_like(image)
    bin_size = 256 // k
    for i in range(3):
        channel = image[:, :, i].astype(np.float32)
        quantized[:, :, i] = (channel // bin_size * bin_size + bin_size // 2).clip(0, 255).astype(np.uint8)
    return quantized


def cartoon_effect(frame, k_colors=8, canny_low=30, canny_high=100):
    filtered = cv2.bilateralFilter(frame, d=7, sigmaColor=80, sigmaSpace=80)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=canny_low, threshold2=canny_high)
    quantized = fast_color_quantize(filtered, k=k_colors)
    result = quantized.copy()
    result[edges > 0] = 0
    return result


# ========== 火焰动画 ==========

class FlameOverlay:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.frames = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frames.append(frame)
        self.cap.release()
        self.total_frames = len(self.frames)
        self.current_frame = 0

    def get_frame(self):
        frame = self.frames[self.current_frame]
        self.current_frame = (self.current_frame + 1) % self.total_frames
        return frame


# ========== 手势识别 ==========

class HandDetector:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.mp = mp

    def detect(self, frame_bgr):
        results = []
        h, w = frame_bgr.shape[:2]
        mp_image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        )
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.hand_landmarks:
            return results
        
        for landmarks in detection_result.hand_landmarks:
            def lm(idx):
                return landmarks[idx]
            
            wrist = lm(0)
            index_mcp, middle_mcp, ring_mcp, pinky_mcp = lm(5), lm(9), lm(13), lm(17)
            index_tip, middle_tip, ring_tip, pinky_tip = lm(8), lm(12), lm(16), lm(20)
            thumb_tip, thumb_mcp = lm(4), lm(2)
            
            palm_x = (index_mcp.x + middle_mcp.x + ring_mcp.x + pinky_mcp.x) / 4
            palm_y = (index_mcp.y + middle_mcp.y + ring_mcp.y + pinky_mcp.y) / 4
            
            tips = [index_tip, middle_tip, ring_tip, pinky_tip]
            mcps = [index_mcp, middle_mcp, ring_mcp, pinky_mcp]
            finger_up = [t.y < m.y for t, m in zip(tips, mcps)]
            thumb_up = thumb_tip.y < thumb_mcp.y
            fingers_up = [thumb_up] + finger_up
            
            hand_up = sum(finger_up) >= 3
            palm_up = wrist.y > palm_y and hand_up
            
            results.append({
                'palm_center': (int(palm_x * w), int(palm_y * h)),
                'all_fingers_up': all(fingers_up) and hand_up,
            })
        
        return results


# ========== 火焰叠加 ==========

def overlay_flame(cartoon_frame, hand_info, flame_frame, scale=0.35):
    palm = hand_info['palm_center']
    h, w = cartoon_frame.shape[:2]
    fw, fh = int(w * scale), int(h * scale)
    
    offset_y = int(h * 0.15)
    x1 = max(0, palm[0] - fw // 2)
    y1 = max(0, palm[1] - fh // 2 - offset_y)
    x2, y2 = min(w, x1 + fw), min(h, y1 + fh)
    
    flame_r = cv2.resize(flame_frame, (x2 - x1, y2 - y1))
    mask = np.any(flame_r < [240, 240, 240], axis=2).astype(np.float32)[:, :, None]
    
    region = cartoon_frame[y1:y2, x1:x2].astype(np.float32)
    blended = region * (1 - mask) + flame_r.astype(np.float32) * mask
    cartoon_frame[y1:y2, x1:x2] = blended.astype(np.uint8)
    return cartoon_frame


# ========== 主程序 ==========

def main():
    PROC_W, PROC_H = 480, 360
    DISPLAY_SCALE = 1.5
    K_COLORS = 8
    FLAME_VIDEO = r"flame-animation-gif-download-4786522.mp4"
    
    # 下载模型
    model_path = "hand_landmarker.task"
    if not os.path.exists(model_path):
        print("下载手势模型...")
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print("模型下载完成")
    
    print("打开摄像头...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    flame = FlameOverlay(FLAME_VIDEO)
    print("火焰动画已加载")
    
    # 先不初始化 MediaPipe，显示画面后再初始化
    hand_detector = None
    
    display_w = int(PROC_W * DISPLAY_SCALE)
    display_h = int(PROC_H * DISPLAY_SCALE)
    
    canny_low, canny_high = 30, 100
    flame_enabled = True
    frame_count = 0
    mp_initialized = False
    
    print("=" * 50)
    print("按键: Q退出 S截图 +/-色彩 W/X边缘 F火焰开关")
    print("手势: 手心朝上+五指张开 → 火焰特效")
    print("=" * 50)
    sys.stdout.flush()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_proc = cv2.resize(frame, (PROC_W, PROC_H), interpolation=cv2.INTER_AREA)
        result = cartoon_effect(frame_proc, K_COLORS, canny_low, canny_high)
        
        # 延迟初始化 MediaPipe（等画面显示出来后再加载）
        if not mp_initialized and flame_enabled and frame_count > 30:
            print("加载手势检测...")
            sys.stdout.flush()
            try:
                hand_detector = HandDetector()
                print("手势检测已加载")
                sys.stdout.flush()
            except Exception as e:
                print(f"手势检测失败: {e}")
                sys.stdout.flush()
            mp_initialized = True
        
        # 手势+火焰
        if flame_enabled and hand_detector is not None:
            hands = hand_detector.detect(frame_proc)
            for hand in hands:
                if hand['all_fingers_up']:
                    overlay_flame(result, hand, flame.get_frame())
        
        display = cv2.resize(result, (display_w, display_h), interpolation=cv2.INTER_NEAREST)
        status = f"C:{K_COLORS} F:{'ON' if flame_enabled else 'OFF'}"
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        
        cv2.imshow("Cartoon + Flame", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"screenshot_{frame_count}.png", display)
            print(f"截图: {frame_count}")
        elif key == ord('+') or key == ord('='):
            K_COLORS = min(K_COLORS + 2, 16)
        elif key == ord('-'):
            K_COLORS = max(K_COLORS - 2, 4)
        elif key == ord('w'):
            canny_low = max(canny_low - 10, 10)
            canny_high = max(canny_high - 15, 30)
        elif key == ord('x'):
            canny_low = min(canny_low + 10, 100)
            canny_high = min(canny_high + 15, 200)
        elif key == ord('f'):
            flame_enabled = not flame_enabled
            print(f"火焰: {'ON' if flame_enabled else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("已退出")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"错误: {e}")
        traceback.print_exc()
