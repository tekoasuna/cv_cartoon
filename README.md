# cv_cartoon
project demo

# 实时卡通/手绘风格滤镜

## 功能
将笔记本摄像头捕捉的画面实时处理成卡通/手绘风格，效果包括：
- **双边滤波**：平滑图像同时保留边缘
- **自适应阈值边缘提取**：生成勾线效果
- **K-Means 色彩量化**：色块填充，动画风格

## 环境
- Python 环境：`cv_inv`
- 依赖：OpenCV, NumPy, scikit-learn

## 安装依赖
```bash
conda activate cv_inv
pip install -r requirements.txt
```

## 运行
```bash
conda activate cv_inv
python cartoon_filter.py
```

## 按键控制
| 按键 | 功能 |
|------|------|
| `Q` / `Esc` | 退出程序 |
| `S` | 保存当前帧截图 |
| `+` / `=` | 增加色彩数量（更多细节） |
| `-` | 减少色彩数量（更简约） |
| `W` | 边缘更敏感 (线条更多) |
| `X` | 边缘更迟钝 (线条更少) |
| `F` | 开关火焰特效 |

## 手势触发火焰特效
**手心朝上 + 五指张开 + 指尖朝上** → 手掌中心出现火焰动画

## 性能优化
- **抽帧处理**：默认每3帧处理一次，可通过 `FRAME_SKIP` 参数调整
- **分辨率缩放**：显示尺寸缩放至70%，可通过 `WINDOW_SCALE` 参数调整
- **MiniBatchKMeans**：使用快速聚类算法，适合实时处理

## 参数调优
在 `cartoon_filter.py` 中可调整：
- `FRAME_SKIP`: 帧跳过数（越大越流畅，但更新越慢）
- `K_COLORS`: 色彩聚类数（越少越卡通，越多越真实）
- `EDGE_BLOCK`: 边缘检测块大小（影响线条粗细）
- `EDGE_C`: 边缘检测常数（影响线条强度）
