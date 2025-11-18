# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 15:14:38 2025

@author: 寻风
"""

import numpy as np
import matplotlib.pyplot as plt

def load_raw_file(filename, shape):
    """加载 RAW 文件"""
    data = np.fromfile(filename, dtype=np.float32)
    return data.reshape(shape[::-1]).T  # 转置以正确显示

# 加载数据
phantom = load_raw_file('phantom.raw', (256, 256))
sinogram = load_raw_file('sinogram.raw', (384, 180))  # 探测器×角度
reconstruction = load_raw_file('reconstruction.raw', (256, 256))

print(f"Phantom shape: {phantom.shape}")
print(f"Sinogram shape: {sinogram.shape}")
print(f"Reconstruction shape: {reconstruction.shape}")

# 创建画布
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. 显示仿体
im1 = axes[0,0].imshow(phantom, cmap='gray')
axes[0,0].set_title('Original Phantom')
axes[0,0].set_xlabel('X')
axes[0,0].set_ylabel('Y')
plt.colorbar(im1, ax=axes[0,0])

# 2. 显示正弦图
im2 = axes[0,1].imshow(sinogram, cmap='gray', aspect='auto', 
                      extent=[0, 180, 384, 0])
axes[0,1].set_title('Sinogram')
axes[0,1].set_xlabel('Angle (degrees)')
axes[0,1].set_ylabel('Detector Index')
plt.colorbar(im2, ax=axes[0,1])

# 3. 显示重建结果
im3 = axes[0,2].imshow(reconstruction, cmap='gray')
axes[0,2].set_title('FBP Reconstruction')
axes[0,2].set_xlabel('X')
axes[0,2].set_ylabel('Y')
plt.colorbar(im3, ax=axes[0,2])

# 4. 显示水平剖面
y_center = phantom.shape[0] // 2
axes[1,0].plot(phantom[y_center, :], 'b-', label='Original', linewidth=2)
axes[1,0].plot(reconstruction[y_center, :], 'r--', label='Reconstructed', linewidth=2)
axes[1,0].set_title('Horizontal Profile (Y=' + str(y_center) + ')')
axes[1,0].set_xlabel('X')
axes[1,0].set_ylabel('Intensity')
axes[1,0].legend()
axes[1,0].grid(True)

# 5. 显示垂直剖面
x_center = phantom.shape[1] // 2
axes[1,1].plot(phantom[:, x_center], 'b-', label='Original', linewidth=2)
axes[1,1].plot(reconstruction[:, x_center], 'r--', label='Reconstructed', linewidth=2)
axes[1,1].set_title('Vertical Profile (X=' + str(x_center) + ')')
axes[1,1].set_xlabel('Y')
axes[1,1].set_ylabel('Intensity')
axes[1,1].legend()
axes[1,1].grid(True)

# 6. 显示误差图
error = np.abs(reconstruction - phantom)
im6 = axes[1,2].imshow(error, cmap='hot')
axes[1,2].set_title('Absolute Error')
axes[1,2].set_xlabel('X')
axes[1,2].set_ylabel('Y')
plt.colorbar(im6, ax=axes[1,2])

plt.tight_layout()
plt.savefig('fbp_results.png', dpi=300, bbox_inches='tight')
plt.show()

# 计算重建质量
mse = np.mean((reconstruction - phantom)**2)
rmse = np.sqrt(mse)
max_val = np.max(phantom)
nrmse = rmse / max_val

print(f"\n重建质量指标:")
print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"NRMSE: {nrmse:.6f}")