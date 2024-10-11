import torch
print(torch.__version__)  # 顯示 PyTorch 版本
print(torch.cuda.is_available())  # 確認 CUDA 是否可用
print(torch.cuda.current_device())  # 當前使用的 GPU 設備
print(torch.cuda.get_device_name(0))  # 獲取 GPU 名稱
