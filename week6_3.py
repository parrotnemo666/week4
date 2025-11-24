import torch

print("=" * 50)
print("任務 1: 從 Python list 建立 tensor")
print("=" * 50)
# 從 Python list 建立 tensor
tensor1 = torch.tensor([[2, 3, 1], [5, -2, 1]])
print(f"Tensor: \n{tensor1}")
print(f"Shape: {tensor1.shape}")
print(f"Dtype: {tensor1.dtype}")

print("\n" + "=" * 50)
print("任務 2: 建立 3x4x2 的隨機 tensor")
print("=" * 50)
# 建立 3x4x2,填充 0~1 隨機浮點數
tensor2 = torch.rand(3, 4, 2)
print(f"Shape: {tensor2.shape}")
print(f"Tensor: \n{tensor2}")

print("\n" + "=" * 50)
print("任務 3: 建立 2x1x5 的全 1 tensor")
print("=" * 50)
# 建立 2x1x5,全部填充 1
tensor3 = torch.ones(2, 1, 5)
print(f"Shape: {tensor3.shape}")
print(f"Tensor: \n{tensor3}")

print("\n" + "=" * 50)
print("任務 4: 矩陣乘法 (Matrix Multiplication)")
print("=" * 50)
# 矩陣乘法
tensor4_a = torch.tensor([[1, 2, 4], [2, 1, 3]])  # shape: (2, 3)
tensor4_b = torch.tensor([[5], [2], [1]])          # shape: (3, 1)
result4 = torch.matmul(tensor4_a, tensor4_b)       # 或用 tensor4_a @ tensor4_b
print(f"Tensor A (2x3): \n{tensor4_a}")
print(f"Tensor B (3x1): \n{tensor4_b}")
print(f"矩陣乘法結果 (2x1): \n{result4}")

print("\n" + "=" * 50)
print("任務 5: 逐元素相乘 (Element-wise Product)")
print("=" * 50)
# Element-wise product (逐元素相乘)
tensor5_a = torch.tensor([[1, 2], [2, 3], [-1, 3]])    # shape: (3, 2)
tensor5_b = torch.tensor([[5, 4], [2, 1], [1, -5]])    # shape: (3, 2)
result5 = tensor5_a * tensor5_b                         # 或用 torch.mul()
print(f"Tensor A (3x2): \n{tensor5_a}")
print(f"Tensor B (3x2): \n{tensor5_b}")
print(f"逐元素相乘結果 (3x2): \n{result5}")