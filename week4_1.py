"""
WeHelp Assignment - Week 4
Neural Network with Activation and Loss Functions

題目要求：
1. 實現 forward() 方法計算輸出
2. 實現損失函數計算總損失
3. 完成四種網絡任務：回歸、二元分類、多標籤分類、多類別分類
"""

import numpy as np

# ============================================================
# 激活函數定義
# ============================================================

def linear(x):
    """Linear activation: f(x) = x"""
    return x

def relu(x):
    """ReLU activation: f(x) = max(0, x)"""
    return np.maximum(0, x)

def sigmoid(x):
    """Sigmoid activation: f(x) = 1/(1 + e^(-x))"""
    return 1 / (1 + np.exp(-x))

def softmax(x):
    """
    Softmax activation with numerical stability
    f(x_i) = e^(x_i - max(x)) / Σe^(x_k - max(x))
    """
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

# ============================================================
# 損失函數定義
# ============================================================

def mse_loss(expected, output):
    """
    Mean Squared Error Loss
    MSE = 1/n * Σ(E_i - O_i)²
    """
    n = len(output)
    return (1/n) * np.sum((expected - output) ** 2)

def binary_cross_entropy_loss(expected, output):
    """
    Binary Cross Entropy Loss
    BCE = -Σ(E_i * ln(O_i) + (1-E_i) * ln(1-O_i))
    """
    # 添加小數值避免 log(0)
    epsilon = 1e-15
    output = np.clip(output, epsilon, 1 - epsilon)
    return -np.sum(expected * np.log(output) + (1 - expected) * np.log(1 - output))

def categorical_cross_entropy_loss(expected, output):
    """
    Categorical Cross Entropy Loss
    CCE = -ΣE_i * ln(O_i)
    """
    epsilon = 1e-15
    output = np.clip(output, epsilon, 1)
    return -np.sum(expected * np.log(output))

# ============================================================
# 神經網絡類別
# ============================================================

class NeuralNetwork:
    """通用神經網絡類別"""
    
    def __init__(self, weights_input_hidden, bias_hidden, 
                 weights_hidden_output, bias_output,
                 hidden_activation, output_activation):
        """
        初始化網絡參數
        
        參數:
        - weights_input_hidden: 輸入層到隱藏層的權重矩陣
        - bias_hidden: 隱藏層偏置
        - weights_hidden_output: 隱藏層到輸出層的權重矩陣
        - bias_output: 輸出層偏置
        - hidden_activation: 隱藏層激活函數
        - output_activation: 輸出層激活函數
        """
        self.W1 = np.array(weights_input_hidden)
        self.b1 = np.array(bias_hidden)
        self.W2 = np.array(weights_hidden_output)
        self.b2 = np.array(bias_output)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
    
    def forward(self, inputs):
        """
        前向傳播計算輸出
        
        參數:
        - inputs: 輸入向量
        
        返回:
        - output: 網絡輸出
        """
        # 輸入層到隱藏層
        X = np.array(inputs)
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.hidden_activation(z1)
        
        # 隱藏層到輸出層
        z2 = np.dot(a1, self.W2) + self.b2
        output = self.output_activation(z2)
        
        return output


# ============================================================
# 任務 1: 回歸任務 (Regression)
# ============================================================

print("=" * 60)
print("任務 1: 神經網絡回歸任務")
print("=" * 60)
print("網絡結構: 2輸入 → 2隱藏層(ReLU) → 2輸出(Linear)")
print("損失函數: MSE (Mean Squared Error)")
print()

# 網絡權重 
weights_ih_regression = [
    [0.5, 0.6],    # X1 → [H1, H2]
    [0.2, -0.6]    # X2 → [H1, H2]
]
bias_h_regression = [0.3, 0.25]

weights_ho_regression = [
    [0.8, 0.4],    # H1 → [O1, O2]
    [-0.5, 0.5]    # H2 → [O1, O2]
]
bias_o_regression = [0.6, -0.25]

# 創建回歸網絡
nn_regression = NeuralNetwork(
    weights_ih_regression, bias_h_regression,
    weights_ho_regression, bias_o_regression,
    relu, linear
)

# 測試案例 1
print("測試 1:")
inputs1 = [1.5, 0.5]
expects1 = np.array([0.8, 1.0])
outputs1 = nn_regression.forward(inputs1)
loss1 = mse_loss(expects1, outputs1)
print(f"輸入 (X1, X2) = {inputs1}")
print(f"預期輸出 (E1, E2) = {expects1}")
print(f"實際輸出 (O1, O2) = {outputs1}")
print(f"Total Loss: {loss1}")
print()

# 測試案例 2
print("測試 2:")
inputs2 = [0, 1]
expects2 = np.array([0.5, 0.5])
outputs2 = nn_regression.forward(inputs2)
loss2 = mse_loss(expects2, outputs2)
print(f"輸入 (X1, X2) = {inputs2}")
print(f"預期輸出 (E1, E2) = {expects2}")
print(f"實際輸出 (O1, O2) = {outputs2}")
print(f"Total Loss: {loss2}")
print()


# ============================================================
# 任務 2: 二元分類任務 (Binary Classification)
# ============================================================

print("=" * 60)
print("任務 2: 神經網絡二元分類任務")
print("=" * 60)
print("網絡結構: 2輸入 → 2隱藏層(ReLU) → 1輸出(Sigmoid)")
print("損失函數: Binary Cross Entropy")
print()

# 網絡權重
weights_ih_binary = [
    [0.5, 0.6],
    [0.2, -0.6]
]
bias_h_binary = [0.3, 0.25]

weights_ho_binary = [
    [0.8],    # H1 → O1
    [0.4]     # H2 → O1
]
bias_o_binary = [-0.5]

# 創建二元分類網絡
nn_binary = NeuralNetwork(
    weights_ih_binary, bias_h_binary,
    weights_ho_binary, bias_o_binary,
    relu, sigmoid
)

# 測試案例 1
print("測試 1:")
inputs1 = [0.75, 1.25]
expects1 = np.array([1])
outputs1 = nn_binary.forward(inputs1)
loss1 = binary_cross_entropy_loss(expects1, outputs1)
print(f"輸入 (X1, X2) = {inputs1}")
print(f"預期輸出 (E) = {expects1}")
print(f"實際輸出 (O) = {outputs1}")
print(f"Total Loss: {loss1}")
print()

# 測試案例 2
print("測試 2:")
inputs2 = [-1, 0.5]
expects2 = np.array([0])
outputs2 = nn_binary.forward(inputs2)
loss2 = binary_cross_entropy_loss(expects2, outputs2)
print(f"輸入 (X1, X2) = {inputs2}")
print(f"預期輸出 (E) = {expects2}")
print(f"實際輸出 (O) = {outputs2}")
print(f"Total Loss: {loss2}")
print()


# ============================================================
# 任務 3: 多標籤分類任務 (Multi-Label Classification)
# ============================================================

print("=" * 60)
print("任務 3: 神經網絡多標籤分類任務")
print("=" * 60)
print("網絡結構: 2輸入 → 2隱藏層(ReLU) → 3輸出(Sigmoid)")
print("損失函數: Binary Cross Entropy")
print()

# 網絡權重
weights_ih_multilabel = [
    [0.5, 0.6],
    [0.2, -0.6]
]
bias_h_multilabel = [0.3, 0.25]

weights_ho_multilabel = [
    [0.8, 0.5, 0.3],      # H1 → [O1, O2, O3]
    [-0.4, 0.4, 0.75]     # H2 → [O1, O2, O3]
]
bias_o_multilabel = [0.6, 0.5, -0.5]

# 創建多標籤分類網絡
nn_multilabel = NeuralNetwork(
    weights_ih_multilabel, bias_h_multilabel,
    weights_ho_multilabel, bias_o_multilabel,
    relu, sigmoid
)

# 測試案例 1
print("測試 1:")
inputs1 = [1.5, 0.5]
expects1 = np.array([1, 0, 1])
outputs1 = nn_multilabel.forward(inputs1)
loss1 = binary_cross_entropy_loss(expects1, outputs1)
print(f"輸入 (X1, X2) = {inputs1}")
print(f"預期輸出 (E1, E2, E3) = {expects1}")
print(f"實際輸出 (O1, O2, O3) = {outputs1}")
print(f"Total Loss: {loss1}")
print()

# 測試案例 2
print("測試 2:")
inputs2 = [0, 1]
expects2 = np.array([1, 1, 0])
outputs2 = nn_multilabel.forward(inputs2)
loss2 = binary_cross_entropy_loss(expects2, outputs2)
print(f"輸入 (X1, X2) = {inputs2}")
print(f"預期輸出 (E1, E2, E3) = {expects2}")
print(f"實際輸出 (O1, O2, O3) = {outputs2}")
print(f"Total Loss: {loss2}")
print()


# ============================================================
# 任務 4: 多類別分類任務 (Multi-Class Classification)
# ============================================================

print("=" * 60)
print("任務 4: 神經網絡多類別分類任務")
print("=" * 60)
print("網絡結構: 2輸入 → 2隱藏層(ReLU) → 3輸出(Softmax)")
print("損失函數: Categorical Cross Entropy")
print()

# 網絡權重 (與多標籤相同)
weights_ih_multiclass = [
    [0.5, 0.6],
    [0.2, -0.6]
]
bias_h_multiclass = [0.3, 0.25]

weights_ho_multiclass = [
    [0.8, 0.5, 0.3],      # H1 → [O1, O2, O3]
    [-0.4, 0.4, 0.75]     # H2 → [O1, O2, O3]
]
bias_o_multiclass = [0.6, 0.5, -0.5]

# 創建多類別分類網絡
nn_multiclass = NeuralNetwork(
    weights_ih_multiclass, bias_h_multiclass,
    weights_ho_multiclass, bias_o_multiclass,
    relu, softmax
)

# 測試案例 1
print("測試 1:")
inputs1 = [1.5, 0.5]
expects1 = np.array([1, 0, 0])
outputs1 = nn_multiclass.forward(inputs1)
loss1 = categorical_cross_entropy_loss(expects1, outputs1)
print(f"輸入 (X1, X2) = {inputs1}")
print(f"預期輸出 (E1, E2, E3) = {expects1}")
print(f"實際輸出 (O1, O2, O3) = {outputs1}")
print(f"Total Loss: {loss1}")
print()

# 測試案例 2
print("測試 2:")
inputs2 = [0, 1]
expects2 = np.array([0, 0, 1])
outputs2 = nn_multiclass.forward(inputs2)
loss2 = categorical_cross_entropy_loss(expects2, outputs2)
print(f"輸入 (X1, X2) = {inputs2}")
print(f"預期輸出 (E1, E2, E3) = {expects2}")
print(f"實際輸出 (O1, O2, O3) = {outputs2}")
print(f"Total Loss: {loss2}")
print()

print("=" * 60)
print("作業完成！")
print("=" * 60)


