"""
Task 1: 神經網路反向傳播 - 迴歸任務
使用 MSE 損失函數
"""

import numpy as np

# ============================================
# 第一部分：定義激活函數及其導數
# ============================================

class ReLU:
    """
    ReLU 激活函數
    白話：把負數變成0，正數保持不變
    """
    @staticmethod
    def forward(x):
        """前向傳播：計算 ReLU(x)"""
        return np.maximum(0, x)
    
    @staticmethod
    def derivative(x):
        """
        計算導數：ReLU'(x)
        白話：如果 x > 0，導數是 1；否則是 0
        """
        return (x > 0).astype(float)


class Linear:
    """
    Linear 激活函數
    白話：什麼都不做，直接輸出
    """
    @staticmethod
    def forward(x):
        """前向傳播：直接返回 x"""
        return x
    
    @staticmethod
    def derivative(x):
        """
        計算導數：Linear'(x) = 1
        白話：線性函數的斜率永遠是 1
        """
        return np.ones_like(x)


# ============================================
# 第二部分：定義損失函數
# ============================================

class MSELoss:
    """
    均方誤差損失函數 (Mean Squared Error)
    白話：計算「預測值」和「真實值」的差距
    """
    
    @staticmethod
    def get_total_loss(outputs, expects):
        """
        計算總損失
        公式：MSE = 平均[(預測 - 真實)²]
        
        白話：
        1. 算出每個輸出的誤差
        2. 誤差平方（確保都是正數）
        3. 取平均
        """
        # outputs: [O1, O2]
        # expects: [E1, E2]
        diff = outputs - expects  # 算誤差
        squared_diff = diff ** 2   # 誤差平方
        loss = np.mean(squared_diff)  # 取平均
        return loss
    
    @staticmethod
    def get_output_gradients(outputs, expects):
        """
        計算損失對輸出的梯度
        公式：∂Loss/∂O_i = (2/n) * (O_i - E_i)
        
        白話：
        - 如果預測太大，梯度是正的（要減少）
        - 如果預測太小，梯度是負的（要增加）
        """
        n = len(outputs)  # 輸出的數量（這裡是 2）
        gradients = (2.0 / n) * (outputs - expects)
        return gradients


# ============================================
# 第三部分：定義神經網路
# ============================================

class Network:
    """
    三層神經網路
    結構：輸入層 -> 隱藏層(ReLU) -> 中間層(Linear) -> 輸出層(Linear)
    """
    
    def __init__(self):
        """
        初始化所有權重
        白話：設定神經網路的「初始公式」
        """
        
        # ===== 第一層：輸入層 -> 隱藏層 =====
        # 隱藏層有 2 個神經元
        # 從圖上讀取初始權重
        
        # 輸入到隱藏層的權重矩陣 (2x2)
        # 第一行：X1 和 X2 到第一個隱藏神經元的權重
        # 第二行：X1 和 X2 到第二個隱藏神經元的權重
        self.W1 = np.array([
            [0.5, 0.2],     # X1->H1: 0.5, X2->H1: 0.2
            [0.6, -0.6]     # X1->H2: 0.6, X2->H2: -0.6
        ])
        
        # 偏差到隱藏層的權重 (2,)
        self.b1 = np.array([0.3, 0.25])  # Bias->H1: 0.3, Bias->H2: 0.25
        
        
        # ===== 第二層：隱藏層 -> 中間層 =====
        # 中間層有 1 個神經元
        
        # 隱藏層到中間層的權重 (2,)
        self.W2 = np.array([0.8, -0.5])  # H1->M1: 0.8, H2->M1: -0.5
        
        # 偏差到中間層的權重 (標量)
        self.b2 = np.array([0.6])  # Bias->M1: 0.6
        
        
        # ===== 第三層：中間層 -> 輸出層 =====
        # 輸出層有 2 個神經元
        
        # 中間層到輸出層的權重矩陣 (1x2)
        self.W3 = np.array([
            [0.6, -0.3]     # M1->O1: 0.6, M1->O2: -0.3
        ])
        
        # 偏差到輸出層的權重 (2,)
        self.b3 = np.array([0.4, 0.75])  # Bias->O1: 0.4, Bias->O2: 0.75
        
        
        # ===== 用來存放梯度的變數 =====
        # 初始化為 None，會在 backward() 時計算
        self.grad_W1 = None
        self.grad_b1 = None
        self.grad_W2 = None
        self.grad_b2 = None
        self.grad_W3 = None
        self.grad_b3 = None
        
        
        # ===== 用來存放前向傳播的中間結果 =====
        # 白話：記住每一層算出來的值，反向傳播時會用到
        self.inputs = None          # 輸入值
        self.hidden_linear = None   # 隱藏層線性輸出（ReLU之前）
        self.hidden_output = None   # 隱藏層輸出（ReLU之後）
        self.middle_linear = None   # 中間層線性輸出
        self.middle_output = None   # 中間層輸出
        self.output_linear = None   # 輸出層線性輸出
        self.outputs = None         # 最終輸出
    
    
    def forward(self, inputs):
        """
        前向傳播
        白話：根據現有權重，一層一層計算到輸出
        
        參數:
            inputs: 輸入值 [X1, X2]
        
        返回:
            outputs: 輸出值 [O1, O2]
        """
        # 記住輸入（反向傳播時需要）
        self.inputs = inputs
        
        print("\n=== 前向傳播開始 ===")
        print(f"輸入: X1={inputs[0]}, X2={inputs[1]}")
        
        
        # ===== 第一層：計算隱藏層 =====
        print("\n--- 第一層：輸入 -> 隱藏層 ---")
        
        # 線性計算：Z = W^T * X + b
        # 白話：把輸入乘以權重，再加上偏差
        self.hidden_linear = np.dot(self.W1.T, inputs) + self.b1
        print(f"隱藏層線性輸出（ReLU之前）: {self.hidden_linear}")
        
        # 激活函數：ReLU
        # 白話：把負數砍掉變成0
        self.hidden_output = ReLU.forward(self.hidden_linear)
        print(f"隱藏層輸出（ReLU之後）: {self.hidden_output}")
        
        
        # ===== 第二層：計算中間層 =====
        print("\n--- 第二層：隱藏層 -> 中間層 ---")
        
        # 線性計算
        self.middle_linear = np.dot(self.W2, self.hidden_output) + self.b2
        print(f"中間層線性輸出: {self.middle_linear}")
        
        # 激活函數：Linear（不做任何事）
        self.middle_output = Linear.forward(self.middle_linear)
        print(f"中間層輸出: {self.middle_output}")
        
        
        # ===== 第三層：計算輸出層 =====
        print("\n--- 第三層：中間層 -> 輸出層 ---")
        
        # 線性計算
        self.output_linear = np.dot(self.W3.T, self.middle_output) + self.b3
        print(f"輸出層線性輸出: {self.output_linear}")
        
        # 激活函數：Linear
        self.outputs = Linear.forward(self.output_linear)
        print(f"最終輸出: O1={self.outputs[0]:.4f}, O2={self.outputs[1]:.4f}")
        
        return self.outputs
    
    
    def backward(self, output_gradients):
        """
        反向傳播
        白話：從輸出往回推，計算每個權重的梯度（該調整多少）
        
        參數:
            output_gradients: 損失對輸出的梯度 ∂Loss/∂O
        
        重要概念：鏈式法則（Chain Rule）
        梯度的計算就像「傳遞責任」，一層一層往回傳
        """
        print("\n=== 反向傳播開始 ===")
        print(f"輸出層的梯度（損失對輸出的導數）: {output_gradients}")
        
        
        # ===== 第三層：輸出層的權重梯度 =====
        print("\n--- 第三層反向：輸出層 -> 中間層 ---")
        
        # 輸出層使用 Linear 激活，導數是 1
        # ∂Loss/∂output_linear = ∂Loss/∂output × ∂output/∂output_linear
        #                       = output_gradients × 1
        delta_output = output_gradients * Linear.derivative(self.output_linear)
        print(f"輸出層的 delta: {delta_output}")
        
        # 計算 W3 的梯度
        # ∂Loss/∂W3 = delta_output × middle_output
        # 白話：這個權重的梯度 = 這層的誤差信號 × 上一層的輸出
        self.grad_W3 = np.outer(self.middle_output, delta_output)
        print(f"W3 的梯度:\n{self.grad_W3}")
        
        # 計算 b3 的梯度
        # 偏差的梯度就是這層的 delta（因為偏差的輸入永遠是1）
        self.grad_b3 = delta_output
        print(f"b3 的梯度: {self.grad_b3}")
        
        # 把誤差信號傳到中間層
        # ∂Loss/∂middle_output = W3 × delta_output
        # 白話：根據權重的大小，把誤差「分配」到上一層
        delta_middle = np.dot(self.W3, delta_output)
        print(f"傳到中間層的梯度: {delta_middle}")
        
        
        # ===== 第二層：中間層的權重梯度 =====
        print("\n--- 第二層反向：中間層 -> 隱藏層 ---")
        
        # 中間層使用 Linear 激活，導數是 1
        delta_middle = delta_middle * Linear.derivative(self.middle_linear)
        print(f"中間層的 delta: {delta_middle}")
        
        # 計算 W2 的梯度
        self.grad_W2 = delta_middle * self.hidden_output
        print(f"W2 的梯度: {self.grad_W2}")
        
        # 計算 b2 的梯度
        self.grad_b2 = delta_middle
        print(f"b2 的梯度: {self.grad_b2}")
        
        # 把誤差信號傳到隱藏層
        delta_hidden = self.W2 * delta_middle
        print(f"傳到隱藏層的梯度: {delta_hidden}")
        
        
        # ===== 第一層：隱藏層的權重梯度 =====
        print("\n--- 第一層反向：隱藏層 -> 輸入層 ---")
        
        # 隱藏層使用 ReLU 激活
        # 重要：如果某個神經元在前向傳播時輸出是 0（沒被激活），
        #       那它的梯度也是 0（沒責任）
        delta_hidden = delta_hidden * ReLU.derivative(self.hidden_linear)
        print(f"隱藏層的 delta（考慮ReLU）: {delta_hidden}")
        
        # 計算 W1 的梯度
        # W1 是 2x2 矩陣，要對每個元素計算梯度
        self.grad_W1 = np.outer(self.inputs, delta_hidden)
        print(f"W1 的梯度:\n{self.grad_W1}")
        
        # 計算 b1 的梯度
        self.grad_b1 = delta_hidden
        print(f"b1 的梯度: {self.grad_b1}")
        
        print("\n=== 反向傳播完成 ===")
    
    
    def zero_grad(self, learning_rate):
        """
        使用梯度更新權重
        白話：根據計算出的梯度，調整每個權重
        
        公式：新權重 = 舊權重 - (學習率 × 梯度)
        
        為什麼是「減去」？
        - 梯度指向「損失增加」的方向
        - 我們要減少損失，所以要往相反方向走
        - 學習率控制「步伐大小」
        """
        print(f"\n=== 更新權重（學習率 = {learning_rate}）===")
        
        print("\n舊權重:")
        print(f"W1:\n{self.W1}")
        print(f"b1: {self.b1}")
        print(f"W2: {self.W2}")
        print(f"b2: {self.b2}")
        print(f"W3:\n{self.W3}")
        print(f"b3: {self.b3}")
        
        # 更新所有權重
        self.W1 = self.W1 - learning_rate * self.grad_W1
        self.b1 = self.b1 - learning_rate * self.grad_b1
        self.W2 = self.W2 - learning_rate * self.grad_W2
        self.b2 = self.b2 - learning_rate * self.grad_b2
        self.W3 = self.W3 - learning_rate * self.grad_W3
        self.b3 = self.b3 - learning_rate * self.grad_b3
        
        print("\n新權重:")
        print(f"W1:\n{self.W1}")
        print(f"b1: {self.b1}")
        print(f"W2: {self.W2}")
        print(f"b2: {self.b2}")
        print(f"W3:\n{self.W3}")
        print(f"b3: {self.b3}")
    
    
    def print_weights(self):
        """列印所有權重（方便檢查）"""
        print("\n" + "="*50)
        print("當前神經網路的所有權重:")
        print("="*50)
        print(f"\n第一層權重 W1 (輸入->隱藏):\n{self.W1}")
        print(f"\n第一層偏差 b1:\n{self.b1}")
        print(f"\n第二層權重 W2 (隱藏->中間):\n{self.W2}")
        print(f"\n第二層偏差 b2:\n{self.b2}")
        print(f"\n第三層權重 W3 (中間->輸出):\n{self.W3}")
        print(f"\n第三層偏差 b3:\n{self.b3}")
        print("="*50)


# ============================================
# 第四部分：Task 1-1 實現
# ============================================

def task_1_1():
    """
    Task 1-1: 單次訓練
    白話：
    1. 建立神經網路
    2. 用初始權重算一次
    3. 看看錯多少
    4. 用反向傳播算出該怎麼調整
    5. 調整一次權重
    """
    print("\n" + "="*70)
    print("Task 1-1: 單次訓練")
    print("="*70)
    
    # 初始化神經網路
    nn = Network()
    
    # 設定輸入和期望輸出
    inputs = np.array([1.5, 0.5])
    expects = np.array([0.8, 1.0])
    
    # 設定損失函數和學習率
    loss_fn = MSELoss()
    learning_rate = 0.01
    
    print(f"\n輸入: X1={inputs[0]}, X2={inputs[1]}")
    print(f"期望輸出: E1={expects[0]}, E2={expects[1]}")
    print(f"學習率: {learning_rate}")
    
    # 列印初始權重
    print("\n初始權重:")
    nn.print_weights()
    
    # 前向傳播
    outputs = nn.forward(inputs)
    
    # 計算損失
    loss = loss_fn.get_total_loss(outputs, expects)
    print(f"\n總損失（MSE）: {loss:.6f}")
    print(f"白話：預測值和真實值的平均誤差平方是 {loss:.6f}")
    
    # 計算輸出梯度
    output_gradients = loss_fn.get_output_gradients(outputs, expects)
    
    # 反向傳播
    nn.backward(output_gradients)
    
    # 更新權重
    nn.zero_grad(learning_rate)
    
    # 列印更新後的權重
    print("\n更新後的權重:")
    nn.print_weights()
    
    print("\n" + "="*70)
    print("Task 1-1 完成！")
    print("="*70)


# ============================================
# 第五部分：Task 1-2 實現
# ============================================

def task_1_2():
    """
    Task 1-2: 重複訓練 1000 次
    白話：
    - 重複做 1000 次「算答案 -> 看錯多少 -> 調整權重」
    - 每次都會越來越準確
    - 最後損失應該接近 0
    """
    print("\n" + "="*70)
    print("Task 1-2: 重複訓練 1000 次")
    print("="*70)
    
    # 初始化神經網路
    nn = Network()
    
    # 設定輸入和期望輸出
    inputs = np.array([1.5, 0.5])
    expects = np.array([0.8, 1.0])
    
    # 設定損失函數和學習率
    loss_fn = MSELoss()
    learning_rate = 0.01
    
    print(f"\n輸入: X1={inputs[0]}, X2={inputs[1]}")
    print(f"期望輸出: E1={expects[0]}, E2={expects[1]}")
    print(f"學習率: {learning_rate}")
    print(f"訓練次數: 1000 次")
    
    print("\n開始訓練...")
    print("-"*70)
    
    # 訓練 1000 次
    for i in range(1000):
        # 前向傳播（不印出詳細過程）
        outputs = nn.forward(inputs)
        
        # 計算損失
        loss = loss_fn.get_total_loss(outputs, expects)
        
        # 每 100 次印一次，或是最後 10 次每次都印
        if i % 100 == 0 or i >= 990:
            print(f"第 {i+1:4d} 次訓練 | 損失 = {loss:.8f} | "
                  f"輸出 O1={outputs[0]:.4f}, O2={outputs[1]:.4f}")
        
        # 計算梯度
        output_gradients = loss_fn.get_output_gradients(outputs, expects)
        
        # 反向傳播（不印出詳細過程）
        nn.backward(output_gradients)
        
        # 更新權重（不印出詳細過程）
        nn.zero_grad(learning_rate)
    
    print("-"*70)
    print(f"\n訓練完成！")
    print(f"最終損失: {loss:.10f}")
    print(f"最終輸出: O1={outputs[0]:.6f}, O2={outputs[1]:.6f}")
    print(f"期望輸出: E1={expects[0]}, E2={expects[1]}")
    print(f"\n是否成功？ {'✓ 是的！損失接近 0' if loss < 0.0001 else '✗ 還需要更多訓練'}")
    
    # 列印最終權重
    nn.print_weights()
    
    print("\n" + "="*70)
    print("Task 1-2 完成！")
    print("="*70)


# ============================================
# 主程式
# ============================================

if __name__ == "__main__":
    print("\n")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║                                                                   ║")
    print("║          神經網路反向傳播 - Task 1 迴歸任務                        ║")
    print("║                                                                   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    
    # 執行 Task 1-1
    task_1_1()
    
    # 暫停一下
    print("\n\n按 Enter 繼續執行 Task 1-2...")
    input()
    
    # 執行 Task 1-2
    task_1_2()
    
    print("\n\n所有任務完成！🎉")
    