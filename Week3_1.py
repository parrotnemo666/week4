class Network:
    def __init__(self, weights, biases):
        """
        通用神經網路類別
        
        參數:
        weights: 權重列表，每個元素是一個二維列表
                例如: [[[w11, w12], [w21, w22]], ...]
                第 i 層的權重矩陣 weights[i]
        biases: 偏置列表，每個元素是一維列表
                例如: [[b1, b2], [b3], ...]
        """
        self.weights = weights
        self.biases = biases
        self.num_layers = len(weights)
    
    def activation(self, x):
        """激活函數 - 線性激活"""
        return x
    
    def forward(self, inputs):
        """
        前向傳播 - 通用版本，適用於任意層數的網路
        
        參數:
        inputs: 輸入向量 [x1, x2, ...]
        
        返回:
        outputs: 輸出向量
        """
        # 當前層的激活值
        activations = inputs
        
        print(f"輸入: {inputs}")
        
        # 逐層計算
        for layer_idx in range(self.num_layers):
            # 取得該層的權重和偏置
            W = self.weights[layer_idx]
            b = self.biases[layer_idx]
            
            # 計算下一層的輸入（加權和 + 偏置）
            next_activations = []
            
            for neuron_idx in range(len(b)):  # 遍歷該層的每個神經元
                # 計算該神經元的加權和
                z = b[neuron_idx]  # 先加偏置
                
                for input_idx in range(len(activations)):
                    z += activations[input_idx] * W[input_idx][neuron_idx]
                
                # 通過激活函數
                a = self.activation(z)
                next_activations.append(a)
            
            # 顯示該層的計算結果
            layer_name = f"隱藏層 {layer_idx + 1}" if layer_idx < self.num_layers - 1 else "輸出層"
            print(f"\n{layer_name}: {next_activations}")
            
            # 更新激活值為下一層的輸入
            activations = next_activations
        
        return activations


def main():
    print("=" * 60)
    print("Neural Network 1 測試")
    print("=" * 60)
    
    # Network 1: 2-2-1 結構
    weights1 = [
        # 輸入層 → 隱藏層 (2個輸入 → 2個神經元)
        [
            [0.5, 0.6],      # X1 的權重 [到H1, 到H2]
            [0.2, -0.6]      # X2 的權重 [到H1, 到H2]
        ],
        # 隱藏層 → 輸出層 (2個輸入 → 1個神經元)
        [
            [0.8],           # H1 的權重 [到O1]
            [0.4]            # H2 的權重 [到O1]
        ]
    ]
    
    biases1 = [
        [0.3, 0.25],         # 隱藏層偏置 [H1, H2]
        [-0.5]               # 輸出層偏置 [O1]
    ]
    
    nn1 = Network(weights1, biases1)
    
    # 測試案例 1
    print("\n【測試案例 1】: (1.5, 0.5)")
    print("-" * 60)
    outputs1 = nn1.forward([1.5, 0.5])
    print(f"\n✅ 最終輸出: {outputs1}\n")
    
    # 測試案例 2
    print("\n【測試案例 2】: (0, 1)")
    print("-" * 60)
    outputs2 = nn1.forward([0, 1])
    print(f"\n✅ 最終輸出: {outputs2}\n")
    
    print("\n" + "=" * 60)
    print("Neural Network 2 測試")
    print("=" * 60)
    
    # Network 2: 2-2-2-2 結構
    weights2 = [
    # Input layer → Hidden layer 1  (每一列對應 Input 的一個來源)
    [
        [0.5, 0.6],    # X1 → [H11, H12]
        [1.5, -0.8],   # X2 → [H11, H12]
        [0.3, 1.25]    # Bias → [H11, H12]
    ],

    # Hidden layer 1 → Hidden layer 2 
    [
        [0.6],         # H11 → [H21]
        [-0.8],        # H12 → [H21]
        [0.3]          # Bias → [H21]
    ],

    # Hidden layer 2 → Output layer (注意 H22 不存在)
    [
        [0.5, -0.4],   # H21 → [O1, O2]
      
        [0.2, 0.5]     # Bias → [O1, O2] 
    ]
]
    
    
    biases2 = [
        [0.3, 1.25],      
        [0.3],           
        [0.2, 0.5]              
    ]
    
    nn2 = Network(weights2, biases2)
    
    # 測試案例 1
    print("\n【測試案例 1】: (0.75, 1.25)")
    print("-" * 60)
    outputs3 = nn2.forward([0.75, 1.25])
    print(f"\n✅ 最終輸出: {outputs3}\n")
    
    # 測試案例 2
    print("\n【測試案例 2】: (-1, 0.5)")
    print("-" * 60)
    outputs4 = nn2.forward([-1, 0.5])
    print(f"\n✅ 最終輸出: {outputs4}\n")
    
    # 展示彈性：可以輕鬆創建第三個網路結構
    print("\n" + "=" * 60)
    print("展示彈性：假設有 Network 3 (2-3-2 結構)")
    print("=" * 60)
    
    # 假設的 Network 3: 2 輸入 → 3 隱藏 → 2 輸出
    weights3 = [
        [[1, 0.5, -0.5], [0.2, -0.3, 0.4]],  # 輸入 → 隱藏
        [[0.6, 0.3], [0.1, -0.2], [0.4, 0.5]]  # 隱藏 → 輸出
    ]
    biases3 = [
        [0.1, 0.2, 0.3],  # 隱藏層偏置
        [0, 0]            # 輸出層偏置
    ]
    
    nn3 = Network(weights3, biases3)
    print("\n【測試】: (1, 1)")
    print("-" * 60)
    outputs5 = nn3.forward([1, 1])
    print(f"\n✅ 最終輸出: {outputs5}\n")


if __name__ == "__main__":
    main()