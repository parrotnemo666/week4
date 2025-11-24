"""
Task 2: 預測鐵達尼號乘客生存狀態的二元分類神經網絡
根據乘客資料預測是否生存

作者：根據 WeHelp Week 5-6 課程內容實作
"""

import csv
import random
import math

# ============================================================
# 第一步：建立基礎神經元類別（與 Task 1 相同）
# ============================================================

class Neuron:
    """單一神經元"""
    
    def __init__(self, num_inputs, activation='relu'):
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(num_inputs)]
        self.bias = random.uniform(-0.5, 0.5)
        self.activation = activation
        
        self.inputs = None
        self.weighted_sum = None
        self.output = None
        
        self.weight_gradients = [0] * num_inputs
        self.bias_gradient = 0
    
    def forward(self, inputs):
        self.inputs = inputs
        self.weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        
        if self.activation == 'relu':
            self.output = max(0, self.weighted_sum)
        elif self.activation == 'sigmoid':
            # 防止數值溢位
            if self.weighted_sum > 500:
                self.output = 1.0
            elif self.weighted_sum < -500:
                self.output = 0.0
            else:
                self.output = 1 / (1 + math.exp(-self.weighted_sum))
        elif self.activation == 'linear':
            self.output = self.weighted_sum
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        
        return self.output
    
    def backward(self, upstream_gradient):
        if self.activation == 'relu':
            activation_derivative = 1 if self.weighted_sum > 0 else 0
        elif self.activation == 'sigmoid':
            activation_derivative = self.output * (1 - self.output)
        elif self.activation == 'linear':
            activation_derivative = 1
        
        delta = upstream_gradient * activation_derivative
        
        self.weight_gradients = [delta * x for x in self.inputs]
        self.bias_gradient = delta
        
        input_gradients = [delta * w for w in self.weights]
        
        return input_gradients


class Layer:
    """神經網絡層"""
    
    def __init__(self, num_neurons, num_inputs_per_neuron, activation='relu'):
        self.neurons = [
            Neuron(num_inputs_per_neuron, activation) 
            for _ in range(num_neurons)
        ]
    
    def forward(self, inputs):
        return [neuron.forward(inputs) for neuron in self.neurons]
    
    def backward(self, upstream_gradients):
        input_gradients_list = [
            neuron.backward(grad) 
            for neuron, grad in zip(self.neurons, upstream_gradients)
        ]
        
        num_inputs = len(input_gradients_list[0])
        input_gradients = [
            sum(grads[i] for grads in input_gradients_list)
            for i in range(num_inputs)
        ]
        
        return input_gradients
    
    def update_weights(self, learning_rate):
        for neuron in self.neurons:
            neuron.weights = [
                w - learning_rate * grad 
                for w, grad in zip(neuron.weights, neuron.weight_gradients)
            ]
            neuron.bias -= learning_rate * neuron.bias_gradient
    
    def zero_grad(self):
        for neuron in self.neurons:
            neuron.weight_gradients = [0] * len(neuron.weight_gradients)
            neuron.bias_gradient = 0


class Network:
    """完整的神經網絡"""
    
    def __init__(self, layer_configs):
        self.layers = []
        
        for i, (num_neurons, activation) in enumerate(layer_configs):
            if i == 0:
                self.first_layer_config = (num_neurons, activation)
            else:
                num_inputs = layer_configs[i-1][0]
                layer = Layer(num_neurons, num_inputs, activation)
                self.layers.append(layer)
        
        self.first_layer_built = False
    
    def forward(self, inputs):
        if not self.first_layer_built:
            num_neurons, activation = self.first_layer_config
            first_layer = Layer(num_neurons, len(inputs), activation)
            self.layers.insert(0, first_layer)
            self.first_layer_built = True
        
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        
        return outputs
    
    def backward(self, output_gradients):
        gradients = output_gradients
        for layer in reversed(self.layers):
            gradients = layer.backward(gradients)
    
    def zero_grad(self, learning_rate):
        for layer in self.layers:
            layer.update_weights(learning_rate)
            layer.zero_grad()


# ============================================================
# 第二步：建立二元分類的損失函數
# ============================================================

class BinaryCrossEntropyLoss:
    """
    二元交叉熵損失函數（Binary Cross Entropy）
    
    用於二元分類任務
    公式：Loss = -[y*log(p) + (1-y)*log(1-p)]
    其中 y 是真實標籤（0 或 1），p 是預測機率
    """
    
    def get_total_loss(self, outputs, expected):
        """
        計算損失值
        
        參數：
        - outputs: 網絡輸出（列表）
        - expected: 期望值（0 或 1）
        
        返回：
        - 損失值
        """
        # 輸出層只有一個神經元，輸出 0~1 之間的機率
        prediction = outputs[0]
        
        # 防止 log(0) 導致數值錯誤，加上小的 epsilon
        epsilon = 1e-10
        prediction = max(epsilon, min(1 - epsilon, prediction))
        
        # Binary Cross Entropy 公式
        if expected == 1:
            loss = -math.log(prediction)
        else:
            loss = -math.log(1 - prediction)
        
        return loss
    
    def get_output_gradients(self, outputs, expected):
        """
        計算損失對輸出的梯度
        
        對於 Sigmoid + BCE 的組合，梯度簡化為：
        gradient = prediction - expected
        
        返回：
        - 梯度列表
        """
        prediction = outputs[0]
        gradient = prediction - expected
        return [gradient]


# ============================================================
# 第三步：資料預處理
# ============================================================

def load_and_preprocess_data(filename):
    """
    載入並預處理 Titanic 資料
    
    特徵工程策略：
    1. Pclass: 艙等（1, 2, 3）→ 標準化
    2. Sex: 性別 → 編碼（male=0, female=1）
    3. Age: 年齡 → 填補缺失值、標準化
    4. SibSp: 兄弟姊妹/配偶數量 → 標準化
    5. Parch: 父母/子女數量 → 標準化
    6. Fare: 票價 → 標準化
    7. Embarked: 登船港口 → One-hot 編碼
    
    返回：
    - xs: 輸入列表
    - es: 期望輸出列表（0 或 1）
    - feature_stats: 特徵統計資訊（用於顯示）
    """
    print("\n" + "="*60)
    print("📁 載入並預處理 Titanic 資料")
    print("="*60)
    
    # 讀取資料
    data = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    print(f"✓ 載入 {len(data)} 筆資料")
    
    # ========================================
    # 步驟 1: 處理缺失值
    # ========================================
    print("\n步驟 1：處理缺失值")
    
    # Age 缺失值：用平均值填補
    ages = [float(d['Age']) for d in data if d['Age']]
    age_mean = sum(ages) / len(ages)
    print(f"  Age 缺失 {sum(1 for d in data if not d['Age'])} 筆")
    print(f"  用平均值填補: {age_mean:.1f} 歲")
    
    for d in data:
        if not d['Age']:
            d['Age'] = str(age_mean)
    
    # Embarked 缺失值：用最常見的值填補
    embarked_counts = {}
    for d in data:
        if d['Embarked']:
            embarked_counts[d['Embarked']] = embarked_counts.get(d['Embarked'], 0) + 1
    most_common_embarked = max(embarked_counts, key=embarked_counts.get)
    print(f"  Embarked 缺失 {sum(1 for d in data if not d['Embarked'])} 筆")
    print(f"  用最常見值填補: {most_common_embarked}")
    
    for d in data:
        if not d['Embarked']:
            d['Embarked'] = most_common_embarked
    
    # ========================================
    # 步驟 2: 特徵編碼
    # ========================================
    print("\n步驟 2：特徵編碼")
    
    # Sex: male=0, female=1
    print("  Sex: male=0, female=1")
    for d in data:
        d['Sex_Encoded'] = 0 if d['Sex'] == 'male' else 1
    
    # Embarked: One-hot 編碼（S, C, Q）
    print("  Embarked: One-hot 編碼")
    print("    S (Southampton) → [1, 0, 0]")
    print("    C (Cherbourg)   → [0, 1, 0]")
    print("    Q (Queenstown)  → [0, 0, 1]")
    
    for d in data:
        if d['Embarked'] == 'S':
            d['Embarked_S'], d['Embarked_C'], d['Embarked_Q'] = 1, 0, 0
        elif d['Embarked'] == 'C':
            d['Embarked_S'], d['Embarked_C'], d['Embarked_Q'] = 0, 1, 0
        else:  # Q
            d['Embarked_S'], d['Embarked_C'], d['Embarked_Q'] = 0, 0, 1
    
    # ========================================
    # 步驟 3: 計算統計值並標準化
    # ========================================
    print("\n步驟 3：標準化數值特徵")
    
    # 需要標準化的特徵
    numeric_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    
    stats = {}
    for feature in numeric_features:
        values = [float(d[feature]) for d in data]
        mean = sum(values) / len(values)
        std = (sum((v - mean)**2 for v in values) / len(values)) ** 0.5
        stats[feature] = {'mean': mean, 'std': std}
        
        print(f"  {feature}: 平均值={mean:.2f}, 標準差={std:.2f}")
        
        # 標準化
        for d in data:
            d[f'{feature}_Normalized'] = (float(d[feature]) - mean) / (std + 1e-10)
    
    # ========================================
    # 步驟 4: 組合特徵
    # ========================================
    print("\n步驟 4：組合特徵向量")
    
    # 特徵順序：[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_S, Embarked_C, Embarked_Q]
    features = [
        'Pclass_Normalized',
        'Sex_Encoded',
        'Age_Normalized',
        'SibSp_Normalized',
        'Parch_Normalized',
        'Fare_Normalized',
        'Embarked_S',
        'Embarked_C',
        'Embarked_Q'
    ]
    
    print(f"  總共 {len(features)} 個特徵:")
    for i, f in enumerate(features, 1):
        print(f"    {i}. {f}")
    
    # 準備訓練資料
    xs = []
    es = []
    
    for d in data:
        # 輸入特徵
        x = [float(d[f]) if f in ['Sex_Encoded', 'Embarked_S', 'Embarked_C', 'Embarked_Q'] 
             else d[f] for f in features]
        xs.append(x)
        
        # 期望輸出（生存狀態：0 或 1）
        es.append(int(d['Survived']))
    
    # 顯示前 3 筆範例
    print("\n處理後的資料範例（前 3 筆）:")
    for i in range(3):
        d = data[i]
        print(f"\n第 {i+1} 筆:")
        print(f"  原始: Pclass={d['Pclass']}, Sex={d['Sex']}, Age={d['Age']}, Survived={d['Survived']}")
        print(f"  編碼: {[f'{v:.3f}' for v in xs[i][:6]]}... → {es[i]}")
    
    return xs, es, stats


# ============================================================
# 第四步：訓練流程
# ============================================================

def train_model(xs, es, epochs=1000, learning_rate=0.01, print_every=100):
    """
    訓練二元分類神經網絡
    
    參數：
    - xs: 輸入資料
    - es: 期望輸出（0 或 1）
    - epochs: 訓練輪數
    - learning_rate: 學習率
    - print_every: 每幾個 epoch 打印一次進度
    """
    print("\n" + "="*60)
    print("🧠 建立神經網絡")
    print("="*60)
    
    # 網絡架構：輸入(9) → 隱藏層(16, ReLU) → 輸出(1, Sigmoid)
    nn = Network([
        (16, 'relu'),      # 隱藏層：16 個神經元，ReLU 激活
        (1, 'sigmoid')     # 輸出層：1 個神經元，Sigmoid 激活（二元分類）
    ])
    
    print("網絡架構：")
    print("  輸入層: 9 個特徵")
    print("    [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_S, C, Q]")
    print("  隱藏層: 16 個神經元（ReLU 激活）")
    print("  輸出層: 1 個神經元（Sigmoid 激活）")
    print(f"\n學習率: {learning_rate}")
    print(f"訓練輪數: {epochs}")
    
    loss_fn = BinaryCrossEntropyLoss()
    
    print("\n" + "="*60)
    print("🏋️ 開始訓練")
    print("="*60)
    
    # 訓練循環
    for epoch in range(epochs):
        epoch_loss_sum = 0
        
        # 遍歷所有訓練資料
        for x, e in zip(xs, es):
            # 前向傳播
            outputs = nn.forward(x)
            
            # 計算損失
            loss = loss_fn.get_total_loss(outputs, e)
            epoch_loss_sum += loss
            
            # 計算梯度
            output_gradients = loss_fn.get_output_gradients(outputs, e)
            
            # 反向傳播
            nn.backward(output_gradients)
            
            # 更新權重
            nn.zero_grad(learning_rate)
        
        # 計算平均損失
        avg_loss = epoch_loss_sum / len(xs)
        
        # 打印進度
        if (epoch + 1) % print_every == 0 or epoch == 0:
            # 計算當前正確率
            correct = 0
            for x, e in zip(xs, es):
                output = nn.forward(x)[0]
                prediction = 1 if output > 0.5 else 0
                if prediction == e:
                    correct += 1
            accuracy = correct / len(xs) * 100
            
            print(f"Epoch {epoch+1:4d}/{epochs}: 損失={avg_loss:.4f}, 正確率={accuracy:.2f}%")
    
    print("\n✓ 訓練完成！")
    
    return nn


# ============================================================
# 第五步：評估模型
# ============================================================

def evaluate_model(nn, xs, es):
    """
    評估二元分類模型
    
    參數：
    - nn: 訓練好的神經網絡
    - xs: 測試資料輸入
    - es: 測試資料期望輸出
    
    返回：
    - correct_rate: 正確率
    """
    print("\n" + "="*60)
    print("📊 評估模型")
    print("="*60)
    
    threshold = 0.5
    
    # 統計各種情況
    true_positive = 0   # 預測生存，實際生存
    true_negative = 0   # 預測死亡，實際死亡
    false_positive = 0  # 預測生存，實際死亡
    false_negative = 0  # 預測死亡，實際生存
    
    predictions = []
    actuals = []
    probabilities = []
    
    for x, e in zip(xs, es):
        # 前向傳播
        output = nn.forward(x)[0]
        
        # 根據閾值判斷
        prediction = 1 if output > threshold else 0
        
        predictions.append(prediction)
        actuals.append(e)
        probabilities.append(output)
        
        # 統計
        if prediction == 1 and e == 1:
            true_positive += 1
        elif prediction == 0 and e == 0:
            true_negative += 1
        elif prediction == 1 and e == 0:
            false_positive += 1
        else:  # prediction == 0 and e == 1
            false_negative += 1
    
    # 計算各種指標
    total = len(xs)
    correct = true_positive + true_negative
    correct_rate = correct / total * 100
    
    # 生存者的召回率（實際生存的人中，預測對了多少）
    survived_total = sum(es)
    survived_recall = true_positive / survived_total * 100 if survived_total > 0 else 0
    
    # 死亡者的召回率
    died_total = total - survived_total
    died_recall = true_negative / died_total * 100 if died_total > 0 else 0
    
    print(f"總資料筆數: {total}")
    print(f"正確預測: {correct} 筆")
    print(f"錯誤預測: {total - correct} 筆")
    print(f"\n整體正確率: {correct_rate:.2f}%")
    print()
    
    # 混淆矩陣
    print("混淆矩陣（Confusion Matrix）:")
    print(f"                預測死亡    預測生存")
    print(f"  實際死亡      {true_negative:4d}        {false_positive:4d}")
    print(f"  實際生存      {false_negative:4d}        {true_positive:4d}")
    print()
    
    # 各類別的準確度
    print(f"生存者召回率: {survived_recall:.2f}% ({true_positive}/{survived_total})")
    print(f"死亡者召回率: {died_recall:.2f}% ({true_negative}/{died_total})")
    
    # 顯示一些預測範例（各種情況都顯示）
    print("\n預測範例:")
    print(f"{'序號':<6} {'預測機率':<12} {'預測結果':<10} {'實際結果':<10} {'是否正確':<10}")
    print("-" * 55)
    
    # 找出各種類型的範例
    examples = {
        'TP': [],  # True Positive
        'TN': [],  # True Negative
        'FP': [],  # False Positive
        'FN': []   # False Negative
    }
    
    for i, (pred, actual, prob) in enumerate(zip(predictions, actuals, probabilities)):
        if pred == 1 and actual == 1:
            examples['TP'].append((i, prob, pred, actual))
        elif pred == 0 and actual == 0:
            examples['TN'].append((i, prob, pred, actual))
        elif pred == 1 and actual == 0:
            examples['FP'].append((i, prob, pred, actual))
        else:
            examples['FN'].append((i, prob, pred, actual))
    
    # 每種類型顯示 2 個範例
    count = 0
    for example_type, example_list in examples.items():
        for i, prob, pred, actual in example_list[:2]:
            correct_mark = "✓" if pred == actual else "✗"
            pred_text = "生存" if pred == 1 else "死亡"
            actual_text = "生存" if actual == 1 else "死亡"
            print(f"{i+1:<6} {prob:<12.4f} {pred_text:<10} {actual_text:<10} {correct_mark:<10}")
            count += 1
            if count >= 8:
                break
        if count >= 8:
            break
    
    # 判斷是否達標
    print("\n" + "="*60)
    if correct_rate >= 80:
        print(f"🎉 太棒了！正確率 {correct_rate:.2f}% ≥ 80%（進階目標達成）")
    elif correct_rate >= 75:
        print(f"✅ 恭喜！正確率 {correct_rate:.2f}% ≥ 75%（基本目標達成）")
    else:
        print(f"📝 正確率 {correct_rate:.2f}% < 75%（需要調整）")
    print("="*60)
    
    return correct_rate


# ============================================================
# 主程式
# ============================================================

def main():
    print("\n" + "="*60)
    print("🎯 Task 2: 預測鐵達尼號乘客生存狀態")
    print("="*60)
    
    # 載入並預處理資料
    xs, es, stats = load_and_preprocess_data(
        'C:/Users/user/Downloads/week4/titanic.csv'
        
        )
    
    # 訓練模型
    nn = train_model(xs, es, epochs=1000, learning_rate=0.01, print_every=100)
    
    # 評估模型
    accuracy = evaluate_model(nn, xs, es)
    
    print("\n✅ 程式執行完畢！")
    print(f"最終正確率: {accuracy:.2f}%")


if __name__ == "__main__":
    main()