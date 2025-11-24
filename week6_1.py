"""
Task 1: 預測體重的回歸神經網絡
使用性別和身高預測體重

作者：根據 WeHelp Week 5-6 課程內容實作

增強版功能：
- 記錄每個 epoch 的損失和誤差
- 自動保存訓練歷史到 CSV 檔案
- 分析訓練過程和收斂情況
"""

import csv
import random
import math
import time

# ============================================================
# 第一步：建立基礎神經元類別
# ============================================================

class Neuron:
    """
    單一神經元
    
    功能：
    1. 儲存權重和偏差
    2. 計算加權和（weighted sum）
    3. 應用激活函數
    4. 儲存梯度用於反向傳播
    """
    
    def __init__(self, num_inputs, activation='relu'):
        """
        初始化神經元
        
        參數：
        - num_inputs: 輸入的數量
        - activation: 激活函數類型 ('relu', 'sigmoid', 'linear')
        """
        # 使用小的隨機數初始化權重（避免對稱性問題）
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(num_inputs)]
        self.bias = random.uniform(-0.5, 0.5)
        self.activation = activation
        
        # 用於儲存前向傳播的中間值（反向傳播會用到）
        self.inputs = None
        self.weighted_sum = None
        self.output = None
        
        # 用於儲存梯度
        self.weight_gradients = [0] * num_inputs
        self.bias_gradient = 0
    
    def forward(self, inputs):
        """
        前向傳播
        
        步驟：
        1. 計算加權和：z = w1*x1 + w2*x2 + ... + b
        2. 應用激活函數：output = activation(z)
        """
        self.inputs = inputs
        
        # 計算加權和
        self.weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        
        # 應用激活函數
        if self.activation == 'relu':
            self.output = max(0, self.weighted_sum)
        elif self.activation == 'sigmoid':
            self.output = 1 / (1 + math.exp(-self.weighted_sum))
        elif self.activation == 'linear':
            self.output = self.weighted_sum
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        
        return self.output
    
    def backward(self, upstream_gradient):
        """
        反向傳播
        
        參數：
        - upstream_gradient: 從後面層傳回來的梯度
        
        返回：
        - 傳給前一層的梯度
        """
        # 計算激活函數的導數
        if self.activation == 'relu':
            activation_derivative = 1 if self.weighted_sum > 0 else 0
        elif self.activation == 'sigmoid':
            activation_derivative = self.output * (1 - self.output)
        elif self.activation == 'linear':
            activation_derivative = 1
        
        # 計算對加權和的梯度
        delta = upstream_gradient * activation_derivative
        
        # 計算對權重和偏差的梯度
        self.weight_gradients = [delta * x for x in self.inputs]
        self.bias_gradient = delta
        
        # 計算傳給前一層的梯度
        input_gradients = [delta * w for w in self.weights]
        
        return input_gradients


class Layer:
    """
    神經網絡層（包含多個神經元）
    """
    
    def __init__(self, num_neurons, num_inputs_per_neuron, activation='relu'):
        """
        初始化層
        
        參數：
        - num_neurons: 這一層有幾個神經元
        - num_inputs_per_neuron: 每個神經元接收幾個輸入
        - activation: 激活函數類型
        """
        self.neurons = [
            Neuron(num_inputs_per_neuron, activation) 
            for _ in range(num_neurons)
        ]
    
    def forward(self, inputs):
        """前向傳播：每個神經元都計算輸出"""
        return [neuron.forward(inputs) for neuron in self.neurons]
    
    def backward(self, upstream_gradients):
        """
        反向傳播
        
        參數：
        - upstream_gradients: 每個神經元對應的梯度列表
        
        返回：
        - 對輸入的梯度
        """
        # 每個神經元計算自己的梯度
        input_gradients_list = [
            neuron.backward(grad) 
            for neuron, grad in zip(self.neurons, upstream_gradients)
        ]
        
        # 將所有神經元對相同輸入的梯度加總
        num_inputs = len(input_gradients_list[0])
        input_gradients = [
            sum(grads[i] for grads in input_gradients_list)
            for i in range(num_inputs)
        ]
        
        return input_gradients
    
    def update_weights(self, learning_rate):
        """更新所有神經元的權重和偏差"""
        for neuron in self.neurons:
            # w_new = w_old - learning_rate * gradient
            neuron.weights = [
                w - learning_rate * grad 
                for w, grad in zip(neuron.weights, neuron.weight_gradients)
            ]
            neuron.bias -= learning_rate * neuron.bias_gradient
    
    def zero_grad(self):
        """清空梯度（為下一次訓練準備）"""
        for neuron in self.neurons:
            neuron.weight_gradients = [0] * len(neuron.weight_gradients)
            neuron.bias_gradient = 0


class Network:
    """
    完整的神經網絡（多層堆疊）
    """
    
    def __init__(self, layer_configs):
        """
        初始化網絡
        
        參數：
        - layer_configs: 列表，每個元素是 (num_neurons, activation)
          例如：[(4, 'relu'), (1, 'linear')] 表示一個隱藏層 4 個神經元用 ReLU，
                輸出層 1 個神經元用 Linear
        """
        self.layers = []
        
        # 建立每一層
        for i, (num_neurons, activation) in enumerate(layer_configs):
            if i == 0:
                # 第一層需要知道輸入維度，這裡先不建立，等 forward 時再處理
                self.first_layer_config = (num_neurons, activation)
            else:
                # 後續層的輸入維度 = 前一層的神經元數量
                num_inputs = layer_configs[i-1][0]
                layer = Layer(num_neurons, num_inputs, activation)
                self.layers.append(layer)
        
        self.first_layer_built = False
    
    def forward(self, inputs):
        """
        前向傳播
        
        參數：
        - inputs: 輸入列表
        
        返回：
        - 輸出列表
        """
        # 如果是第一次 forward，建立第一層
        if not self.first_layer_built:
            num_neurons, activation = self.first_layer_config
            first_layer = Layer(num_neurons, len(inputs), activation)
            self.layers.insert(0, first_layer)
            self.first_layer_built = True
        
        # 依序通過每一層
        outputs = inputs
        for layer in self.layers:
            outputs = layer.forward(outputs)
        
        return outputs
    
    def backward(self, output_gradients):
        """
        反向傳播
        
        參數：
        - output_gradients: 對輸出的梯度
        """
        gradients = output_gradients
        for layer in reversed(self.layers):
            gradients = layer.backward(gradients)
    
    def zero_grad(self, learning_rate):
        """
        更新權重並清空梯度
        
        注意：這裡的方法名稱叫 zero_grad，但實際上包含了權重更新
        （根據作業範例的程式碼結構）
        """
        for layer in self.layers:
            layer.update_weights(learning_rate)
            layer.zero_grad()


# ============================================================
# 第二步：建立損失函數類別
# ============================================================

class MSELoss:
    """
    均方誤差損失函數（Mean Squared Error）
    
    用於回歸任務
    公式：Loss = (output - expected)^2
    """
    
    def get_total_loss(self, outputs, expected):
        """
        計算損失值
        
        參數：
        - outputs: 網絡輸出（列表）
        - expected: 期望值（單一數值）
        
        返回：
        - 損失值
        """
        # 因為輸出層只有一個神經元，所以 outputs[0] 就是預測的體重
        prediction = outputs[0]
        return (prediction - expected) ** 2
    
    def get_output_gradients(self, outputs, expected):
        """
        計算損失對輸出的梯度
        
        公式：d(Loss)/d(output) = 2 * (output - expected)
        
        返回：
        - 梯度列表
        """
        prediction = outputs[0]
        gradient = 2 * (prediction - expected)
        return [gradient]


# ============================================================
# 第三步：資料預處理
# ============================================================

def load_and_preprocess_data(filename):
    """
    載入並預處理資料
    
    步驟：
    1. 讀取 CSV
    2. 性別編碼：Male=0, Female=1
    3. 標準化身高和體重
    
    返回：
    - xs: 輸入列表（每個元素是 [性別編碼, 標準化身高]）
    - es: 期望輸出列表（標準化體重）
    - height_mean, height_std: 身高的平均值和標準差（用於還原）
    - weight_mean, weight_std: 體重的平均值和標準差（用於還原）
    """
    print("\n" + "="*60)
    print("📁 載入並預處理資料")
    print("="*60)
    
    # 讀取資料
    data = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'Gender': row['Gender'],
                'Height': float(row['Height']),
                'Weight': float(row['Weight'])
            })
    
    print(f"✓ 載入 {len(data)} 筆資料")
    
    # 步驟 1：性別編碼
    print("\n步驟 1：性別編碼")
    print("  Male → 0")
    print("  Female → 1")
    
    for d in data:
        d['Gender_Encoded'] = 0 if d['Gender'] == 'Male' else 1
    
    # 步驟 2：計算統計值（用於標準化）
    heights = [d['Height'] for d in data]
    weights = [d['Weight'] for d in data]
    
    height_mean = sum(heights) / len(heights)
    height_std = (sum((h - height_mean)**2 for h in heights) / len(heights)) ** 0.5
    
    weight_mean = sum(weights) / len(weights)
    weight_std = (sum((w - weight_mean)**2 for w in weights) / len(weights)) ** 0.5
    
    print("\n步驟 2：計算統計值")
    print(f"  身高平均值: {height_mean:.2f} 英吋")
    print(f"  身高標準差: {height_std:.2f} 英吋")
    print(f"  體重平均值: {weight_mean:.2f} 磅")
    print(f"  體重標準差: {weight_std:.2f} 磅")
    
    # 步驟 3：標準化
    print("\n步驟 3：標準化")
    print("  公式：normalized = (value - mean) / std")
    
    for d in data:
        d['Height_Normalized'] = (d['Height'] - height_mean) / height_std
        d['Weight_Normalized'] = (d['Weight'] - weight_mean) / weight_std
    
    # 打印前 3 筆範例
    print("\n標準化前 vs 標準化後（前 3 筆）:")
    for i in range(3):
        d = data[i]
        print(f"  原始: 性別={d['Gender']}, 身高={d['Height']:.2f}, 體重={d['Weight']:.2f}")
        print(f"  標準化: 性別={d['Gender_Encoded']}, 身高={d['Height_Normalized']:.4f}, 體重={d['Weight_Normalized']:.4f}")
        print()
    
    # 準備訓練資料
    xs = [[d['Gender_Encoded'], d['Height_Normalized']] for d in data]
    es = [d['Weight_Normalized'] for d in data]
    
    return xs, es, height_mean, height_std, weight_mean, weight_std


# ============================================================
# 第四步：訓練流程
# ============================================================

def train_model(xs, es, epochs=500, learning_rate=0.01, print_every=50, 
                weight_mean=None, weight_std=None, save_history=True):
    """
    訓練神經網絡（增強版 - 記錄所有訓練過程）
    
    參數：
    - xs: 輸入資料
    - es: 期望輸出
    - epochs: 訓練輪數
    - learning_rate: 學習率
    - print_every: 每幾個 epoch 打印一次進度
    - weight_mean, weight_std: 用於計算真實誤差（可選）
    - save_history: 是否保存訓練歷史
    """
    import time
    
    print("\n" + "="*60)
    print("🧠 建立神經網絡")
    print("="*60)
    
    # 網絡架構：輸入(2) → 隱藏層(8, ReLU) → 輸出(1, Linear)
    nn = Network([
        (8, 'relu'),      # 隱藏層：8 個神經元，ReLU 激活
        (1, 'linear')     # 輸出層：1 個神經元，Linear 激活（回歸任務）
    ])
    
    print("網絡架構：")
    print("  輸入層: 2 個特徵 [性別編碼, 標準化身高]")
    print("  隱藏層: 8 個神經元（ReLU 激活）")
    print("  輸出層: 1 個神經元（Linear 激活）")
    print(f"\n學習率: {learning_rate}")
    print(f"訓練輪數: {epochs}")
    
    loss_fn = MSELoss()
    
    # 訓練歷史記錄
    history = {
        'epoch': [],
        'loss': [],
        'avg_error_pounds': [] if weight_mean and weight_std else None
    }
    
    print("\n" + "="*60)
    print("🏋️ 開始訓練")
    print("="*60)
    
    start_time = time.time()
    
    # 訓練循環
    for epoch in range(epochs):
        epoch_loss_sum = 0
        epoch_error_sum = 0 if weight_mean and weight_std else None
        
        # 遍歷所有訓練資料
        for x, e in zip(xs, es):
            # 前向傳播
            outputs = nn.forward(x)
            
            # 計算損失
            loss = loss_fn.get_total_loss(outputs, e)
            epoch_loss_sum += loss
            
            # 如果提供了統計值，計算真實誤差
            if weight_mean and weight_std:
                predicted_weight = outputs[0] * weight_std + weight_mean
                actual_weight = e * weight_std + weight_mean
                error = abs(predicted_weight - actual_weight)
                epoch_error_sum += error
            
            # 計算梯度
            output_gradients = loss_fn.get_output_gradients(outputs, e)
            
            # 反向傳播
            nn.backward(output_gradients)
            
            # 更新權重
            nn.zero_grad(learning_rate)
        
        # 計算平均損失
        avg_loss = epoch_loss_sum / len(xs)
        avg_error = epoch_error_sum / len(xs) if epoch_error_sum is not None else None
        
        # 記錄歷史
        history['epoch'].append(epoch + 1)
        history['loss'].append(avg_loss)
        if avg_error is not None:
            history['avg_error_pounds'].append(avg_error)
        
        # 打印進度
        if (epoch + 1) % print_every == 0 or epoch == 0:
            if avg_error is not None:
                print(f"Epoch {epoch+1:4d}/{epochs}: "
                      f"損失={avg_loss:.6f}, 誤差={avg_error:.2f}磅")
            else:
                print(f"Epoch {epoch+1:4d}/{epochs}: 平均損失 = {avg_loss:.6f}")
    
    elapsed_time = time.time() - start_time
    
    print("\n✓ 訓練完成！")
    print(f"總訓練時間: {elapsed_time:.2f} 秒")
    print(f"平均每個 epoch: {elapsed_time/epochs*1000:.2f} 毫秒")
    
    # 保存訓練歷史
    if save_history:
        save_training_history(history, weight_mean, weight_std)
    
    return nn, history


def save_training_history(history, weight_mean, weight_std):
    """
    將訓練歷史保存到 CSV 檔案
    
    參數：
    - history: 訓練歷史字典
    - weight_mean, weight_std: 體重統計值
    """
    print("\n" + "="*60)
    print("💾 保存訓練歷史")
    print("="*60)
    
    filename = 'training_history.csv'
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 寫入表頭
        if history['avg_error_pounds'] is not None:
            writer.writerow(['Epoch', 'Loss', 'Avg_Error_Pounds'])
            # 寫入資料
            for i in range(len(history['epoch'])):
                writer.writerow([
                    history['epoch'][i],
                    history['loss'][i],
                    history['avg_error_pounds'][i]
                ])
        else:
            writer.writerow(['Epoch', 'Loss'])
            # 寫入資料
            for i in range(len(history['epoch'])):
                writer.writerow([
                    history['epoch'][i],
                    history['loss'][i]
                ])
    
    print(f"✓ 訓練歷史已保存到: {filename}")
    print(f"✓ 共 {len(history['epoch'])} 筆記錄")
    
    # 顯示統計資訊
    analyze_training_history(history)


def analyze_training_history(history):
    """
    分析訓練歷史
    
    參數：
    - history: 訓練歷史字典
    """
    print("\n📈 訓練過程分析:")
    
    # 找出最佳結果
    best_loss_idx = history['loss'].index(min(history['loss']))
    
    print(f"  最低損失: Epoch {history['epoch'][best_loss_idx]} = {history['loss'][best_loss_idx]:.6f}")
    
    if history['avg_error_pounds'] is not None:
        best_error_idx = history['avg_error_pounds'].index(min(history['avg_error_pounds']))
        print(f"  最低誤差: Epoch {history['epoch'][best_error_idx]} = {history['avg_error_pounds'][best_error_idx]:.2f} 磅")
    
    # 計算改進幅度
    initial_loss = history['loss'][0]
    final_loss = history['loss'][-1]
    loss_improvement = (initial_loss - final_loss) / initial_loss * 100
    
    print(f"  損失改善: {initial_loss:.6f} → {final_loss:.6f} ({loss_improvement:.2f}%)")
    
    if history['avg_error_pounds'] is not None:
        initial_error = history['avg_error_pounds'][0]
        final_error = history['avg_error_pounds'][-1]
        error_improvement = (initial_error - final_error) / initial_error * 100
        print(f"  誤差改善: {initial_error:.2f} → {final_error:.2f} 磅 ({error_improvement:.2f}%)")
    
    # 檢查收斂情況
    if len(history['loss']) >= 100:
        last_100_losses = history['loss'][-100:]
        loss_mean = sum(last_100_losses) / 100
        loss_std = (sum((l - loss_mean)**2 for l in last_100_losses) / 100) ** 0.5
        
        print(f"\n  收斂情況（最後 100 個 epoch）:")
        print(f"    損失標準差: {loss_std:.8f}")
        
        if loss_std < 0.0001:
            print(f"    ✅ 模型已充分收斂")
        elif loss_std < 0.001:
            print(f"    ⚠️ 模型接近收斂")
        else:
            print(f"    📈 模型仍在學習")


# ============================================================
# 第五步：評估模型
# ============================================================

def evaluate_model(nn, xs, es, weight_mean, weight_std):
    """
    評估模型性能
    
    參數：
    - nn: 訓練好的神經網絡
    - xs: 測試資料輸入
    - es: 測試資料期望輸出
    - weight_mean, weight_std: 用於還原體重的統計值
    
    返回：
    - avg_error_pounds: 平均誤差（磅）
    """
    print("\n" + "="*60)
    print("📊 評估模型")
    print("="*60)
    
    loss_fn = MSELoss()
    loss_sum = 0
    error_sum_pounds = 0
    
    predictions = []
    actuals = []
    
    for x, e in zip(xs, es):
        # 前向傳播
        outputs = nn.forward(x)
        
        # 計算標準化的損失
        loss = loss_fn.get_total_loss(outputs, e)
        loss_sum += loss
        
        # 還原成真實的體重（磅）
        predicted_weight = outputs[0] * weight_std + weight_mean
        actual_weight = e * weight_std + weight_mean
        
        predictions.append(predicted_weight)
        actuals.append(actual_weight)
        
        # 計算誤差（磅）
        error = abs(predicted_weight - actual_weight)
        error_sum_pounds += error
    
    # 計算平均值
    avg_loss = loss_sum / len(xs)
    avg_error_pounds = error_sum_pounds / len(xs)
    
    print(f"平均 MSE 損失（標準化）: {avg_loss:.6f}")
    print(f"平均絕對誤差: {avg_error_pounds:.2f} 磅")
    print(f"相對誤差: {avg_error_pounds / weight_mean * 100:.2f}%")
    
    # 顯示一些預測範例
    print("\n預測範例（前 10 筆）:")
    print(f"{'序號':<6} {'預測體重':<12} {'實際體重':<12} {'誤差':<10}")
    print("-" * 45)
    for i in range(min(10, len(predictions))):
        error = abs(predictions[i] - actuals[i])
        print(f"{i+1:<6} {predictions[i]:<12.2f} {actuals[i]:<12.2f} {error:<10.2f}")
    
    # 判斷是否達標
    print("\n" + "="*60)
    if avg_error_pounds < 15:
        print(f"🎉 恭喜！平均誤差 {avg_error_pounds:.2f} 磅 < 15 磅（目標達成）")
    else:
        print(f"📝 平均誤差 {avg_error_pounds:.2f} 磅 > 15 磅（需要調整）")
    print("="*60)
    
    return avg_error_pounds


# ============================================================
# 主程式
# ============================================================

def main():
    print("\n" + "="*60)
    print("🎯 Task 1: 根據性別和身高預測體重（增強版）")
    print("="*60)
    
    # 載入並預處理資料
    xs, es, height_mean, height_std, weight_mean, weight_std = load_and_preprocess_data(
       'C:/Users/user/Downloads/week4/gender-height-weight.csv'
        
    )
    
    # 訓練模型（增強版 - 記錄所有訓練過程）
    # 可以調整參數：
    # - epochs: 訓練輪數（500 或 1000）
    # - print_every: 每幾個 epoch 顯示一次（1=全部顯示, 50=每50次顯示）
    # - save_history: 是否保存訓練歷史到 CSV
    nn, history = train_model(
        xs, es, 
        epochs=1000,           # 訓練 1000 輪
        learning_rate=0.01, 
        print_every=100,       # 每 100 個 epoch 顯示一次（避免刷屏）
        weight_mean=weight_mean,
        weight_std=weight_std,
        save_history=True      # 保存詳細記錄到 CSV
    )
    
    # 評估模型
    avg_error = evaluate_model(nn, xs, es, weight_mean, weight_std)
    
    print("\n✅ 程式執行完畢！")
    print(f"\n💡 提示:")
    print(f"  - 訓練歷史已保存到 training_history.csv")
    print(f"  - 可以用 Excel 或記事本打開查看每個 epoch 的詳細數據")
    print(f"  - 總共記錄了 {len(history['epoch'])} 個 epoch 的訓練過程")


if __name__ == "__main__":
    main()