# SCRIPT THUYẾT TRÌNH ASSIGNMENT 2 - IMPLEMENT YOUR AGENT

## I. GIỚI THIỆU TỔNG QUAN

### 1. Mục tiêu Assignment
- **Chủ đề**: Xây dựng một agent học tăng cường (Reinforcement Learning) sử dụng mạng neural để giải quyết bài toán Lunar Lander
- **Công nghệ chính**: Deep RL - kết hợp mạng neural và thuật toán học tăng cường
- **Môi trường**: Lunar Lander từ OpenAI Gym (đã cập nhật lên v3)

### 2. Các thành phần chính cần implement
1. **Action-Value Network**: Mạng neural xấp xỉ hàm action-value
2. **Adam Optimizer**: Thuật toán tối ưu hóa cho mạng neural
3. **Experience Replay Buffer**: Bộ đệm trải nghiệm để học hiệu quả
4. **Softmax Policy**: Chính sách chọn hành động dựa trên xác suất
5. **Expected Sarsa Agent**: Agent học tăng cường hoàn chỉnh

---

## II. CHI TIẾT CÁC THÀNH PHẦN

### 1. ACTION-VALUE NETWORK (Mạng Neural)

#### Mục đích:
- Xấp xỉ hàm Q(s,a) - giá trị của từng hành động trong mỗi trạng thái
- Khác với state-value function: output có nhiều unit (số lượng hành động) thay vì 1 unit

#### Kiến trúc mạng:
```python
# Cấu trúc: [state_dim, num_hidden_units, num_actions]
# Ví dụ: [8, 256, 4] cho Lunar Lander
self.layer_sizes = [self.state_dim, self.num_hidden_units, self.num_actions]
```

#### Các phương thức chính:
- **`get_action_values(s)`**: Tính toán Q-values cho batch states
- **`get_TD_update(s, delta_mat)`**: Tính gradient × TD error cho backpropagation
- **`init_saxe()`**: Khởi tạo trọng số theo phương pháp Saxe (tránh vanishing gradient)

### 2. ADAM OPTIMIZER

#### Nguyên lý hoạt động:
Adam cải tiến SGD bằng 2 khái niệm:
- **Adaptive vector stepsizes**: Kích thước bước học thích ứng
- **Momentum**: Động lượng để tránh dao động

#### Công thức toán học:
```
m_t = β_m * m_{t-1} + (1 - β_m) * g_t
v_t = β_v * v_{t-1} + (1 - β_v) * g_t²

m̂_t = m_t / (1 - β_m^t)  # Bias correction
v̂_t = v_t / (1 - β_v^t)  # Bias correction

w_t = w_{t-1} + α * m̂_t / (√v̂_t + ε)
```

#### Tham số quan trọng:
- **α (step_size)**: Tốc độ học
- **β_m (beta_m)**: Hệ số momentum (thường 0.9)
- **β_v (beta_v)**: Hệ số adaptive (thường 0.999)
- **ε (epsilon)**: Tránh chia cho 0 (thường 1e-8)

### 3. EXPERIENCE REPLAY BUFFER

#### Mục đích:
- Lưu trữ trải nghiệm (state, action, reward, terminal, next_state)
- Lấy mẫu ngẫu nhiên để học offline
- Tránh correlation giữa các trải nghiệm liên tiếp

#### Cơ chế hoạt động:
```python
def append(self, state, action, reward, terminal, next_state):
    # Thêm trải nghiệm mới, xóa cũ nếu đầy
    if len(self.buffer) == self.max_size:
        del self.buffer[0]
    self.buffer.append([state, action, reward, terminal, next_state])

def sample(self):
    # Lấy mẫu ngẫu nhiên batch_size trải nghiệm
    idxs = self.rand_generator.choice(np.arange(len(self.buffer)), size=self.minibatch_size)
    return [self.buffer[idx] for idx in idxs]
```

### 4. SOFTMAX POLICY

#### Công thức xác suất:
```
P(A_t = a | S_t = s) = e^(Q(s,a)/τ) / Σ_b e^(Q(s,b)/τ)
```

#### Tính ổn định số học:
```
P(A_t = a | S_t = s) = e^(Q(s,a)/τ - max_c Q(s,c)/τ) / Σ_b e^(Q(s,b)/τ - max_c Q(s,c)/τ)
```

#### Tham số τ (tau):
- **τ nhỏ**: Tập trung vào hành động có giá trị cao nhất (greedy)
- **τ lớn**: Chọn hành động đồng đều hơn (exploration)

### 5. EXPECTED SARSA ALGORITHM

#### Pseudocode:
```
Q_t ← action-value network tại timestep t
Khởi tạo Q_{t+1}^1 ← Q_t
For i in [1, ..., N] (N replay steps):
    s, a, r, t, s' ← Sample batch từ replay buffer
    Q_{t+1}^{i+1}(s,a) ← Q_{t+1}^i(s,a) + α * [r + γ * Σ_b π(b|s') * Q_t(s',b) - Q_{t+1}^i(s,a)]
Set Q_{t+1} ← Q_{t+1}^N
```

#### Các bước tính TD Error:
1. **Tính Q-values cho next states**: `q_next_mat = current_q.get_action_values(next_states)`
2. **Tính policy cho next states**: `probs_mat = softmax(q_next_mat, tau)`
3. **Tính value của next state**: `v_next_vec = Σ_b π(b|s') * Q(s',b)`
4. **Tính Expected Sarsa target**: `target = r + γ * v_next * (1 - terminal)`
5. **Tính Q-values cho current states**: `q_mat = network.get_action_values(states)`
6. **Tính TD error**: `delta = target - Q(s,a)`

---

## III. IMPLEMENTATION CHI TIẾT

### 1. ActionValueNetwork Class
```python
class ActionValueNetwork:
    def __init__(self, network_config):
        self.state_dim = network_config.get("state_dim")
        self.num_hidden_units = network_config.get("num_hidden_units") 
        self.num_actions = network_config.get("num_actions")
        self.layer_sizes = [self.state_dim, self.num_hidden_units, self.num_actions]
        # Khởi tạo trọng số với Saxe initialization
```

### 2. Adam Class
```python
class Adam:
    def __init__(self, layer_sizes, optimizer_info):
        # Khởi tạo m và v với zeros
        self.m = [{"W": zeros, "b": zeros} for i in range(len(layer_sizes)-1)]
        self.v = [{"W": zeros, "b": zeros} for i in range(len(layer_sizes)-1)]
        
    def update_weights(self, weights, td_errors_times_gradients):
        # Cập nhật m, v, tính m_hat, v_hat, và cập nhật weights
```

### 3. Softmax Function
```python
def softmax(action_values, tau=1.0):
    preferences = action_values / tau
    max_preference = np.max(preferences, axis=1)
    exp_preferences = np.exp(preferences - max_preference.reshape(-1,1))
    sum_exp = np.sum(exp_preferences, axis=1)
    action_probs = exp_preferences / sum_exp.reshape(-1,1)
    return action_probs.squeeze()
```

### 4. Agent Class
```python
class Agent(BaseAgent):
    def agent_step(self, reward, state):
        # 1. Chọn action
        action = self.policy(state)
        
        # 2. Thêm vào replay buffer
        self.replay_buffer.append(self.last_state, self.last_action, reward, 0, state)
        
        # 3. Replay learning
        if self.replay_buffer.size() > self.minibatch_size:
            current_q = deepcopy(self.network)
            for _ in range(self.num_replay):
                experiences = self.replay_buffer.sample()
                optimize_network(experiences, self.discount, self.optimizer, 
                               self.network, current_q, self.tau)
        
        # 4. Cập nhật state và action
        self.last_state = state
        self.last_action = action
        return action
```

---

## IV. KẾT QUẢ VÀ ĐÁNH GIÁ

### 1. Tham số thí nghiệm
```python
experiment_parameters = {
    "num_runs": 1,
    "num_episodes": 300,
    "timeout": 500
}

agent_parameters = {
    'network_config': {
        'state_dim': 8,
        'num_hidden_units': 256,
        'num_actions': 4
    },
    'optimizer_config': {
        'step_size': 1e-3,
        'beta_m': 0.9, 
        'beta_v': 0.999,
        'epsilon': 1e-8
    },
    'replay_buffer_size': 50000,
    'minibatch_sz': 8,
    'num_replay_updates_per_step': 4,
    'gamma': 0.99,
    'tau': 0.001
}
```

### 2. Quá trình học
- **Episode 0**: Agent crash ngay lập tức
- **Episode 50-100**: Học cách tránh crash bằng cách sử dụng nhiên liệu
- **Episode 200+**: Học cách hạ cánh mượt mà trong vùng landing zone

### 3. So sánh hiệu suất
- **Random Agent**: Tổng reward trung bình ~-150 đến -200
- **Expected Sarsa Agent**: Tổng reward trung bình >200 sau 3000 episodes

---

## V. Ý NGHĨA VÀ ỨNG DỤNG

### 1. Tầm quan trọng của Deep RL
- Kết hợp sức mạnh của mạng neural với thuật toán RL
- Có thể xử lý không gian trạng thái liên tục và lớn
- Đã tạo ra những thành tựu ấn tượng như AlphaGo, AlphaZero

### 2. Các kỹ thuật quan trọng
- **Experience Replay**: Tăng hiệu quả học bằng cách tái sử dụng dữ liệu
- **Target Network**: Ổn định quá trình học bằng cách sử dụng network cũ
- **Adam Optimizer**: Cải thiện tốc độ và chất lượng hội tụ
- **Softmax Policy**: Cân bằng giữa exploration và exploitation

### 3. Ứng dụng thực tế
- **Game AI**: Cờ vua, cờ vây, video games
- **Robotics**: Điều khiển robot, autonomous vehicles
- **Finance**: Algorithmic trading, portfolio management
- **Healthcare**: Drug discovery, treatment optimization

---

## VI. KẾT LUẬN

### 1. Thành tựu đạt được
- ✅ Implement thành công Expected Sarsa agent với mạng neural
- ✅ Sử dụng Adam optimizer cho việc tối ưu hóa
- ✅ Áp dụng Experience Replay để học hiệu quả
- ✅ Giải quyết thành công bài toán Lunar Lander

### 2. Kiến thức thu được
- Hiểu sâu về Deep Reinforcement Learning
- Nắm vững các kỹ thuật tối ưu hóa mạng neural
- Biết cách implement các thuật toán RL phức tạp
- Có kinh nghiệm thực hành với môi trường thực tế

### 3. Hướng phát triển
- Thử nghiệm với các thuật toán RL khác (DQN, A3C, PPO)
- Tối ưu hóa hyperparameters
- Áp dụng vào các bài toán phức tạp hơn
- Nghiên cứu các kỹ thuật mới trong Deep RL

---

## VII. DEMO VÀ THẢO LUẬN

### 1. Chạy thử nghiệm
- Chạy code để xem agent học trong thời gian thực
- Quan sát learning curve và sự cải thiện performance
- So sánh với random agent

### 2. Thảo luận
- Tại sao cần Experience Replay?
- Lợi ích của Adam so với SGD?
- Tác động của tham số tau trong Softmax policy?
- Cách cải thiện performance của agent?

### 3. Mở rộng
- Thử nghiệm với các môi trường khác
- Implement các thuật toán RL khác
- Tối ưu hóa architecture của mạng neural
- Áp dụng vào các bài toán thực tế

---

**Cảm ơn các bạn đã lắng nghe!**