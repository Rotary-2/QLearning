# 添加状态离散函数改进  奖励函数简化  经验回放增强

import numpy as np
import random
from collections import deque
import struct
import serial
import time
import csv
import threading

csv_file = open("log.csv", mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["time", "reward", "position", "velocity", "acceleration", "action", "Q_value"])
start_time = time.time()

metrics_file = open("metrics.csv", mode='w', newline='')
metrics_writer = csv.writer(metrics_file)
metrics_writer.writerow(["step", "avg_reward", "avg_action"])
metrics_file.flush()  # 立即写入表头

reward_window = deque(maxlen=100)
center_count = 0
action_magnitude_sum = 0
total_steps = 0

# --------------------------- 串口初始化 ----------------------------
while True:
    try:
        ser = serial.Serial('COM3', 115200, timeout=2)
        print("串口已打开")
        break
    except serial.SerialException:
        print("未找到串口 COM3，等待1秒后重试...")
        time.sleep(1)

# --------------------------- Q-learning 参数 ----------------------------
ALPHA = 0.1         # 学习率
GAMMA = 0.85        # 折扣因子
EPSILON = 0.4       # 探索率
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

# ACTIONS = np.arange(-1, 1 + 0.05, 0.05)
ACTIONS = np.arange(1, -1 - 0.05, -0.05)
Q_table = {}

last_velocity = 0
center_stable_steps = 0
STABLE_THRESHOLD = 0.02
STABLE_VELOCITY = 0.03
REQUIRED_STABLE_STEPS = 20

# 需要在全局初始化
stable_time_counter = 0
last_position = 0

MAX_VELOCITY = 500

# 在参数部分添加经验回放相关参数
REPLAY_BUFFER_SIZE = 1000  # 经验回放缓冲区大小
BATCH_SIZE = 32            # 每次回放的批次大小
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)  # 经验回放缓冲区

# --------------------------- 改进的状态离散函数 ----------------------------
def discretize_state(position, velocity, acceleration):
    # 对位置使用更精细的分段（中心区域更精细）
    if abs(position) < 0.2:
        pos_d = int(np.interp(position, [-0.2, 0.2], [0, 9]))
    else:
        pos_d = int(np.interp(position, [-1.0, -0.2, 0.2, 1.0], [0, 4, 5, 9]))
    
    # 对速度使用对数尺度（更关注低速区域）
    vel_sign = 1 if velocity >= 0 else 0
    vel_abs = min(abs(velocity), 1.0)
    vel_d = vel_sign * 5 + int(np.interp(np.log1p(vel_abs*10), [0, np.log1p(10)], [0, 4]))
    
    # 对加速度使用线性分段
    acc_d = int(np.interp(acceleration, [-5.0, 5.0], [0, 9]))
    
    return (pos_d, vel_d, acc_d)


def compute_reward(position, velocity):
    """
    优化版奖励函数 v1.2
    核心改进：
    1. 软化边界惩罚 (-32 → -21)
    2. 调整距离衰减系数 (4→3)
    3. 动态速度惩罚 (远离中心时惩罚更强)
    4. 提高精准停止奖励 (+2→+5)
    """
    # 输入安全检查
    position = np.clip(position, -1.0, 1.0)
    velocity = np.clip(velocity, -2.0, 2.0)  # 假设归一化后速度最大为2
    
    distance = abs(position)
    speed = abs(velocity)
    
    # ================= 边界惩罚 =================
    # 梯度惩罚：距离0.8时-15，距离1.0时-21
    if distance > 0.8:
        boundary_penalty = -15 * (1 + 2*(distance-0.8))
        return max(boundary_penalty, -25)  # 限制最大惩罚
    
    # ================= 基础奖励 =================
    # 距离奖励：中心=10，衰减系数3（比原版更平缓）
    position_reward = 10 * np.exp(-distance * 3)
    
    # ================= 速度惩罚 =================
    # 动态惩罚：远离中心时惩罚更强
    if distance > 0.3:
        speed_penalty = 1.2 * speed  # 外围强惩罚
    else:
        speed_penalty = 0.5 * speed  # 中心区弱惩罚
    
    # ================= 精准控制奖励 =================
    if distance < 0.1:
        # 位置精细奖励：线性增长
        position_reward += 5 * (1 - distance/0.1)
        
        # 超精准停止奖励
        if speed < 0.05:
            position_reward += 5
    
    # ================= 最终合成 =================
    total_reward = position_reward - speed_penalty
    
    # 输出保护（理论上应在[-21, 20]之间）
    return np.clip(total_reward, -25, 20)

# --------------------------- 选择动作（epsilon贪婪） ----------------------------
def choose_action(state):
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    if state not in Q_table:
        Q_table[state] = np.zeros(len(ACTIONS))
    best_index = int(np.argmax(Q_table[state]))
    return ACTIONS[best_index]

# --------------------------- 增强的经验回放 ----------------------------
def replay_experience():
    if len(replay_buffer) < BATCH_SIZE:
        return
    
    batch = random.sample(replay_buffer, BATCH_SIZE)
    for state, action, reward, next_state in batch:
        if state not in Q_table:
            Q_table[state] = np.zeros(len(ACTIONS))
        if next_state not in Q_table:
            Q_table[next_state] = np.zeros(len(ACTIONS))
            
        # action_index = ACTIONS.index(action)
        action_index = int(np.where(ACTIONS == action)[0][0])

        best_next_action_index = int(np.argmax(Q_table[next_state]))
        td_target = reward + GAMMA * Q_table[next_state][best_next_action_index]
        td_error = td_target - Q_table[state][action_index]
        Q_table[state][action_index] += ALPHA * td_error

# --------------------------- 串口接收 STM32 状态 ----------------------------
def receive_from_stm32():
    try:
        line = ser.readline().decode().strip()
        if not line:
            return None
        parts = line.split(',')
        if len(parts) != 3:
            return None
        position = float(parts[0])
        velocity = float(parts[1])
        acceleration = float(parts[2])
        
        # 限幅
        position = max(min(position, 250), -250)
        velocity = max(min(velocity, 500), -500)
        acceleration = max(min(acceleration, 1000), -1000)

        # 归一化
        norm_position = position / 250
        norm_velocity = velocity / 500
        norm_acceleration = acceleration / 1000

        return norm_position, norm_velocity, norm_acceleration
    except:
        return None

# --------------------------- 发送动作到 STM32 ----------------------------
def send_to_stm32(action):
    try:
        message = f"{action}\r\n"
        ser.write(message.encode())
        # print(f"[发送串口] {message.strip()}")
    except Exception as e:
        print(f"[错误] 发送失败: {e}")

# --------------------------- Q-learning 学习步骤 ----------------------------
# 修改后的train_step函数，添加经验存储
def train_step(position, velocity, acceleration):
    global EPSILON, center_stable_steps

    state = discretize_state(position, velocity, acceleration)
    action = choose_action(state)
    send_to_stm32(action)
    # print(f"[发送] 状态: {state} -> 动作: {action:.3f}")

    next_data = receive_from_stm32()
    if not next_data:
        print("[警告] 未收到下一个状态数据")
        return

    next_position, next_velocity, next_acceleration = next_data
    next_velocity = np.clip(next_velocity / MAX_VELOCITY, -1.0, 1.0)
    next_state = discretize_state(next_position, next_velocity, next_acceleration)

    reward = compute_reward(position, velocity)

    # 存储经验到回放缓冲区
    replay_buffer.append((state, action, reward, next_state))

    global total_steps, center_count, action_magnitude_sum

    total_steps += 1
    reward_window.append(reward)
    action_magnitude_sum += abs(action)
    if abs(position) < STABLE_THRESHOLD:
        center_count += 1
 
    if total_steps % 10 == 0:
        avg_reward = np.mean(reward_window)
        center_rate = center_count / total_steps
        avg_action = action_magnitude_sum / total_steps
        metrics_writer.writerow([total_steps, avg_reward, center_rate, avg_action])
        metrics_file.flush()

    if abs(position) < STABLE_THRESHOLD and abs(velocity) < STABLE_VELOCITY:
        center_stable_steps += 1
    else:
        center_stable_steps = 0

    done = False
    if center_stable_steps >= REQUIRED_STABLE_STEPS:
        done = True
        reward += 200  # 成功奖励
        print("[成功] 小球稳定在中心，完成任务")

    # 直接更新Q值
    if state not in Q_table:
        Q_table[state] = np.zeros(len(ACTIONS))
    if next_state not in Q_table:
        Q_table[next_state] = np.zeros(len(ACTIONS))

    action_index = int(np.where(ACTIONS == action)[0][0])
    best_next_action_index = int(np.argmax(Q_table[next_state]))
    td_target = reward + GAMMA * Q_table[next_state][best_next_action_index]
    td_error = td_target - Q_table[state][action_index]
    Q_table[state][action_index] += ALPHA * td_error

    # 执行经验回放
    replay_experience()

    # print(f"[反馈] pos={position:.3f}, 新状态: {next_state}, 奖励: {reward:.2f}, 探索率: {EPSILON:.4f}, 串口={action:.2f}")

    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

    timestamp = time.time() - start_time
    q_value = Q_table[state][action_index] if state in Q_table else 0
    csv_writer.writerow([timestamp, reward, position, velocity, acceleration, action, q_value])
    csv_file.flush()

running = True

def training_loop():
    while running:
        sensor_data = receive_from_stm32()
        if sensor_data:
            pos, vel, acc = sensor_data
            train_step(pos, vel, acc)

# --------------------------- 主循环 ----------------------------
print("开始训练，按 Ctrl+C 停止")

try:
    t = threading.Thread(target=training_loop)
    t.start()

    while True:
        time.sleep(1)  # 保持主线程运行，可用于监控或后续拓展

except KeyboardInterrupt:
    running = False
    t.join()
    csv_file.close()
    metrics_file.close()
    print("训练结束，关闭串口")
    ser.close()