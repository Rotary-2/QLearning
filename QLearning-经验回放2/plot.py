import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

plt.style.use('ggplot')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
fig.suptitle('Cricket System Training Monitor', fontsize=14)

ax1b = ax1.twinx()
ax1.pos_line, = ax1.plot([], [], 'b-', label='Position')
ax1.reward_line, = ax1b.plot([], [], 'r-', label='Reward')
ax1.q_line, = ax1b.plot([], [], 'g-', label='Q_value')

ax1.set_ylabel('Position', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1b.set_ylabel('Reward / Q_value', color='r')
ax1b.tick_params(axis='y', labelcolor='r')
ax1.set_xlabel('Time (s)')
ax1.set_title('Position, Reward, Q_value')
ax1.legend([ax1.pos_line, ax1.reward_line, ax1.q_line], ['Position', 'Reward', 'Q_value'], loc='upper left')

def init():
    ax2.clear()
    ax3.clear()
    ax2.grid(True)
    ax3.grid(True)
    return fig,

def update(frame):
    try:
        with open('log.csv', newline='') as f:
            reader = csv.DictReader(f)
            data = list(reader)
            if not data:
                return fig,
            
            times = [float(row['time']) for row in data]
            rewards = [float(row['reward']) for row in data]
            positions = [float(row['position']) for row in data]
            velocities = [float(row['velocity']) for row in data]
            actions = [float(row['action']) for row in data]
            q_values = [float(row['Q_value']) for row in data]

        # ax1.clear()
        ax2.clear()
        ax3.clear()

        # 图1：reward、position、Q_value，固定横坐标窗口实时滚动
        if times:
            t_window = 30
            t_end = times[-1]
            t_start = max(times[0], t_end - t_window)
            idx = [i for i, t in enumerate(times) if t_start <= t <= t_end]

            t_subset = [times[i] for i in idx]
            pos_subset = [positions[i] for i in idx]
            reward_subset = [rewards[i] for i in idx]
            q_subset = [q_values[i] for i in idx]

            ax1.pos_line.set_data(t_subset, pos_subset)
            ax1.set_xlim(t_start, t_end)
            ax1.set_ylim(min(pos_subset)-0.1, max(pos_subset)+0.1)

            ax1.reward_line.set_data(t_subset, reward_subset)
            ax1.q_line.set_data(t_subset, q_subset)

            all_values = reward_subset + q_subset
            y_min, y_max = min(all_values), max(all_values)
            y_range = y_max - y_min
            y_buffer = 0.1 * max(y_range, 1e-3)
            ax1b.set_ylim(y_min - y_buffer, y_max + y_buffer)

            plt.tight_layout()

        # --- 图2：position-velocity 网格，红色/绿色 ---
        grid_size = 20
        heat_red = np.zeros((grid_size, grid_size))
        heat_blue = np.zeros((grid_size, grid_size))

        for p, v, a, q in zip(positions, velocities, actions, q_values):
            px = int((p + 1) / 2 * (grid_size - 1))
            vy = int((v + 1) / 2 * (grid_size - 1))
            if a < 0:
                heat_red[vy][px] += q
            else:
                heat_blue[vy][px] += q

        ax2.imshow(heat_red, cmap='Reds', origin='lower', extent=[-1, 1, -1, 1], alpha=0.6)
        ax2.imshow(heat_blue, cmap='Greens', origin='lower', extent=[-1, 1, -1, 1], alpha=0.6)

        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-1, 1)

        ax2.set_xticks(np.arange(-1, 1.01, 0.1))
        ax2.set_yticks(np.arange(-1, 1.01, 0.1))

        # 这里设置宽高比，<1 横向拉宽，>1 纵向拉高，None 为自动
        ax2.set_aspect(0.5)  # 横轴显示加宽，纵轴保持

        ax2.set_title('Position-Velocity Q-map\nred(action<0) / green(action>0)')
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Velocity')

        # 图3保留 metrics.csv 可选（此处略）

        plt.tight_layout()

    except Exception as e:
        print(f"Update error: {e}")

    return fig,

ani = FuncAnimation(fig, update, frames=None, init_func=init, blit=False, interval=1000, cache_frame_data=False)
plt.tight_layout()
plt.show()
