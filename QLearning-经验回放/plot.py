import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('ggplot')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
fig.suptitle('Cricket System Training Monitor', fontsize=14)

def init():
    ax1.clear()
    ax2.clear()
    ax3.clear()
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position', color='blue')
    ax1.grid(True)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Value')
    ax2.grid(True)
    
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Value')
    ax3.grid(True)
    
    return fig,

def update(frame):
    try:
        with open('log.csv', newline='') as f:
            reader = csv.DictReader(f)
            log_data = list(reader)
            times = [float(row['time']) for row in log_data]
            rewards = [float(row['reward']) for row in log_data]
            positions = [float(row['position']) for row in log_data]
            velocities = [float(row['velocity']) for row in log_data]
            accelerations = [float(row['acceleration']) for row in log_data]
            epsilons = [float(row['epsilon']) for row in log_data]
        
        with open('metrics.csv', newline='') as f:
            reader = csv.DictReader(f)
            metrics_data = list(reader)
            steps = [int(row['step']) for row in metrics_data]
            avg_rewards = [float(row['avg_reward']) for row in metrics_data]
            center_rates = [float(row['center_rate']) for row in metrics_data]
            avg_actions = [float(row['avg_action']) for row in metrics_data]

        ax1.clear()
        ax2.clear()
        ax3.clear()

        time_window = 30  # seconds
        if times:
            t_end = times[-1]
            t_start = t_end - time_window
            indices = [i for i, t in enumerate(times) if t >= t_start]
        else:
            indices = []

        # Ax1: Position & Reward
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position', color='blue')
        ax1.grid(True)

        if indices:
            t_subset = [times[i] for i in indices]
            pos_subset = [positions[i] for i in indices]
            reward_subset = [rewards[i] for i in indices]

            line_pos, = ax1.plot(t_subset, pos_subset, 'b-', label='Position')
            ax1.set_xlim(t_subset[0], t_subset[-1])
            ax1.set_ylim(min(pos_subset) - 0.1, max(pos_subset) + 0.1)
            ax1.tick_params(axis='y', labelcolor='blue')

            ax1b = ax1.twinx()
            ax1b.set_ylabel('Reward', color='red')
            line_reward, = ax1b.plot(t_subset, reward_subset, 'r-', label='Reward')
            ax1b.tick_params(axis='y', labelcolor='red')
            ax1b.set_ylim(min(reward_subset) - 0.1, max(reward_subset) + 0.1)

            ax1.set_title(f'Position & Reward (Latest: pos={pos_subset[-1]:.3f}, reward={reward_subset[-1]:.2f})')
            ax1.legend([line_pos, line_reward], ['Position', 'Reward'], loc='upper left')

        # # Ax2: Velocity, Acceleration, Epsilon
        # if indices:
        #     v_subset = [velocities[i] for i in indices]
        #     a_subset = [accelerations[i] for i in indices]
        #     e_subset = [epsilons[i] for i in indices]

        #     ax2.plot(t_subset, v_subset, color='green', label='Velocity')
        #     ax2.plot(t_subset, a_subset, color='magenta', label='Acceleration')
        #     ax2.plot(t_subset, e_subset, color='orange', label='Epsilon')
        #     ax2.set_xlim(t_subset[0], t_subset[-1])
        #     ax2.set_title(f'Dynamic Parameters (v={v_subset[-1]:.3f}, a={a_subset[-1]:.3f}, ε={e_subset[-1]:.3f})')
        #     ax2.legend()

        # Ax3: Training Metrics (full range)
        if steps:
            ax3.plot(steps, avg_rewards, color='blue', label='Avg Reward')
            ax3.plot(steps, center_rates, color='green', label='Center Rate')
            ax3.plot(steps, avg_actions, color='red', label='Avg Action')
            ax3.set_title(f'Training Metrics (Step={steps[-1]}, Avg Reward={avg_rewards[-1]:.2f}, Center Rate={center_rates[-1]:.2%})')
            ax3.legend()
            ax3.grid(True)

        plt.tight_layout()

    except Exception as e:
        print(f"Update error: {e}")
    
    return fig,


ani = FuncAnimation(fig, update, frames=None, 
                   init_func=init, blit=False, interval=1000, cache_frame_data=False)

plt.tight_layout()
plt.show()
