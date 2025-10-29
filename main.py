import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from tqdm import tqdm
from env import TradingEnv
from q_learning import QLearningAgent


print("Downloading stock data...")
data = yf.download("AAPL", start="2022-01-01", end="2024-01-01")
prices = data["Close"].values


env = TradingEnv(prices)
state_size = 2
action_size = 3
agent = QLearningAgent(state_size, action_size)

EPISODES = 400
rewards = []

print("Training agent...")
for episode in tqdm(range(EPISODES)):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state

    rewards.append(total_reward)

plt.figure(figsize=(10,5))
plt.plot(rewards)
plt.title("Training Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()


print("Evaluating trained model...")
state = env.reset()
done = False
actions = []
while not done:
    action = agent.get_action(state, explore=False)
    next_state, _, done, _ = env.step(action)
    actions.append(action)
    state = next_state

actions = np.array(actions)
prices = prices[:len(actions)]

plt.figure(figsize=(10, 5))
plt.plot(prices, label="Stock Price", color='blue')
plt.scatter(np.where(actions == 1)[0], prices[actions == 1], color='green', label='Buy', marker='^')
plt.scatter(np.where(actions == 2)[0], prices[actions == 2], color='red', label='Sell', marker='v')
plt.legend()
plt.title("Trading Actions on AAPL Stock Data")
plt.xlabel("Time Step")
plt.ylabel("Price ($)")
plt.show()
