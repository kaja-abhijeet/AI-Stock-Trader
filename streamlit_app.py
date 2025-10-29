import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import time

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="AI Stock Trader", page_icon="💹", layout="wide")

st.title("💹 AI Stock Trading Bot using Q-Learning")
st.markdown("A simple reinforcement learning demo using **Q-Learning** on real stock data.")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
symbol = st.sidebar.text_input("Stock Symbol (e.g. AAPL, TSLA, MSFT)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))
episodes = st.sidebar.slider("Training Episodes", 10, 500, 100)
epsilon = st.sidebar.slider("Exploration Rate (Epsilon)", 0.01, 1.0, 0.1)
alpha = st.sidebar.slider("Learning Rate (Alpha)", 0.01, 1.0, 0.1)
gamma = st.sidebar.slider("Discount Factor (Gamma)", 0.1, 0.99, 0.9)

st.sidebar.markdown("---")
st.sidebar.info("👆 Adjust parameters and click below to start training.")
start_training = st.sidebar.button("🚀 Train the Agent")

# -----------------------------
# Download Stock Data
# -----------------------------
@st.cache_data
def load_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    data = data[["Close"]].dropna()
    return data

data = load_data(symbol, start_date, end_date)
prices = data["Close"].values

st.line_chart(data["Close"], height=250, use_container_width=True)
st.caption(f"Showing {len(prices)} days of data for **{symbol}**")

# -----------------------------
# Define Trading Environment
# -----------------------------
class TradingEnv:
    def __init__(self, prices):
        self.prices = np.array(prices)
        self.reset()

    def reset(self):
        self.cash = 10000.0
        self.stock = 0
        self.step_idx = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        price = float(self.prices[self.step_idx])
        return np.array([price, self.cash, self.stock], dtype=float)

    def step_env(self, action):
        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        price = float(self.prices[self.step_idx])
        reward = 0

        if action == 1 and self.cash >= price:
            self.stock += 1
            self.cash -= price
        elif action == 2 and self.stock > 0:
            self.stock -= 1
            self.cash += price
            reward = price  # profit for selling

        self.step_idx += 1
        if self.step_idx >= len(self.prices) - 1:
            self.done = True

        return self._get_state(), reward, self.done, {}

# -----------------------------
# Train Q-Learning Agent
# -----------------------------
if start_training:
    env = TradingEnv(prices)
    q_table = {}
    rewards = []

    progress_bar = st.progress(0)
    status = st.empty()

    for ep in range(episodes):
        state = tuple(np.round(env.reset(), 2))
        total_reward = 0

        while not env.done:
            if np.random.rand() < epsilon:
                action = np.random.choice([0, 1, 2])
            else:
                action = np.argmax(q_table.get(state, [0, 0, 0]))

            next_state, reward, done, _ = env.step_env(action)
            next_state = tuple(np.round(next_state, 2))
            total_reward += reward

            old_q = q_table.get(state, [0, 0, 0])[action]
            next_max = np.max(q_table.get(next_state, [0, 0, 0]))
            new_q = old_q + alpha * (reward + gamma * next_max - old_q)

            if state not in q_table:
                q_table[state] = [0, 0, 0]
            q_table[state][action] = new_q

            state = next_state

        rewards.append(total_reward)
        progress_bar.progress((ep + 1) / episodes)
        status.text(f"Training Episode {ep + 1}/{episodes} | Reward: {total_reward:.2f}")

    st.success("✅ Training Completed!")

    # -----------------------------
    # Evaluate Agent
    # -----------------------------
    env = TradingEnv(prices)
    state = tuple(np.round(env.reset(), 2))
    actions = []
    portfolio_values = []

    while not env.done:
        action = np.argmax(q_table.get(state, [0, 0, 0]))
        actions.append(action)
        next_state, reward, done, _ = env.step_env(action)
        portfolio_value = env.cash + env.stock * env.prices[env.step_idx]
        portfolio_values.append(float(portfolio_value))
        state = tuple(np.round(next_state, 2))

    final_value = float(np.squeeze(portfolio_values[-1]))
    buy_hold_value = float(env.prices[-1] / env.prices[0] * 10000)

    # -----------------------------
    # Display Results
    # -----------------------------
    st.subheader("📊 Performance Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("🤖 AI Trader Value", f"${final_value:,.2f}")
    col2.metric("📉 Buy & Hold Value", f"${buy_hold_value:,.2f}")
    col3.metric("🏆 Net Gain", f"${(final_value - 10000):,.2f}")

    st.write("---")

    fig, ax = plt.subplots(figsize=(10, 5))
    actions_array = np.array(actions)
    price_array = np.array(prices[:len(actions)])

    ax.plot(price_array, label="Stock Price", color="blue")
    ax.scatter(np.where(actions_array == 1), price_array[actions_array == 1],
               label="Buy", color="green", marker="^")
    ax.scatter(np.where(actions_array == 2), price_array[actions_array == 2],
               label="Sell", color="red", marker="v")
    ax.set_title(f"{symbol} — Q-Learning Trading Decisions")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    st.line_chart(portfolio_values, height=250, use_container_width=True)
    st.caption("Portfolio Value over Time")

    st.success("✅ Simulation Complete! Check out the charts above.")
