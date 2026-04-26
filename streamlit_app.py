import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="AI Stock Trader", layout="wide")
st.title("💹 AI Stock Trading Bot (Q-Learning)")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("⚙️ Settings")

symbol = st.sidebar.text_input("Stock Symbol", "AAPL")

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))

episodes = st.sidebar.slider("Training Episodes", 10, 500, 100)
epsilon = st.sidebar.slider("Exploration Rate (ε)", 0.01, 1.0, 0.1)
alpha = st.sidebar.slider("Learning Rate (α)", 0.01, 1.0, 0.1)
gamma = st.sidebar.slider("Discount Factor (γ)", 0.1, 0.99, 0.9)

start_training = st.sidebar.button("🚀 Train the Agent")

# Show current settings
st.write("### Current Settings")
st.write({
    "Episodes": episodes,
    "Epsilon": epsilon,
    "Alpha": alpha,
    "Gamma": gamma
})

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    data = data[["Close"]].dropna()
    return data

data = load_data(symbol, start_date, end_date)

# ✅ FIX: force 1D float array
prices = data["Close"].to_numpy().flatten()

st.line_chart(data["Close"], height=250, use_container_width=True)

# -----------------------------
# Trading Environment
# -----------------------------
class TradingEnv:
    def __init__(self, prices):
        self.prices = np.array(prices, dtype=np.float64).flatten()
        self.n = len(self.prices)
        self.reset()

    def reset(self):
        self.t = 0
        self.cash = 10000.0
        self.stock = 0
        self.total_asset = self.cash
        return self._get_state()

    def _get_state(self):
        price = self.prices[self.t]
        return np.array([price, self.stock], dtype=np.float64)

    def step(self, action):
        price = self.prices[self.t]
        done = False

        # 0 = Hold, 1 = Buy, 2 = Sell
        if action == 1 and self.cash > 0:
            self.stock += self.cash / price
            self.cash = 0.0

        elif action == 2 and self.stock > 0:
            self.cash += self.stock * price
            self.stock = 0.0

        self.t += 1
        if self.t >= self.n - 1:
            done = True

        new_total = self.cash + self.stock * price
        reward = new_total - self.total_asset
        self.total_asset = new_total

        return self._get_state(), reward, done, {}

# -----------------------------
# Training
# -----------------------------
if start_training:

    env = TradingEnv(prices)
    q_table = {}
    rewards = []

    progress_bar = st.progress(0)

    for ep in range(episodes):
        state = tuple(np.round(env.reset(), 2))
        total_reward = 0

        while True:
            # ε-greedy policy
            if np.random.rand() < epsilon:
                action = np.random.choice([0, 1, 2])
            else:
                action = np.argmax(q_table.get(state, [0, 0, 0]))

            next_state, reward, done, _ = env.step(action)
            next_state = tuple(np.round(next_state, 2))

            # Q-learning update
            old_q = q_table.get(state, [0, 0, 0])[action]
            next_max = np.max(q_table.get(next_state, [0, 0, 0]))

            new_q = old_q + alpha * (reward + gamma * next_max - old_q)

            if state not in q_table:
                q_table[state] = [0, 0, 0]

            q_table[state][action] = new_q

            state = next_state
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)
        progress_bar.progress((ep + 1) / episodes)

    st.success("✅ Training Completed!")

    # -----------------------------
    # Evaluation
    # -----------------------------
    env = TradingEnv(prices)
    state = tuple(np.round(env.reset(), 2))

    actions = []
    portfolio_values = []

    while True:
        action = np.argmax(q_table.get(state, [0, 0, 0]))
        actions.append(action)

        next_state, _, done, _ = env.step(action)

        portfolio_value = env.cash + env.stock * env.prices[env.t]
        portfolio_values.append(float(portfolio_value))

        state = tuple(np.round(next_state, 2))

        if done:
            break

    final_value = portfolio_values[-1]
    buy_hold_value = (prices[-1] / prices[0]) * 10000

    # -----------------------------
    # Results
    # -----------------------------
    st.subheader("📊 Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("🤖 AI Trader", f"${final_value:,.2f}")
    col2.metric("📈 Buy & Hold", f"${buy_hold_value:,.2f}")
    col3.metric("💰 Profit", f"${(final_value - 10000):,.2f}")

    # -----------------------------
    # Plot trades
    # -----------------------------
    actions = np.array(actions)
    price_array = prices[:len(actions)]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(price_array, label="Price")

    ax.scatter(np.where(actions == 1),
               price_array[actions == 1], label="Buy", marker="^")

    ax.scatter(np.where(actions == 2),
               price_array[actions == 2], label="Sell", marker="v")

    ax.legend()
    ax.set_title(f"{symbol} Trading Decisions")

    st.pyplot(fig)

    # Portfolio chart
    st.line_chart(portfolio_values, height=250, use_container_width=True)

else:
    st.info("👈 Adjust parameters and click 'Train the Agent'")