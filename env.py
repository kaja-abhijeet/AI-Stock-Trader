import numpy as np

class TradingEnv:
    def __init__(self, prices):
        # ✅ FIX: force 1D float array
        self.prices = np.array(prices, dtype=np.float64).flatten()
        self.n = len(self.prices)
        self.reset()

    def reset(self):
        self.t = 0
        self.cash = 10000.0
        self.stock = 0.0
        self.total_asset = self.cash
        return self._get_state()

    def _get_state(self):
        price = self.prices[self.t]  # already float
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