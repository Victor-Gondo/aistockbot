import pandas as pd
import datetime as datetime
import cryptocompare
import csv
import numpy as np
import random
from collections import deque
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from alpaca.trading.client import TradingClient
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Attention
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
     

cryptocompare.cryptocompare._set_api_key_parameter("8632283f7e59e6054da95d4fc2728404b2931eee776abce0d69d86af35a86e54")

#The limit to request is 999 hours
data = cryptocompare.get_historical_price_hour('ETH', 'USD', limit=999, exchange='CCCAGG', toTs=datetime.datetime.now())

#Removing unwanted values
headers_to_exclude = ['conversionType', 'conversionSymbol']
cleaned_data = [{k: v for k, v in item.items() if k not in headers_to_exclude} for item in data]

csv_file = "/content/eth_1_year_data.csv"

#Writing data to the csv
with open(csv_file, 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=cleaned_data[0].keys())
    writer.writeheader()
    writer.writerows(cleaned_data)
file.close()

#Using data from the CSV to calculate sma and adding it to the csv
df = pd.read_csv('/content/eth_1_year_data.csv')
df['50sma'] = df['close'].rolling(50).mean()
df['20sma'] = df['close'].rolling(20).mean()

# Drop rows with NaN values after calculating SMAs
df.dropna(inplace=True)

df.to_csv(csv_file)
     

# Load data
csv_file = "/content/eth_1_year_data.csv"
df = pd.read_csv(csv_file)

# Assuming 'close' is the feature we want to predict
# Let's also include 'high', 'low', 'open', 'volumefrom', 'volumeto', '20sma', and '50sma' as features
features = ['high', 'low', 'open', 'volumefrom', 'volumeto', 'close', '20sma', '50sma']
df[features] = df[features].fillna(method='ffill')  # Forward fill for missing values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[features].values)

# Function to create sequences
def create_sequences(data, time_steps=20):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i])
        y.append(data[i, 0])  # Assuming 'close' price is at index 5
    return np.array(X), np.array(y)

# Create sequences
time_steps = 20  # Number of time steps you're looking back to predict the future
X, y = create_sequences(scaled_data, time_steps)

# Split the data
split = int(0.8 * len(X))  # 80% for training and 20% for testing
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# LSTM expects input to be in the shape of [samples, time steps, features]
input_shape = (X_train.shape[1], X_train.shape[2])

# Define the LSTM model
def lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    lstm_out, hidden_state, cell_state = LSTM(50, activation='relu', return_sequences=True, return_state=True)(inputs)
    lstm_out = Dropout(0.2)(lstm_out)
    query_value_attention_seq = Attention()([lstm_out, lstm_out])
    query_value_attention_concat = Concatenate(axis=-1)([lstm_out, query_value_attention_seq])
    lstm_out_2 = LSTM(50, activation='relu', return_sequences=False)(query_value_attention_concat)
    lstm_out_2 = Dropout(0.2)(lstm_out_2)
    output = Dense(1, activation='linear')(lstm_out_2)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    return model

# Build the model
attention_lstm_model = lstm_model(input_shape)

# Train the model
history = attention_lstm_model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Save the trained LSTM model
attention_lstm_model.save('attention_lstm_model.h5')

# Evaluate the model
loss = attention_lstm_model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {loss}')
     

# After training, use the most recent data to make a prediction
last_sequence = X_test[-1].reshape((1, time_steps, X_test.shape[2]))
predicted_price_scaled = attention_lstm_model.predict(last_sequence)

# Invert the scaling for the predicted price
predicted_price = scaler.inverse_transform(np.concatenate((predicted_price_scaled, np.zeros((predicted_price_scaled.shape[0], len(features)-1))), axis=1))[:,0]

# Fetch the current price of ETH from CryptoCompare
current_price_data = cryptocompare.get_price('ETH', currency='USD')
current_price = current_price_data['ETH']['USD']

# Calculate the prediction error
percentage_error = abs(current_price - predicted_price) / current_price * 100

# Output the results
print(f"Predicted ETH Price: {predicted_price[0]} USD")
print(f"Current ETH Price: {current_price} USD")
print(f"Prediction Error: {percentage_error[0]:.2f}%")
     

# Initialize your API keys (these should be kept secret and not hardcoded in production)
trading_client = TradingClient('PK7LPFB3RFNB9XC9C9SR', 'qxtASZdlaDRJDDkQINDlRSVhsRSwPDBUO6L9bnG3')
     

class MarketEnvironment:
    def __init__(self, csv_file_path, initial_balance=100000, lookback_window_size=20):
        # Load historical data from CSV file
        self.data = pd.read_csv("/content/eth_1_year_data.csv")

        # Initialize environment parameters
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size

        # Determine state size
        self.state_size = (lookback_window_size, self.data.shape[1] + 2)

        # Initialize variables to keep track of state
        self.balance = initial_balance
        self.holdings = 0
        self.total_portfolio_value = self.balance
        self.done = False
        self.current_step = 0

        # Initialize any trading client or API if needed
        self.alpaca_api = trading_client

    def reset(self):
        # Reset environment state
        self.balance = self.initial_balance
        self.holdings = 0
        self.total_portfolio_value = self.balance
        self.current_step = 0
        self.done = False
        return self.get_state(self.current_step)

    def step(self, action):
        # Validate action
        assert action in [0, 1, 2]

        # Fetch current market data and update balance/holdings from Alpaca
        self._update_portfolio()

        current_price = self._get_current_price()
        reward = 0

        fixed_qty_to_trade = 1

        if action == 1:  # Buy
            # Implement buy action
            reward = self._buy(current_price, fixed_qty_to_trade)

        elif action == 2 and self.holdings > 0:  # Sell
            # Implement sell action
            qty_to_sell = min(self.holdings, fixed_qty_to_trade)
            reward = self._sell(current_price, qty_to_sell)

        # Update the state and check if done
        self.current_step += 1
        if self.current_step >= len(self.data):
            self.done = True

        next_state = self.get_state(self.current_step)
        return next_state, reward, self.done, {}

    def _update_portfolio(self):
        # Fetch the latest account info and update balance and holdings
        # Implement the logic based on your actual portfolio on Alpaca
        self.balance = float(self.alpaca_api.get_account().cash)
        # self.holdings needs to be updated based on the current holdings in Alpaca

    def _get_current_price(self):
        # Fetch the current market price from CryptoCompare
        return cryptocompare.get_price('ETH', currency='USD')['ETH']['USD']

    def _buy(self, current_price, qty):
        try:
            # Creating MarketOrderRequest object for buy order
            market_order_data = MarketOrderRequest(
            symbol="ETHUSD",
            qty=1,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC  # Good Till Cancelled
            )

            # Submit the market order
            market_order = self.alpaca_api.submit_order(order_data=market_order_data)
            print("Buy order submitted successfully:", market_order)
        except Exception as e:
            print(f"An error occurred while submitting the buy order: {e}")

            # Update balance and holdings
            self.balance -= 1 * current_price
            self.holdings += 1
            return -1

    def _sell(self, current_price, qty):
        try:
            # Creating MarketOrderRequest object for sell order
            market_order_data = MarketOrderRequest(
                symbol="ETHUSD",
                qty=1,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.GTC
            )

            # Submitting market order
            market_order = self.alpaca_api.submit_order(order_data=market_order_data)
            print("Sell order submitted successfully:", market_order)
        except Exception as e:
            print(f"An error occurred while submitting the sell order: {e}")

        # Update balance and holdings
        self.balance += self.holdings * current_price
        reward = self.balance - self.initial_balance
        self.holdings = 0
        return reward

    def send_test_order(self, symbol, qty, side, order_type, time_in_force):
        print(f"Sending a {side} order for {qty} shares of {symbol}...")
        try:
            if "paper-api" in self.alpaca_api._base_url:
                order = self.alpaca_api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type=order_type,
                    time_in_force=time_in_force
                )
                print("Order submitted successfully!")
                print(f"Order details: ID={order.id}, Status={order.status}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def get_state(self, current_step):
        # Determine the window of data to consider.
        window_start = max(current_step - self.lookback_window_size, 0)
        window_end = current_step + 1
        window_data = self.data.iloc[window_start:window_end].values  # Use .iloc for DataFrame

        # Debug statement to print shapes
        print(f"window_data shape: {window_data.shape}")

        # Create a 2D array for balance and holdings, repeat it to match the window_data's first dimension
        balance_and_holdings = np.array([[self.balance, self.holdings]] * window_data.shape[0])

        # Debug statement to print shapes
        print(f"balance_and_holdings shape: {balance_and_holdings.shape}")

        # Concatenate along the second axis (axis=1) to include balance and holdings in the state
        state = np.concatenate((window_data, balance_and_holdings), axis=1)

        # Debug statement to print final state shape
        print(f"state shape: {state.shape}")

        return state
     

class TradingAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Memory for experience replay
        self.gamma = 0.95  # Discount rate for future rewards
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration probability
        self.learning_rate = 0.001  # Learning rate
        self.model = self._build_model()  # The DQN model
        self.lstm_model = load_model('attention_lstm_model.h5')

    def _build_model(self):
        """Builds a NN model."""
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Stores experiences in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Selects an action based on the current state."""
        if random.uniform(0, 1) <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # Returns the action with the highest Q-value

    def replay(self, batch_size):
        """Trains the agent by replaying experiences from memory."""
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """Loads a saved model."""
        self.model.load_weights(name)

    def save(self, name):
        """Saves the model."""
        self.model.save_weights(name)
     

# MAIN LOOP

# Initialize the market environment
initial_balance = 100000  # Set an initial balance, for example, $100,000
lookback_window_size = 1  # The number of past time points to consider for the state

# Initialize the market environment
market_environment = MarketEnvironment(data, initial_balance, lookback_window_size)

# Define the size of the state and action space
state_size = market_environment.state_size[0] * market_environment.state_size[1]
action_size = 3  # Assuming three actions: [hold, buy, sell]

# Initialize the trading agent
trading_agent = TradingAgent(state_size, action_size)

# How many episodes we want the agent to train for
num_episodes = 150 # Tweak according to performance

# How many timesteps per episode
timesteps_per_episode = len(data) - lookback_window_size
print(f"timesteps_per_episode: {timesteps_per_episode}") # Currently 999 hours or 41.625 days of historical data

# Batch size for experience replay
batch_size = 32 # 32 is a commonly used default because it's been found to be a good trade-off between training speed and model update stability

for e in range(num_episodes):
    # Reset the market environment at the start of each episode
    state = market_environment.reset()

    # Flatten the state to feed into the neural network
    state = np.reshape(state, (1, -1))  # This will ensure the state is a 2D array with the correct shape

    for time in range(timesteps_per_episode):
        # The agent takes action
        action = trading_agent.act(state)

        # Apply the action to the market environment to get the next state and reward
        next_state, reward, done, _ = market_environment.step(action)

        # Remember the previous state, action, reward, and next state
        trading_agent.remember(state, action, reward, next_state, done)

        # Make the next_state the new current state for the next frame.
        state = next_state

        # If done, exit the loop
        if done:
            print(f"Episode: {e + 1}/{num_episodes}, Total Portfolio Value: {market_environment.total_portfolio_value}, P/L: {market_environment.total_portfolio_value - initial_balance}")
            break

        # Train the agent with experiences in replay memory
        if len(trading_agent.memory) > batch_size:
            trading_agent.replay(batch_size)

    # Optionally, you can save the model every X episodes
    if e % 10 == 0:
        trading_agent.save(f"trading_agent_{e}.h5")

# Testing the buy/sell functions here
# Initializing our API keys
trading_client = TradingClient('PK7LPFB3RFNB9XC9C9SR', 'qxtASZdlaDRJDDkQINDlRSVhsRSwPDBUO6L9bnG3', base_url='https://paper-api.alpaca.markets')

# Ensuring the market_environment is using the trading_client instance with our API keys
market_environment.alpaca_api = trading_client

# Example usage
csv_file_path = '/content/eth_1_year_data.csv'
market_environment = MarketEnvironment(csv_file_path, initial_balance=200000, lookback_window_size=20)
market_environment.send_test_order(
    symbol='ETHUSD',
    qty=1,
    side='buy',
    order_type='market',
    time_in_force='gtc'
)
market_environment.send_test_order(
    symbol='ETHUSD',
    qty=1,
    side='sell',
    order_type='market',
    time_in_force='gtc'
)

"""
MODEL TIMING

The total time per episode in seconds would be 0.024 seconds * 999 timesteps = 23.976 seconds.

For 150 episodes, the total time would be 23.976 seconds/episode * 150 episodes = 3596.4 seconds.

Converting this to hours: 3596.4 seconds / 3600 seconds/hour ≈ 0.999 hours.
"""
     

"""
HYPERPARAMETERS

Time Steps (time_steps): Used in the sequence creation function, it defines the number of historical data points the LSTM model will consider for predicting the next value. It's set to 20, meaning the model looks at the last 20 data points to make a prediction.

Data Split (split): Determines the ratio of training to testing data. It's set to 80% for training and the remaining 20% for testing.

LSTM Units: Inside the lstm_model function, LSTM layers with 50 units are defined, which signifies the dimensionality of the output space.

Dropout Rate: Dropout layers with a rate of 0.2 are used to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.

Dense Layer Units: The dense layers in the LSTM model have 24 units each, which is another hyperparameter defining the output dimension of the layer.

Batch Size for Training LSTM Model (batch_size): The LSTM model is trained with a batch size of 32, which means 32 sequences are passed through the network at once.

Epochs for Training LSTM Model: The model is trained for 100 epochs, where one epoch means that the entire dataset is passed forward and backward through the neural network once.

Lookback Window Size (lookback_window_size): Set to 20, this defines how many past observations the trading environment will return as the state.

State Size (state_size): Calculated based on the lookback_window_size and the number of features in the data. It represents the shape of the input that the trading agent will receive.

Gamma (gamma): This is the discount factor for future rewards in the DQN agent, set to 0.95.

Epsilon (epsilon): The exploration rate for the agent, starting at 1.0 (100% exploration).

Epsilon Minimum (epsilon_min): The minimum exploration rate, set to 0.01.

Epsilon Decay (epsilon_decay): The rate at which the exploration rate decays, set to 0.995.

Learning Rate (learning_rate): For the Adam optimizer in the DQN, it's set to 0.001.

Number of Episodes (num_episodes): The number of episodes for training the DQN model. It's initially set to 1800, which indicates how many times the training process will iterate over the entire dataset.

Timesteps Per Episode (timesteps_per_episode): The number of steps the agent will take in the environment per episode. It's initially logged as 999, which is derived from the available historical price hours minus the lookback_window_size.
"""
