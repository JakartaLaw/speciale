"""Deep Q function iteration"""

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pickle

from agents.utils import ArrayScaler


class MemoryBuffer:
    def __init__(
        self, action_size, state_size, mem_size=1500000, batch_size=50000, **kwargs
    ):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.batch_size = batch_size
        self.action_size = action_size
        self.state_size = state_size

        self.state_memory = np.zeros((self.mem_size, state_size))
        self.q_memory = np.zeros((self.mem_size, action_size))

    def store_transition(self, state, q_values):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.q_memory[index] = q_values

        self.mem_cntr += 1

    def sample_batch(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        _batch_size = min(self.mem_cntr, self.batch_size)
        batch = np.random.choice(max_mem, _batch_size)

        states = self.state_memory[batch]
        q_values = self.q_memory[batch]

        return states, q_values


def build_network(input_dims, output_dims, fc1_size, fc2_size, lr):
    model = Sequential()
    model.add(Dense(fc1_size, activation="relu", input_dim=input_dims))
    model.add(Dense(fc2_size, activation="relu"))
    model.add(Dense(output_dims, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(lr=lr))
    return model


class DQIterationAgent:
    def __init__(
        self,
        alpha,
        gamma,
        layer1_size,
        layer2_size,
        statespace_dims,
        action_dims,
        state_scaler_mu,
        state_scaler_sigma,
        **kwargs,
    ):
        self.lr = alpha
        self.gamma = gamma
        self.fc1_size = layer1_size
        self.fc2_size = layer2_size
        self.statespace_dims = statespace_dims
        self.action_dims = action_dims
        self.q_network = build_network(
            self.statespace_dims,
            self.action_dims,
            self.fc1_size,
            self.fc2_size,
            self.lr,
        )

        self.memory_buffer = MemoryBuffer(action_dims, statespace_dims, **kwargs)
        self.q_scaler = ArrayScaler(as_scalar=True)
        self.state_scaler = ArrayScaler()
        self.state_scaler.mu = state_scaler_mu
        self.state_scaler.sigma = state_scaler_sigma

    def save_model(self, name="dqfi_model"):

        self.q_network.save(name + ".h5", save_format="h5")
        q_scaler_params = {"mu": self.q_scaler.mu, "sigma": self.q_scaler.sigma}
        print(q_scaler_params)
        with open(name + ".q_scaler.pkl", "wb") as f:
            pickle.dump(q_scaler_params, f)

    def load_model(self, name="dqfi_model"):
        self.q_network = load_model(name + ".h5",)
        with open(name + ".q_scaler.pkl", "rb") as f:
            q_scaler_params = pickle.load(f)

        self.q_scaler.mu = q_scaler_params["mu"]
        self.q_scaler.mu = q_scaler_params["sigma"]

    def sample(self):
        states, q_vals = self.memory_buffer.sample_batch()
        self.q_scaler.fit(q_vals)
        q_vals, states = (
            self.q_scaler.transform(q_vals),
            self.state_scaler.transform(states),
        )
        return states, q_vals

    def learn(self, verbose=True, rebuild_network=True):

        states, q_vals = self.sample()

        if rebuild_network:
            print("rebuilding network")
            self.q_network = build_network(
                self.statespace_dims,
                self.action_dims,
                self.fc1_size,
                self.fc2_size,
                self.lr,
            )

        history_ = self.q_network.fit(
            states,
            q_vals,
            epochs=150,
            validation_split=0.3,
            callbacks=[EarlyStopping(patience=5)],
            verbose=0,
        )

        n_epochs = history_.epoch[-1]
        loss_, val_loss_ = (
            history_.history["loss"][-1],
            history_.history["val_loss"][-1],
        )
        scaled_val_loss_ = np.float(np.mean(self.q_scaler.sigma) * val_loss_)

        if verbose:
            print(
                f"# epochs: {n_epochs}, loss: {loss_:.3}, val loss: {val_loss_:.3}, scaled val loss: {scaled_val_loss_:.3}"
            )

    def remember(self, state, q_values):
        self.memory_buffer.store_transition(state, q_values)

    def act(self, state):
        state = self.state_scaler.transform(state)
        state = state[np.newaxis, :]
        q_values = self.q_network.predict(state)
        action, q_value = np.argmax(q_values), self.q_scaler.inverse_transform(q_values)
        return action, q_value

    def predict_q_vals(self, new_states, rewards, done):
        rewards = rewards[np.newaxis, :]
        new_states = self.state_scaler.transform(new_states)

        if done:
            q_vals = rewards
        else:
            q_vals_next = self.q_network.predict(new_states)
            q_vals_next = self.q_scaler.inverse_transform(q_vals_next)

            # the correct axis should be axis=1
            q_vals_next_star = np.max(q_vals_next, axis=1)
            # q_vals_next_star = np.max(q_vals_next, axis=0)

            q_vals = rewards + self.gamma * q_vals_next_star

        return q_vals
