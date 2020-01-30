from abc import ABC, abstractmethod


class InterfaceEnvironment(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def states(self):
        """The representation of the state space"""
        pass

    @abstractmethod
    def step(self, action):
        """Takes a step. Uses action for the step

        returns
        =======


        """
        pass

    @abstractmethod
    def reset(self):
        """resets environment. Defaults back to starting period.
        reset method also can take a state (used for solving the model)"""
        pass
