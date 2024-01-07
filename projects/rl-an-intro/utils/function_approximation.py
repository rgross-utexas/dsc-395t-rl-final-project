from abc import ABCMeta, abstractmethod
from functools import reduce
from typing import Tuple

import numpy as np


class StateValueFunctionWithApproximation(object):
    def __call__(self, state: np.array) -> float:
        """
        Return the value of given state

        input:
            state: state apply get the value for
        return:
            value of the given state
        """
        raise NotImplementedError()

    def update(self, alpha: float, g: float, state: np.array):
        """
        Perform the update

        input:
            alpha: learning rate
            g: reward
            state: target state for updating
        return:
            None
        """
        raise NotImplementedError()


class StateValueFunctionWithTile(StateValueFunctionWithApproximation):

    def __init__(self,
                 state_low: np.array,
                 state_high: np.array,
                 num_tilings: int,
                 tile_width: np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """

        self.state_low = state_low
        self.state_high = state_high
        self.num_tilings = num_tilings
        self.tile_width = tile_width
        self.tiling_lows = [self.state_low - i / self.num_tilings * self.tile_width for i in range(self.num_tilings)]
        self.tiling_shape = (np.ceil((state_high - state_low) / tile_width) + 1).astype(int)
        self.weights = np.zeros([num_tilings] + self.tiling_shape.tolist())

    def __call__(self, state: np.array) -> float:
        """
        Return the value of given state

        input:
            state: state apply get the value for
        return:
            value of the given state
        """
        return self._get_feature_weight(state)

    def update(self, alpha: float, g: float, state: np.array):
        """
        Apply update rule

        input:
            alpha: learning rate
            g: reward
            state: target state for updating
        return:
            None
        """

        # the feature mask acts as the gradient
        feature_weight, feature_mask = self._get_feature_weight_and_mask(state)

        # w <- w + \alpha[G - \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)
        error = g - feature_weight
        delta = alpha * error * feature_mask
        self.weights += delta

    def _get_feature_weight(self, state: np.array) -> float:
        return self._get_feature_weight_and_mask(state)[0]

    def _get_feature_weight_and_mask(self, state: np.array) -> Tuple[float, np.array]:
        state = np.around(state, 5)
        tiles = self._map_state_to_tiles(state)
        mask = np.zeros(self.weights.shape)
        flat_weight_indices = np.ravel_multi_index(tiles.T, mask.shape)
        np.ravel(mask)[flat_weight_indices] = 1
        feature_weight = np.sum(self.weights * mask) / self.num_tilings

        return feature_weight, mask

    def _map_state_to_tiles(self, state: np.array) -> np.array:
        tiles = ((state - self.tiling_lows) / self.tile_width).astype(int)
        tiling_indices = np.reshape(np.arange(self.num_tilings).T, [-1, 1])

        return np.hstack((tiling_indices, tiles))


class StateActionFeatureVector(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, state: np.array, action: int, done:bool) -> np.array:
        raise NotImplementedError()


class StateActionFeatureVectorWithTile(StateActionFeatureVector):

    def __init__(self,
                 state_low: np.array,
                 state_high: np.array,
                 num_actions: int,
                 num_tilings: int,
                 num_tile_parts: int):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        num_tile_parts: number of parts to divide each tile by
        """

        self.state_low = state_low
        self.state_high = state_high
        self.num_actions = num_actions
        self.num_tilings = num_tilings
        self.tile_width = (self.state_high - self.state_low) / num_tile_parts
        self.tiling_lows = [self.state_low - i / self.num_tilings * self.tile_width for i in range(self.num_tilings)]
        self.tiling_shape = (np.ceil((self.state_high - self.state_low) / self.tile_width) + 1).astype(int)

        weight_shape = [self.num_actions] + [self.num_tilings] + self.tiling_shape.tolist()
        self.weights = np.random.normal(loc=0, scale=2, size=tuple(weight_shape))

        self.feature_vector_length = reduce(lambda x, y: x * y, self.weights.shape)
        self.done_vector = np.zeros([self.feature_vector_length, 1])

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return self.feature_vector_length

    def __call__(self, state: np.array, action: int, done: bool) -> np.array:

        if done:
            return self.done_vector

        weights = self._get_feature_action_weights(state, action)
        flat_weights = np.reshape(weights, [-1, 1])

        return flat_weights

    def _get_feature_action_weights(self, state: np.array, action: int) -> np.array:

        if isinstance(state, tuple):
            state = state[0]

        state = np.around(state, 5)
        tiles = self._map_state_to_tiles(state)
        mask = np.zeros(self.weights[action].shape)
        flat_weight_indices = np.ravel_multi_index(tiles.T, mask.shape)

        np.ravel(mask)[flat_weight_indices] = 1

        weights = np.zeros_like(self.weights)
        for a in range(self.num_actions):
            if a == action:
                weights[a] = self.weights[a] * mask
            else:
                weights[a] = np.zeros_like(self.weights[a])

        return weights

    def _map_state_to_tiles(self, state: np.array) -> np.array:

        tiles = ((state - self.tiling_lows) / self.tile_width).astype(int)
        tiling_indices = np.reshape(np.arange(self.num_tilings), [-1, 1])

        return np.hstack((tiling_indices, tiles))
