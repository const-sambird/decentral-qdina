from dataclasses import dataclass
import gymnasium as gym
import numpy as np

@dataclass
class QIAProblem:
    name: str
    benefits: list[int]
    weights: list[int]

    def num_candidates(self) -> int:
        return len(self.benefits)
    
QIA_PROBLEMS = {
    'I5': QIAProblem('I5', [14, 12, 10, 8, 3], [7, 6, 4, 2, 1]),
    'I6': QIAProblem('I6', [14, 12, 10, 8, 3, 13], [7, 6, 4, 2, 1, 7]),
    'I7': QIAProblem('I7', [14, 12, 10, 8, 3, 13, 12], [7, 6, 4, 2, 1, 7, 6]),
    'CDB_I7': QIAProblem('CDB_I7', [165811, 178871, 1213770, 1213770, 1213770, 44370, 44370], [266, 232, 8, 132, 199, 2, 9])
}

class QIAEnvironment(gym.Env):
    def __init__(self, type: str, budget: int):
        '''
        This is a toy environment for the problem instances
        created for the evaluation of SQIA/OQIA by the paper by
        Kesarwani & Haritsa:
            M. Kesarwani and J. R. Haritsa, "Index advisors on quantum platforms," Proc.
            VLDB Endow., vol. 17, no. 11, p. 3615-3628, Jul. 2024. [Online]. Available:
            https://doi.org/10.14778/3681954.3682025
        
        The problem instances, called 'I5', 'I6', 'I7', and 'CDB_I7',
        have precomputed benefits and storage costs. This environment
        returns the reward and updates the space budget according to the
        hard-coded values.

        :param type:   which problem instance is this? one of 'I5', 'I6', 'I7', or 'CDB_I7'
        :param budget: the space budget, 19 in the paper
        '''
        assert type in QIA_PROBLEMS, 'unknown problem instance!'

        self.problem = QIA_PROBLEMS[type]
        self.n_candidates = self.problem.num_candidates()
        self.budget = budget

        self._state = np.zeros(self.n_candidates)
        
        self.observation_space = gym.spaces.MultiBinary(self.n_candidates)
        self.action_space = gym.spaces.Discrete(self.n_candidates * 2)
        self.action_drop_threshold = self.action_space.n // 2
        self._action_mask = np.ones((self.n_candidates * 2,), dtype=np.int8)
        self._action_mask[0:self.action_drop_threshold] = 0

    def _get_obs(self):
        return self._state

    def _get_info(self):
        return {
            'problem': self.problem,
            'mask': self._action_mask
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._state = np.zeros(self.n_candidates)
        self._action_mask = np.ones((self.n_candidates * 2,), dtype=np.int8)
        self._action_mask[0:self.action_drop_threshold] = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action: int):
        '''
        
        '''
        action_to_mask = action
        creating = action >= self.action_drop_threshold
        if creating:
            action = action - (self.action_space.n // 2)
        
        current_space = self.get_used_storage()
        if current_space + self.problem.weights[action] > self.budget:
            return self._step_threshold_exceeded()

        if creating:
            self._state[action] = 1
        else:
            self._state[action] = 0

        reward = self.get_benefit()
        terminated = self.get_used_storage() == self.budget
        truncated = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    
    def _step_threshold_exceeded(self):
        '''
        Space budget exceeded. Do not reward; terminate episode.
        '''
        observation = self._get_obs()
        info = self._get_info()
        reward = self.get_benefit()
        terminated = True
        truncated = False

        return observation, reward, terminated, truncated, info
    
    def get_benefit(self):
        return sum(self.problem.benefits[self._state == 1])

    def get_used_storage(self):
        return sum(self.problem.weights[self._state == 1])
    
    def _mask_action(self, action):
        pass
