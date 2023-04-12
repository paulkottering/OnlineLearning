import numpy as np
import itertools

class potential_game():

    def __init__(self,Potential, unknown_utilitys,nl):

        self.n = np.shape(Potential)[0]
        self.k = len(unknown_utilitys)
        self.nl = nl

        shape = [self.n]*self.k

        self.Potential = Potential
        self.UnknownUs = unknown_utilitys
        self.MaxU = np.max(unknown_utilitys)
        self.MinU = np.min(unknown_utilitys)

        self.check_game()

        self.number = np.zeros(shape)
        self.sum = [np.zeros(shape) for i in range(self.k)]
        self.sum_squared = [np.zeros(shape) for i in range(self.k)]

        self.t = 1

    def sample(self,sample_tuple):

        self.t += 1

        for p in range(self.k):

            u_val = self.UnknownUs[p][sample_tuple] + np.random.normal(0,self.nl)

            self.number[sample_tuple] += 1
            self.sum[p][sample_tuple] += u_val
            self.sum_squared[p][sample_tuple] += u_val**2

    def check_game(self):

        for p in range(self.k):
            for i in range(1, self.n):
                rel_slice = [slice(None)] * p + [i] + [Ellipsis]
                prev_slice = [slice(None)] * p + [i - 1] + [Ellipsis]

                Us = self.UnknownUs[p][tuple(rel_slice)] - self.UnknownUs[p][tuple(prev_slice)]
                Ps = self.Potential[tuple(rel_slice)] - self.Potential[tuple(prev_slice)]
                if not np.allclose(Us,Ps):
                    raise ValueError("Not a valid Potential Game!")


class congestion_game():

    def __init__(self,facility_means,number_agents,nl):

        self.number_agents = number_agents
        self.number_facilities = len(facility_means)
        self.facility_means = facility_means
        self.nl = nl
        self.agent_rewards = []
        self.actions_chosen = []
        self.t = 1

        num_range = np.arange(self.number_facilities)
        actions = []
        # Iterate through all possible lengths of combinations (including empty set)
        for combination_length in range(len(num_range) + 1):
            actions.extend(itertools.combinations(num_range, combination_length))

        self.actions = actions
        self.n = len(actions)
        self.k = number_agents

        shape = [self.n] * self.k

        self.number = np.zeros(shape)
        self.sum = [np.zeros(shape) for i in range(self.k)]
        self.sum_squared = [np.zeros(shape) for i in range(self.k)]

        self.sample([np.random.randint(1, len(facility_means), size=1) for k in range(number_agents)])

    def sample_potential(self,sample_tuple):

        self.t += 1

        actions_chosen = [self.actions[k] for k in sample_tuple]

        facility_rewards = [self.facility_means[k] + np.random.normal(0, self.nl) for k in
                            range(self.number_facilities)]

        rewards = np.zeros(self.number_agents)

        for i, agent_actions in enumerate(actions_chosen):
            for facility in agent_actions:
                rewards[i] += facility_rewards[facility]

        for p in range(self.k):
            self.number[sample_tuple] += 1
            self.sum[p][sample_tuple] += rewards[p]
            self.sum_squared[p][sample_tuple] += rewards[p]**2

    def sample(self,action_chosen):

        self.t += 1

        numbers = self.number_for_each_facility(action_chosen)

        facility_rewards = [self.facility_means[k] + np.random.normal(0,self.nl) for k in range(self.number_facilities)]
        rewards = np.zeros(self.number_agents)

        for i,agent_actions in enumerate(action_chosen):
            for facility in agent_actions:
                num = int(numbers[facility])
                rewards[i] += facility_rewards[facility][num-1]

        self.agent_rewards.append(rewards)
        self.actions_chosen.append(action_chosen)

    def number_for_each_facility(self,action_chosen):
        numbers = np.zeros(self.number_facilities)
        for agent_actions in action_chosen:
            for facility in agent_actions:
                numbers[facility] += 1
        return numbers


