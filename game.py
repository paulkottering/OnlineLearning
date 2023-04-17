import numpy as np
import itertools

class potential_game():

    def __init__(self,Potential, unknown_utilitys,nl):

        self.n = np.shape(Potential)[0]
        self.k = len(unknown_utilitys)
        self.nl = nl

        self.shape = [self.n]*self.k

        self.Potential = Potential
        self.utility_matrices = unknown_utilitys
        self.MaxU = np.max(unknown_utilitys)
        self.MinU = np.min(unknown_utilitys)

        self.check_game()

        self.number = np.zeros(self.shape)
        self.sum = [np.zeros(self.shape) for i in range(self.k)]
        self.sum_squared = [np.zeros(self.shape) for i in range(self.k)]

        self.rewards = []
        self.actions = []

        self.t = 1

    def sample(self,sample_tuple):

        rewards = np.zeros(self.k)

        self.t += 1
        self.number[sample_tuple] += 1

        for p in range(self.k):

            u_val = self.utility_matrices[p][sample_tuple] + np.random.normal(0,self.nl)
            self.sum[p][sample_tuple] += u_val
            self.sum_squared[p][sample_tuple] += u_val**2

            rewards[p] = u_val

        self.rewards.append(rewards)
        self.actions.append(sample_tuple)

    def check_game(self):

        for p in range(self.k):
            for i in range(1, self.n):
                rel_slice = [slice(None)] * p + [i] + [Ellipsis]
                prev_slice = [slice(None)] * p + [i - 1] + [Ellipsis]

                Us = self.utility_matrices[p][tuple(rel_slice)] - self.utility_matrices[p][tuple(prev_slice)]
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

        self.shape = [self.n] * self.k

        self.number = np.zeros(self.shape)
        self.sum = [np.zeros(self.shape) for i in range(self.k)]
        self.sum_squared = [np.zeros(self.shape) for i in range(self.k)]
        self.sample(tuple(np.random.randint(1, len(self.actions), size=self.number_agents)))
        self.Potential, self.utility_matrices = self.potential_utilities_for_regret()
        self.MaxU = self.number_facilities
        self.MinU = 0

        self.check_game()

    def sample(self,action_chosen):

        self.t += 1
        numbers = self.number_for_each_facility(action_chosen)
        facility_rewards = np.clip([self.facility_means[i][int(numbers[i])-1] + np.random.normal(0, self.nl) for i in range(self.number_facilities)], 0, 1)
        rewards = np.zeros(self.number_agents)

        for i,agent_action in enumerate(list(action_chosen)):
            facilities = list(self.actions[agent_action])
            for k in facilities:
                rewards[i] += facility_rewards[k]

        self.agent_rewards.append(rewards)
        self.actions_chosen.append(action_chosen)

        for p in range(self.k):
            self.number[action_chosen] += 1
            self.sum[p][action_chosen] += rewards[p]
            self.sum_squared[p][action_chosen] += rewards[p]**2

    def number_for_each_facility(self,action_chosen):

        numbers = np.zeros(self.number_facilities)

        for agent_action in list(action_chosen):
            facilities = list(self.actions[agent_action])
            for i in facilities:
                numbers[i] += 1
        return numbers

    def potential_utilities_for_regret(self):
        tuples = list(itertools.product(*[range(dim) for dim in self.shape]))
        potential_matrix = np.zeros(self.shape)
        utility_matrices = [np.zeros(self.shape) for i in range(self.number_agents)]

        for tuple in tuples:
            numbers = self.number_for_each_facility(tuple)
            for i, agent in enumerate(range(self.number_agents)):
                facilities_visited = self.actions[tuple[i]]
                utility = 0
                for facility in facilities_visited:
                    utility += self.facility_means[facility][int(numbers[facility])-1]

                utility_matrices[i][tuple] = utility

            potential = 0
            for i,number in enumerate(numbers):
                if number == 0:
                    continue
                else:
                    potential += sum(self.facility_means[i][:int(number)])

            potential_matrix[tuple] = potential

        return potential_matrix, utility_matrices
    def check_game(self):

        for p in range(self.k):
            for i in range(1, self.n):
                rel_slice = [slice(None)] * p + [i] + [Ellipsis]
                prev_slice = [slice(None)] * p + [i - 1] + [Ellipsis]

                Us = self.utility_matrices[p][tuple(rel_slice)] - self.utility_matrices[p][tuple(prev_slice)]
                Ps = self.Potential[tuple(rel_slice)] - self.Potential[tuple(prev_slice)]
                if not np.allclose(Us,Ps):
                    raise ValueError("Not a valid Potential Game!")





