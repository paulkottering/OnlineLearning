import numpy as np
from numpy import random as rand
from utils.updates import opt_pes_recalc, opt_pes_recalc_make
import matplotlib.pyplot as plt
import itertools

class optimistic_solver():

    def __init__(self,game,c,alpha):

        self.n = game.n
        self.k = game.k
        self.c = c

        shape = [self.n]*self.k
        self.alpha = alpha
        self.matrices = opt_pes_recalc_make(self.n, self.k)

        self.possible = np.zeros(shape)
        self.active = np.ones(shape)

        self.number_samples = 0
        self.diff = []

        self.OptUs = [np.ones([self.n]*self.k) * game.MaxU for i in range(self.k)]
        self.PesUs = [np.ones([self.n]*self.k) * game.MinU for i in range(self.k)]

    def update_us(self,game):

        shape = [self.n] * self.k
        tuples = list(itertools.product(*[range(dim) for dim in shape]))

        for p in range(self.k):
            for tuple_ in tuples:

                opt_mean = game.sum[p][tuple_] / game.number[tuple_] if game.number[tuple_] > 0 else game.MaxU
                pes_mean = game.sum[p][tuple_] / game.number[tuple_] if game.number[tuple_] > 0 else game.MinU

                number = 1 if game.number[tuple_] == 0 else game.number[tuple_]

                a = np.log(game.t)
                b = self.c

                self.OptUs[p][tuple_] = opt_mean + self.c * np.sqrt(np.log(game.t) / number)
                self.PesUs[p][tuple_] = pes_mean - self.c * np.sqrt(np.log(game.t) / number)

    def update_potential(self):
        self.OptPhi, self.PesPhi, diff = opt_pes_recalc(self.matrices, self.OptUs, self.PesUs, self.n, self.k)

    def next_sample_prob(self,game):

        self.update_us(game)
        self.update_potential()


        max_tuple = np.unravel_index(np.argmax(self.OptPhi), self.OptPhi.shape)
        phi = np.zeros([self.n] * self.k)
        phi[max_tuple] = 1

        a = np.random.rand(1)
        print(a)
        if a < self.alpha:
            phi_2 = np.copy(self.OptPhi)
            phi_2 = phi_2 - np.min(self.OptPhi)
            phi_2 /= np.sum(phi_2)
            phi = np.copy(phi_2)

        return phi

    # def OptPesUUpdate(self,p,sample_tuple):
    #
    #     slices = [sample_tuple[i] if i != p else slice(None) for i in range(self.k)]
    #
    #     upper = self.OptUs[p][sample_tuple] + self.PhiMax - self.PhiMin
    #     lower = self.PesUs[p][sample_tuple] + self.PhiMin - self.PhiMax
    #
    #     self.OptUs[p][tuple(slices)] = np.minimum(self.OptUs[p][tuple(slices)],upper)
    #     self.PesUs[p][tuple(slices)] = np.maximum(self.PesUs[p][tuple(slices)],lower)

    # def check_bounds(self):
    #
    #     if np.any(self.OptPhi < self.UnknownGame):
    #         print("OptPhi")
    #
    #     if np.any(self.PesPhi > self.UnknownGame):
    #         print("PesPhi")
    #
    #     for p in range(self.k):
    #         if np.any(self.OptUs[p] < self.UnknownUs[p]):
    #             print(p,"Opt")
    #         if np.any(self.PesUs[p] > self.UnknownUs[p]):
    #             print(p,"Pes")
    #
    # def check_end(self):
    #     if np.max(self.OptPhi) != np.max(self.UnknownGame):
    #         print("Opt Max != Potential Max")

class nash_ucb():
    def __init__(self,game,iterations):

        self.number_agents = game.number_agents
        self.number_facilities = game.number_facilities
        delta = 0.01
        self.const = iterations/delta
        self.facility_counter = np.zeros(self.number_facilities)
        self.iota = 2*np.log(4*(self.number_agents + 1)*self.const)
        self.d = self.number_facilities*self.number_agents
        self.theta_k = np.zeros(self.d)
        self.V_k = np.eye(self.d)
        self.ar_sum = np.zeros(self.d)
        self.epsilon = 0.01

        num_range = np.arange(self.number_facilities)
        actions = []
        # Iterate through all possible lengths of combinations (including empty set)
        for combination_length in range(len(num_range) + 1):
            actions.extend(itertools.combinations(num_range, combination_length))

        self.shape = [len(actions)]*self.number_agents
        self.action_space = actions
        self.full_action_vector,self.n_f_s = self.make_full_action_vector()
        self.t = 0

    def a_i_function(self, i, joint_action,joint_action_index=-1):

        vector = np.zeros(self.d)

        if joint_action_index == -1:
            n_f = self.calculate_n_f(joint_action)
        else:
            n_f = self.n_f_s[joint_action_index]

        for facility in joint_action[i]:
            n = n_f[facility]
            ind = int(n + self.number_agents*(facility-1))
            vector[ind] = 1

        return vector

    def generate_tuples(self):
        tuples = list(itertools.product(*[range(dim) for dim in self.shape]))

        return tuples

    def calculate_n_f(self,joint_action):
        n_f = np.zeros(self.number_facilities)

        for action in joint_action:
            for f in action:
                n_f[f] += 1

        return n_f

    def make_full_action_vector(self):
        full_action_vector = []
        full_n_f_vector = []
        tuples = self.generate_tuples()
        for tuple in tuples:
            joint_action = []
            for p in range(self.number_agents):
                joint_action.append(self.action_space[tuple[p]])
            full_action_vector.append(joint_action)
            full_n_f_vector.append(self.calculate_n_f(joint_action))
        return full_action_vector,full_n_f_vector

    def update_vk(self, action_chosen):
        sum_matrices = 0

        for agent in range(self.number_agents):
            a_i = self.a_i_function(agent, action_chosen)
            matrix = np.inner(a_i, np.transpose(a_i))
            sum_matrices += matrix

        return self.V_k + sum_matrices

    def update_theta_k(self, action_chosen, rewards):
        sum_ar = 0
        for agent in range(self.number_agents):
            a_i = self.a_i_function(agent, action_chosen)
            sum_ar += a_i * rewards[agent]

        self.ar_sum += sum_ar

        self.v_k_inverse = np.linalg.inv(self.V_k)

        return np.matmul(self.v_k_inverse, self.ar_sum)

    def update_vectors(self,rewards,actions_chosen):

        #Update theta_k and V_k
        self.V_k = self.update_vk(actions_chosen)
        self.theta_k = self.update_theta_k(actions_chosen,rewards)
        self.sqrt_beta = np.sqrt(self.d) + np.sqrt(self.d *np.log(1 + self.number_facilities*self.number_agents*self.t/self.d) + self.iota)


    def create_potential_game(self, rewards, actions_chosen):

        self.update_vectors(rewards, actions_chosen)

        reward_matrices = []

        bonus_vector = np.zeros(len(self.full_action_vector))
        for k,joint_action in enumerate(self.full_action_vector):
            norm = 0
            for p in range(self.number_agents):
                a_i = self.a_i_function(p, joint_action,k)
                p_norm = np.matmul(np.transpose(a_i),np.matmul(self.v_k_inverse,a_i))
                norm = np.maximum(norm, p_norm)
            bonus_vector[k] = norm * self.sqrt_beta
        bonus_matrix = np.reshape(bonus_vector,self.shape)

        reward_vector = np.zeros(len(self.full_action_vector))
        for p in range(self.number_agents):
            for k,joint_action in enumerate(self.full_action_vector):
                a_i = self.a_i_function(p, joint_action,k)
                reward = np.dot(a_i,self.theta_k)
                reward_vector[k] = reward
            reward_matrix = np.reshape(reward_vector,self.shape)
            reward_matrices.append(reward_matrix + bonus_matrix)

        return reward_matrices


    def solve_potential_game(self,reward_matrices):
        policy_tuple = np.zeros(self.number_agents)

        K = np.ceil(self.number_agents*np.max(reward_matrices)/self.epsilon)

        for k in range(int(K)):
            deltas = np.zeros(self.number_agents)
            indices = np.zeros(self.number_agents)
            for p in range(self.number_agents):

                cut_slice = [policy_tuple[i] if i != p else slice(None) for i in range(self.number_agents)]

                indices[p] = np.argmax(reward_matrices[p][cut_slice] - reward_matrices[p][policy_tuple])
                deltas[p] = np.max(reward_matrices[p][cut_slice] - reward_matrices[p][policy_tuple])

            if np.max(deltas) <= self.epsilon:
                phi = np.zeros([len(self.action_space)] * self.number_agents)
                phi[policy_tuple] = 1

                return phi

            j = np.argmax(deltas)
            policy_tuple[j] = indices[j]

        phi = np.zeros([len(self.action_space)] * self.number_agents)
        phi[policy_tuple] = 1

        return phi

    def next_sample_prob(self, Game):
        self.t = Game.t
        previous_action = Game.actions_chosen[-1]
        previous_reward = Game.agent_rewards[-1]

        reward_matrices = self.create_potential_game(previous_reward, previous_action)

        return self.solve_potential_game(reward_matrices)
