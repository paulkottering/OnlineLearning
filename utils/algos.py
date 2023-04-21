import numpy as np
from numpy import random as rand
from utils.updates import opt_pes_recalc
import itertools

class nash_ucb():
    """
    Nash UCB class for learning in congestion games with bandit feedback.
    """
    def __init__(self,game,c,iterations):
        """
        Initialize the nash_ucb class.

        :param game: The game object representing the environment.
        :param c: The exploration-exploitation trade-off constant.
        :param iterations: The number of iterations to run the algorithm.
        """

        self.number_agents = game.k
        self.number_facilities = game.number_facilities
        delta = c
        self.const = iterations/delta
        self.facility_counter = np.zeros(self.number_facilities)
        self.iota = 2*np.log(4*(self.number_agents + 1)*self.const)
        self.d = self.number_facilities*self.number_agents
        self.theta_k = np.zeros(self.d)
        self.V_k = np.eye(self.d)
        self.ar_sum = np.zeros(self.d)
        self.epsilon = 0.001

        num_range = np.arange(self.number_facilities)
        actions = []
        # Iterate through all possible lengths of combinations (including empty set)
        for combination_length in range(len(num_range) + 1):
            actions.extend(itertools.combinations(num_range, combination_length))

        self.shape = [len(actions)]*self.number_agents
        self.action_space = actions
        self.t = 0

    def a_i_function(self, i: int, ja_tuple: tuple) -> list:

        vector = np.zeros(self.d)

        n_f = self.calculate_n_f(ja_tuple)
        facilities = self.action_space[ja_tuple[i]]
        for facility in facilities:
            n = n_f[facility]
            ind = int(n + self.number_agents*(facility-1))
            vector[ind] = 1

        return vector

    def generate_tuples(self) -> list:
        tuples = list(itertools.product(*[range(dim) for dim in self.shape]))
        return tuples

    def calculate_n_f(self, ja_tuple: tuple) -> list:

        n_f = np.zeros(self.number_facilities)
        for agent_action in list(ja_tuple):
            facilities = list(self.action_space[agent_action])
            for i in facilities:
                n_f[i] += 1
        return n_f

    def update_vk(self, action_chosen):

        sum_matrices = 0

        for agent in range(self.number_agents):
            a_i = self.a_i_function(agent, action_chosen)
            matrix = np.outer(a_i, a_i)
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
        self.sqrt_beta = np.sqrt(self.d) + np.sqrt(self.d * np.log(1 + self.number_facilities*self.number_agents*self.t/self.d) + self.iota)


    def create_potential_game(self, rewards, actions_chosen):

        self.update_vectors(rewards, actions_chosen)

        reward_matrices = []

        tuples = self.generate_tuples()
        bonus_matrix = np.zeros(self.shape)

        for tuple in tuples:
            norm = -np.inf
            for p in range(self.number_agents):
                a_i = self.a_i_function(p, tuple)
                p_norm = np.matmul(np.transpose(a_i),np.matmul(self.v_k_inverse, a_i))
                norm = np.maximum(norm, p_norm)
            bonus_matrix[tuple] = norm * self.sqrt_beta

        for p in range(self.number_agents):

            reward_matrix = np.zeros(self.shape)
            for tuple in tuples:
                a_i = self.a_i_function(p, tuple)
                reward = np.dot(a_i,self.theta_k)
                reward_matrix[tuple] = reward

            reward_matrices.append(reward_matrix + bonus_matrix)

        return reward_matrices


    def solve_potential_game(self, reward_matrices: object) -> object:
        policy_tuple = tuple([int(np.random.randint(0,len(self.action_space))) for i in range(self.number_agents)])

        K = np.ceil(self.number_agents*np.max(reward_matrices)/self.epsilon)

        for k in range(int(K)):
            deltas = np.zeros(self.number_agents)
            indices = np.zeros(self.number_agents)
            for p in range(self.number_agents):
                policy_reward = reward_matrices[p][policy_tuple]
                slices = [slice(None) if j == p else int(policy_tuple[j]) for j in range(len(policy_tuple))]

                indices[p] = np.argmax(reward_matrices[p][tuple(slices)] - policy_reward)
                deltas[p] = np.max(reward_matrices[p][tuple(slices)] - policy_reward)

            if np.max(deltas) <= self.epsilon:
                phi = np.zeros([len(self.action_space)] * self.number_agents)
                phi[policy_tuple] = 1

                return phi

            j = np.argmax(deltas)
            temp_list = list(policy_tuple)
            temp_list[j] = int(indices[j])
            policy_tuple = tuple(temp_list)

        phi = np.zeros([len(self.action_space)] * self.number_agents)
        phi[policy_tuple] = 1

        return phi

    def next_sample_prob(self, Game):
        """
        Calculate the probabilities for the next sample.

        :param Game: The game object representing the environment.
        :return: The probability matrix for the next sample.
        """
        self.t = Game.t
        previous_action = Game.actions_chosen[-1]
        previous_reward = Game.agent_rewards[-1]

        reward_matrices = self.create_potential_game(previous_reward, previous_action)

        return self.solve_potential_game(reward_matrices)

class exponential_weights_annealing():
    """
    Exponential Weights Annealing class for online learning.
    """
    def __init__(self,game,beta,alpha):
        """
        Initialize the exponential_weights_annealing class.

        :param game: The game object representing the environment.
        :param beta: The learning rate parameter.
        """

        self.n = game.n
        self.k = game.k
        self.beta = beta
        self.alpha = alpha

        self.t = 1
        self.number_samples = 0
        self.diff = []
        self.Ys = [np.ones(self.n)/self.n for k in range(self.k)]
        self.Xs = [np.ones(self.n)/self.n for k in range(self.k)]
        self.step_size = 0
        self.epsilon = 0

    def next_sample_prob(self, game):
        """
        Calculate the probabilities for the next sample.

        :param game: The game object representing the environment.
        :return: The probability matrix for the next sample.
        """
        self.t += 1

        if self.t != 2:
            self.update_ys(game)

        self.epsilon = (1 / self.t) ** self.alpha

        for i in range(self.k):
            self.Xs[i] = self.epsilon*(np.ones(self.n)/self.n) + (1-self.epsilon)*self.logit_choice_map(self.Ys[i])

        tuples = list(itertools.product(*[range(dim) for dim in game.shape]))
        phi = np.zeros(game.shape)
        for tuple_ in tuples:
            vec = [self.Xs[p][tuple_[p]] for p in range(self.k)]
            phi[tuple_] = np.prod(vec)
        return phi
    def update_ys(self,game):

        self.step_size = (1/self.t)**self.beta

        for i in range(self.k):
            alpha_i = game.actions_chosen[-1][i]

            self.Ys[i][alpha_i] += self.step_size*game.rewards[-1][i]/self.Xs[i][alpha_i]

    def logit_choice_map(self,vector):

        sum_exp = np.sum(np.exp(vector))

        return np.exp(vector)/sum_exp

class nash_ca():
    """
    Nash co-ordinate ascent class for learning in games.
    """

    def __init__(self,game,c,thresh):
        """
        Initialize the nash_ca class.

        :param game: The game object representing the environment.
        :param c: The exploration-exploitation trade-off constant.
        :param thresh: The threshold for looping through the current policy.
        """

        self.n = game.n
        self.k = game.k
        self.c = c

        shape = [self.n]*self.k

        self.current_player = 0
        self.current_policy_counter = 0
        self.current_policy = tuple([int(0) for i in range(self.k)])
        self.subroutine = False
        self.subroutine_episode_counter = 0
        self.thresh = thresh
        self.deltas = np.zeros(self.k)
        self.a_hat = np.zeros(self.k)
        self.temp_episode_counter = 0
        self.temp_policy = tuple(np.zeros(self.k))
        self.means = [np.zeros(shape) for i in range(self.k)]
        self.t = 0
    def next_sample_prob(self,game):
        """
        Calculate the probabilities for the next sample.

        :param game: The game object representing the environment.
        :return: The probability matrix for the next sample.
        """

        if self.t > 0:
            self.update_means(game)

        self.t += 1

        #Loop through current policy
        while self.current_policy_counter < self.thresh:
            self.current_policy_counter += 1
            phi = np.zeros([self.n] * self.k)
            phi[self.current_policy] = 1
            return phi

        if self.subroutine_episode_counter >= self.thresh:

            if self.temp_episode_counter == 0:
                #End of agent subroutine, start evaluation of temp policy
                self.a_hat[self.current_player] = self.get_a_hat()
                self.temp_policy = list(self.current_policy)
                self.temp_policy[self.current_player] = int(self.a_hat[self.current_player])
                self.temp_policy = tuple(self.temp_policy)

                self.temp_episode_counter += 1
                phi = np.zeros([self.n] * self.k)
                phi[self.temp_policy] = 1
                return phi

            if self.temp_episode_counter < self.thresh:
                # Continue evaluation of temp policy
                self.temp_episode_counter += 1
                phi = np.zeros([self.n] * self.k)
                phi[self.temp_policy] = 1
                return phi

            # End of temp policy evaluation, return to subroutines
            self.deltas[self.current_player] = self.means[self.current_player][self.temp_policy] - self.means[self.current_player][self.current_policy]
            self.current_player += 1
            self.subroutine_episode_counter = 0
            self.temp_episode_counter = 0

            if self.current_player >= self.k:
                #if all agents subroutines are done, make new policy
                j = np.argmax(self.deltas)
                new_policy = list(self.current_policy)
                new_policy[j] = int(self.a_hat[j])
                self.current_policy = tuple(new_policy)

                #end of agent for loop
                self.current_policy_counter = 0
                self.subroutine_episode_counter = 0
                self.current_player = 0
                self.temp_episode_counter = 0
                self.deltas = np.zeros(self.k)
                self.a_hat = np.zeros(self.k)

                phi = np.zeros([self.n] * self.k)
                phi[self.current_policy] = 1
                return phi

        #Continue agent subroutine
        self.subroutine_episode_counter += 1
        return self.ucb_sub_routine(game)

    def get_a_hat(self):
        current_policy = self.current_policy
        current_agent = self.current_player
        slices = [slice(None) if j == current_agent else int(current_policy[j]) for j in range(len(current_policy))]

        return np.argmax(self.means[current_agent][slices])

    def ucb_sub_routine(self,game):
        current_policy = self.current_policy
        current_agent = self.current_player

        slices = [slice(None) if j == current_agent else int(current_policy[j]) for j in range(len(current_policy))]
        mean_vector = self.means[current_agent][slices]
        number_vector = game.number[slices]
        number_vector_clean = [number_vector[i] if number_vector[i] != 0 else 1 for i in range(self.n)]

        t = np.sum(number_vector)
        ucb = mean_vector + self.c*np.sqrt(np.log(t)/number_vector_clean)

        current_policy_list = list(current_policy)
        current_policy_list[current_agent] = np.argmax(ucb)

        phi = np.zeros([self.n] * self.k)
        phi[tuple(current_policy_list)] = 1

        return phi

    def update_means(self,game):

        last_action = game.actions_chosen[-1]

        for p in range(self.k):
            self.means[p][last_action] = game.sum[p][last_action]/game.number[last_action]

class optimistic_solver():
    """
    custom optimistic solver algorithm class.
    """
    def __init__(self, game, c, alpha, matrices):
        """
        Initialize the optimistic_solver class.

        :param game: The game object representing the game to solve.
        :param c: The exploration-exploitation trade-off constant.
        :param alpha: The random probability threshold for exploration.
        """

        self.n = game.n
        self.k = game.k
        self.c = c

        shape = [self.n] * self.k
        self.alpha = alpha
        # Calculate and store the optimization and pessimistic matrices
        self.matrices = matrices

        # Initialize number of samples and differences
        self.number_samples = 0

        # Initialize optimistic and pessimistic utility matrices
        self.OptUs = [np.ones([self.n] * self.k) * game.MaxU for i in range(self.k)]
        self.PesUs = [np.ones([self.n] * self.k) * game.MinU for i in range(self.k)]

    def update_us(self, game):
        """
        Update the optimistic and pessimistic utility matrices.

        :param game: The game object representing the game to solve.
        """
        shape = [self.n] * self.k
        tuples = list(itertools.product(*[range(dim) for dim in shape]))

        for p in range(self.k):
            for tuple_ in tuples:
                # Calculate the optimistic and pessimistic means
                opt_mean = game.sum[p][tuple_] / game.number[tuple_] if game.number[tuple_] > 0 else game.MaxU
                pes_mean = game.sum[p][tuple_] / game.number[tuple_] if game.number[tuple_] > 0 else game.MinU

                number = 1 if game.number[tuple_] == 0 else game.number[tuple_]

                # Update the optimistic and pessimistic utility matrices with UCB and LCB respectively
                self.OptUs[p][tuple_] = opt_mean + self.c * np.sqrt(np.log(game.t) / number)
                self.PesUs[p][tuple_] = pes_mean - self.c * np.sqrt(np.log(game.t) / number)

    def update_potential(self):
        """
        Update the optimistic and pessimistic potential matrices.
        """
        self.OptPhi, self.PesPhi = opt_pes_recalc(self.matrices, self.OptUs, self.PesUs, self.n, self.k)

    def next_sample_prob(self, game):
        """
        Calculate the probabilities for the next sample.

        :param game: The game object representing the environment.
        :return: The probability matrix for the next sample.
        """

        # Update utility matrices and potentials
        self.update_us(game)
        self.update_potential()

        # Find the tuple with the maximum optimistic potential
        max_tuple = np.unravel_index(np.argmax(self.OptPhi), self.OptPhi.shape)
        phi = np.zeros([self.n] * self.k)
        phi[max_tuple] = 1

        # Explore other sample with random probability
        a = np.random.rand(1)
        alpha = 0.1*np.exp(np.log(self.alpha)*game.t/(self.n**self.k))
        if (a < alpha) and (game.t > 1):
            phi_2 = np.copy(self.OptPhi)
            phi_2 = phi_2 - np.min(self.OptPhi)
            phi_2 /= np.sum(phi_2)
            phi = np.copy(phi_2)
        return phi
