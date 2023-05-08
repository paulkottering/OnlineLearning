import numpy as np
from numpy import random as rand
from utils.updates import opt_pes_recalc
import itertools

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

        self.k = game.k
        self.c = c


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
        self.shape = np.shape(game.number)
        self.means = [np.zeros(self.shape) for i in range(self.k)]
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
            return self.returner(self.current_policy)

        if self.subroutine_episode_counter >= self.thresh:

            if self.temp_episode_counter == 0:
                #End of agent subroutine, start evaluation of temp policy
                self.a_hat[self.current_player] = self.get_a_hat()
                self.temp_policy = list(self.current_policy)
                self.temp_policy[self.current_player] = int(self.a_hat[self.current_player])
                self.temp_policy = tuple(self.temp_policy)

                self.temp_episode_counter += 1
                return self.returner(self.temp_policy)

            if self.temp_episode_counter < self.thresh:
                # Continue evaluation of temp policy
                self.temp_episode_counter += 1
                return self.returner(self.temp_policy)

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

                return self.returner(self.current_policy)

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
        number_vector_clean = [number_vector[i] if number_vector[i] != 0 else 1 for i in range(game.number.shape[current_agent])]
        t = np.sum(number_vector)
        ucb = mean_vector + self.c*np.sqrt(np.log(t)/number_vector_clean)

        current_policy_list = list(current_policy)
        current_policy_list[current_agent] = np.argmax(ucb)

        return self.returner(tuple(current_policy_list))

    def update_means(self,game):

        last_action = game.actions_chosen[-1]

        for p in range(self.k):
            self.means[p][last_action] = game.sum[p][last_action]/game.number[last_action]

    def returner(self,sample_tuple):
        phi = np.zeros(self.shape)
        phi[sample_tuple] = 1
        return phi

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

        self.k = game.k
        self.c = c

        self.alpha = alpha
        # Calculate and store the optimization and pessimistic matrices
        self.matrices = matrices

        # Initialize number of samples and differences
        self.number_samples = 0
        self.shape = game.shape

        # Initialize optimistic and pessimistic utility matrices
        self.OptUs = [np.ones(self.shape) * game.MaxU for i in range(self.k)]
        self.PesUs = [np.ones(self.shape) * game.MinU for i in range(self.k)]

    def update_us(self, game):
        """
        Update the optimistic and pessimistic utility matrices.

        :param game: The game object representing the game to solve.
        """
        tuples = list(itertools.product(*[range(dim) for dim in self.shape]))

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
        self.OptPhi, self.PesPhi = opt_pes_recalc(self.matrices, self.OptUs, self.PesUs, self.shape)

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
        phi = np.zeros(self.shape)
        phi[max_tuple] = 1

        # Explore other sample with random probability
        a = np.random.rand(1)
        alpha = 0.1*np.exp(np.log(self.alpha)*game.t/(np.prod(self.shape)))
        if (a < alpha) and (game.t > 1):
            phi_2 = np.copy(self.OptPhi)
            phi_2 = phi_2 - np.min(self.OptPhi)
            phi_2 /= np.sum(phi_2)
            phi = np.copy(phi_2)
        return phi

class nash_ucb():
    """
    Nash UCB class for learning in congestion games with bandit feedback.
    """

    def __init__(self, game, c, iterations):
        """
        Initialize the nash_ucb class.

        :param game: The game object representing the environment.
        :param c: The exploration-exploitation trade-off constant.
        :param iterations: The number of iterations to run the algorithm.
        """

        self.number_agents = game.k
        self.number_facilities = game.number_facilities
        delta = c
        self.const = iterations / delta
        self.facility_counter = np.zeros(self.number_facilities)
        self.iota = 2 * np.log(4 * (self.number_agents + 1) * self.const)
        self.d = self.number_facilities * self.number_agents
        self.theta_k = np.zeros(self.d)
        self.V_k = np.eye(self.d)
        self.ar_sum = np.zeros(self.d)
        self.epsilon = 0.001
        self.action_spaces = game.actions
        self.t = 0

    def a_i_function(self, i: int, ja_tuple: tuple) -> list:
        vector = np.zeros(self.d)
        n_f = self.number_for_each_facility(ja_tuple)
        facilities = self.action_spaces[i][ja_tuple[i]]
        for facility in facilities:
            n = n_f[facility]
            ind = int(n + self.number_agents * (facility - 1))
            vector[ind] = 1

        return vector
    def number_for_each_facility(self, action_chosen):
        # Initialize an array to store the number of agents visiting each facility
        numbers = np.zeros(self.number_facilities)

        # Loop through each agent's action
        for i, agent_action in enumerate(list(action_chosen)):
            # Get the list of facilities visited by the agent
            facilities = list(self.action_spaces[i][agent_action])
            # Increment the count of agents visiting each facility
            for k in facilities:
                numbers[k] += 1
        # Return the array of agent counts for each facility
        return numbers

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

    def update_vectors(self, rewards, actions_chosen):

        # Update theta_k and V_k
        self.V_k = self.update_vk(actions_chosen)
        self.theta_k = self.update_theta_k(actions_chosen, rewards)
        a = self.d * np.log(1 + self.number_facilities * self.number_agents * self.t / self.d) + self.iota
        self.sqrt_beta = np.sqrt(self.d) + np.sqrt(
        self.d * np.log(1 + self.number_facilities * self.number_agents * self.t / self.d) + self.iota)

    def solve_potential_game(self):

        policy_tuple = tuple([np.random.randint(1, len(self.action_spaces[i])) for i in range(self.number_agents)])

        K = np.ceil(self.number_agents * self.number_agents*self.number_facilities / self.epsilon)

        for k in range(int(K)):
            deltas = np.zeros(self.number_agents)
            indices = np.zeros(self.number_agents)
            for p in range(self.number_agents):
                policy_reward = self.reward_calc(policy_tuple,p)
                indexs = [self.tuple_changer(policy_tuple, k, p) for k in range(len(self.action_spaces[p]))]
                vector = [self.reward_calc(tuple(tuple_), p) for tuple_ in indexs]

                indices[p] = np.argmax(vector - policy_reward)
                deltas[p] = np.max(vector - policy_reward)

            if np.max(deltas) <= self.epsilon:
                return policy_tuple

            j = np.argmax(deltas)
            temp_list = list(policy_tuple)
            temp_list[j] = int(indices[j])
            policy_tuple = tuple(temp_list)

        return policy_tuple

    def next_sample_prob(self, Game):
        """
        Calculate the probabilities for the next sample.

        :param Game: The game object representing the environment.
        :return: The probability matrix for the next sample.
        """
        self.t = Game.t
        previous_action = Game.actions_chosen[-1]
        previous_reward = Game.agent_rewards[-1]

        self.update_vectors(previous_reward, previous_action)

        return self.solve_potential_game()
    def tuple_changer(self,policy_tuple,k,p):
        tuple_list = list(policy_tuple)
        tuple_list[p] = k
        return tuple(tuple_list)

    def reward_calc(self,tuple_, k):
        norm = -np.inf

        for p in range(self.number_agents):
            a_i = self.a_i_function(p, tuple_)
            p_norm = np.matmul(np.transpose(a_i),np.matmul(self.v_k_inverse, a_i))
            norm = np.maximum(norm, p_norm)

        bonus = norm * self.sqrt_beta
        a_i = self.a_i_function(k, tuple_)
        reward = np.dot(a_i,self.theta_k)

        return reward + bonus