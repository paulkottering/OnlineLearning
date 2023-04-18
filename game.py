import numpy as np
import itertools

class potential_game():
    """
    A class to simulate and get noisy samples rom potential games from a given potential and utility matrices.f

    Attributes:
        n (int): Number of states in the game.
        k (int): Number of players in the game.
        nl (float): Noise level for the utilities.
        shape (list): Shape of the arrays for internal use.
        Potential (numpy.ndarray): The potential matrix of the game.
        utility_matrices (list): A list of utility matrices for each player.
        MaxU (float): The maximum utility value among all players.
        MinU (float): The minimum utility value among all players.
        number (numpy.ndarray): Array for counting the number of times an action has been sampled.
        sum (list): A list of arrays to accumulate the utilities for each player.
        sum_squared (list): A list of arrays to accumulate the squared utilities for each player.
        rewards (list): A list to store the rewards history.
        actions_chosen (list): A list to store the chosen actions history.
        t (int): Time step counter.

    Methods:
        __init__(self, Potential, unknown_utilitys, nl): Initialize the potential game object with necessary parameters.
        sample(self, sample_tuple): Sample a new action and update parameters.
        check_game(self): Check if the given game is a valid potential game.
    """

    # Initialize the potential game object with necessary parameters
    def __init__(self, Potential, unknown_utilitys, nl):
        # Set number of actions for each player
        self.n = np.shape(Potential)[0]
        # Set number of players
        self.k = len(unknown_utilitys)
        # Set noise level
        self.nl = nl

        # Set shape of arrays for internal use
        self.shape = [self.n] * self.k

        # Store input parameters
        self.Potential = Potential
        self.utility_matrices = unknown_utilitys
        self.MaxU = np.max(unknown_utilitys)
        self.MinU = np.min(unknown_utilitys)

        # Validate if the game is a potential game
        self.check_game()

        # Initialize counters and accumulators
        self.number = np.zeros(self.shape)
        self.sum = [np.zeros(self.shape) for i in range(self.k)]
        self.sum_squared = [np.zeros(self.shape) for i in range(self.k)]

        # Initialize history of rewards and actions chosen
        self.rewards = []
        self.actions_chosen = []

        # Initialize time step counter
        self.t = 1

    # Sample a new action and update the game's state
    def sample(self, sample_tuple):
        # Initialize rewards array for the current step
        rewards = np.zeros(self.k)
        # Increment time step counter
        self.t += 1
        # Increment the count of the action being sampled
        self.number[sample_tuple] += 1
        # Update the accumulators and rewards for each player
        for p in range(self.k):
            u_val = self.utility_matrices[p][sample_tuple] + np.random.normal(0, self.nl)
            self.sum[p][sample_tuple] += u_val
            self.sum_squared[p][sample_tuple] += u_val**2
            rewards[p] = u_val

        # Append the rewards and action chosen to the history
        self.rewards.append(rewards)
        self.actions_chosen.append(sample_tuple)

    # Check if the given game is a valid potential game
    def check_game(self):
        # Loop through all players and states
        for p in range(self.k):
            for i in range(1, self.n):
                # Construct slices
                rel_slice = [slice(None)] * p + [i] + [Ellipsis]
                prev_slice = [slice(None)] * p + [i - 1] + [Ellipsis]

                # Calculate differences in utilities and potentials
                Us = self.utility_matrices[p][tuple(rel_slice)] - self.utility_matrices[p][tuple(prev_slice)]
                Ps = self.Potential[tuple(rel_slice)] - self.Potential[tuple(prev_slice)]
                # Check if the differences are equal (within tolerance)
                if not np.allclose(Us, Ps):
                    raise ValueError("Not a valid Potential Game!")


class congestion_game():
    """
    A class to simulate and get noisy samples from a congestion games with a given set of facility means, number of agents, and noise level.

    Attributes:
        number_facilities (int): Number of facilities in the game.
        facility_means (list): A list of utility means for each facility.
        nl (float): Noise level.
        agent_rewards (list): A list to store the rewards history for each agent.
        actions_chosen (list): A list to store the chosen actions history.
        t (int): Time step counter.
        actions (list): A list of all possible actions in the extended game.
        n (int): Number of actions in the extended game.
        k (int): Number of agents in the game.
        shape (list): Shape of the arrays for internal use.
        number (numpy.ndarray): Array for counting the number of times a joint action has been sampled.
        sum (list): A list of arrays to accumulate the sum of utilities for each agent.
        sum_squared (list): A list of arrays to accumulate the sum of squared utilities for each agent.
        Potential (numpy.ndarray): The potential matrix of the equivalent potential game.
        utility_matrices (list): A list of utility matrices for each agent in the extended game.
        MaxU (int): The maximum utility value in the game.
        MinU (int): The minimum utility value in the game.

    Methods:
        __init__(self, facility_means, number_agents, nl): Initialize the congestion game object with necessary parameters.
        sample(self, action_chosen): Sample a new action and update the game's state.
        number_for_each_facility(self, action_chosen): Calculate the number of agents visiting each facility.
        potential_utilities_for_regret(self): Calculate potential and utility matrices
        check_game(self): Check if the given game is a valid potential game.
    """

    # Initialize the congestion game object with necessary parameters
    def __init__(self,facility_means,number_agents,nl):

        # Store input parameters
        self.number_facilities = len(facility_means)
        self.facility_means = facility_means
        self.nl = nl

        # Initialize history of agent rewards and actions chosen
        self.agent_rewards = []
        self.actions_chosen = []

        # Initialize time step counter
        self.t = 1

        # Generate all possible actions for the game
        num_range = np.arange(self.number_facilities)
        actions = []
        for combination_length in range(len(num_range) + 1):
            actions.extend(itertools.combinations(num_range, combination_length))

        # Store the generated actions and their count
        self.actions = actions
        self.n = len(actions)

        # Set number of players (agents)
        self.k = number_agents

        # Set shape of arrays for internal use
        self.shape = [self.n] * self.k

        # Initialize counters and accumulators
        self.number = np.zeros(self.shape)
        self.sum = [np.zeros(self.shape) for i in range(self.k)]
        self.sum_squared = [np.zeros(self.shape) for i in range(self.k)]

        # Initialize the game with random actions for each agent
        self.sample(tuple(np.random.randint(1, len(self.actions), size=self.k)))

        # Calculate the potential and utility matrices for the game
        self.Potential, self.utility_matrices = self.potential_utilities_for_regret()

        # Set the maximum and minimum utility values
        self.MaxU = self.number_facilities
        self.MinU = 0

        # Validate if the game is a potential game
        self.check_game()

    def sample(self,action_chosen):
        # Increment time step counter
        self.t += 1

        # Calculate the number of agents visiting each facility
        numbers = self.number_for_each_facility(action_chosen)

        # Compute facility rewards with noise and clip them between 0 and 1
        facility_rewards = np.clip([self.facility_means[i][int(numbers[i]) - 1] + np.random.normal(0, self.nl) for i in range(self.number_facilities)], 0, 1)

        # Initialize rewards array for the current step
        rewards = np.zeros(self.k)

        # Calculate rewards for each agent based on the facilities they visited
        for i, agent_action in enumerate(list(action_chosen)):
            facilities = list(self.actions[agent_action])
            for k in facilities:
                rewards[i] += facility_rewards[k]

        # Append the rewards and action chosen to the history
        self.agent_rewards.append(rewards)
        self.actions_chosen.append(action_chosen)

        # Update the accumulators and counters for each agent
        for p in range(self.k):
            self.number[action_chosen] += 1
            self.sum[p][action_chosen] += rewards[p]
            self.sum_squared[p][action_chosen] += rewards[p] ** 2

    # Calculate the number of agents visiting each facility
    def number_for_each_facility(self, action_chosen):
        # Initialize an array to store the number of agents visiting each facility
        numbers = np.zeros(self.number_facilities)

        # Loop through each agent's action
        for agent_action in list(action_chosen):
            # Get the list of facilities visited by the agent
            facilities = list(self.actions[agent_action])
            # Increment the count of agents visiting each facility
            for i in facilities:
                numbers[i] += 1
        # Return the array of agent counts for each facility
        return numbers

    # Calculate potential and utility matrices
    def potential_utilities_for_regret(self):
        # Generate all possible action combinations for all agents
        tuples = list(itertools.product(*[range(dim) for dim in self.shape]))
        # Initialize potential and utility matrices
        potential_matrix = np.zeros(self.shape)
        utility_matrices = [np.zeros(self.shape) for i in range(self.k)]

        # Loop through each action combination
        for tuple in tuples:
            # Calculate the number of agents visiting each facility for the given action combination
            numbers = self.number_for_each_facility(tuple)
            # Loop through each agent
            for i, agent in enumerate(range(self.k)):
                # Get the facilities visited by the agent in the action combination
                facilities_visited = self.actions[tuple[i]]
                # Calculate the utility for the agent
                utility = 0
                for facility in facilities_visited:
                    utility += self.facility_means[facility][int(numbers[facility]) - 1]

                # Assign the utility value to the corresponding position in the utility matrix
                utility_matrices[i][tuple] = utility

            # Calculate the potential value for the given action combination
            potential = 0
            for i, number in enumerate(numbers):
                if number == 0:
                    continue
                else:
                    potential += sum(self.facility_means[i][:int(number)])

            # Assign the potential value to the corresponding position in the potential matrix
            potential_matrix[tuple] = potential

        return potential_matrix, utility_matrices

    # Check if the given game is a valid potential game
    def check_game(self):
        # Loop through all agents and actions
        for p in range(self.k):
            for i in range(1, self.n):
                # Construct slices for current and previous actions
                rel_slice = [slice(None)] * p + [i] + [Ellipsis]
                prev_slice = [slice(None)] * p + [i - 1] + [Ellipsis]

                # Calculate differences in utilities and potentials
                Us = self.utility_matrices[p][tuple(rel_slice)] - self.utility_matrices[p][tuple(prev_slice)]
                Ps = self.Potential[tuple(rel_slice)] - self.Potential[tuple(prev_slice)]

                # Check if the differences are equal (within tolerance)
                if not np.allclose(Us, Ps):
                    raise ValueError("Not a valid Potential Game!")





