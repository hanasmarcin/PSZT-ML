import numpy as np

class ModifiedEvolutionaryAlgorithm:

    def __init__(self, population, evaluation_function, CEC_function_number, lmbd, iter_count):
        """
        Constructor for EvolutionaryAlgorithm
        :param population: array sized mi*2*2d, where mi is a population size and d is a dimension of population
        individuals, each row is one element of population, first d rows are values of each individual and second d rows
        are coefficients for normal distribution for each individual's value
        :param evaluation_function: function for evaluating every individual, whether it should be taken to next
        population
        :param lmbd: Size of temporary population, which will be reproduced
        """
        assert population.shape[1] == 2
        self.P = population
        self.J = evaluation_function
        self.nCEC = CEC_function_number
        self.lmbd = int(lmbd)
        self.iter_count = int(iter_count)
        self.mi = int(self.P.shape[0]*2)
        self.d = int(self.P.shape[2]/2)
        self.tau = 1/np.sqrt(2*self.d)
        self.tau_prim = 1/np.sqrt(2*np.sqrt(self.d))

    def iteration(self):
        T = self.generate_T()
        R = self.reproduce(T)
        self.P = self.choose_new_population(R)

    def generate_T(self):
        """
        Function generates temporary population, which will be reproduced
        :return: temporary population
        """
        T = np.empty([self.lmbd, 2, self.d * 2])
        # loop for sampling (with replacement) lambda pairs to reproduce
        for i in range(self.lmbd):
            random_id = np.random.randint(low=0, high=int(self.mi/2-1))
            T[i, :, :] = self.P[random_id, :, :]

        return T

    def reproduce(self, T):
        """
        Function creates new individuals from T by crossover and mutation
        :return:
        """

        R = np.empty([self.lmbd, self.d * 2])

        for i in range(0, self.lmbd):
            x = self.crossover(T[i, 0], T[i, 1])
            R[i, :] = self.mutate(x)

        return R

    def pair_children(self, R):
        R_paired = np.empty([int(self.lmbd/2), 2, self.d * 2])

        random_ids = np.random.randint(low=0, high=R.shape[0] - 1, size=R.shape[0])
        R = R[random_ids]
        for i in range(0, self.lmbd, 2):
            R_paired[int(i/2), 0] = R[i]
            R_paired[int(i/2), 1] = R[i+1]
        return R_paired

    def choose_new_population(self, R):
        R_paired = self.pair_children(R)
        return self.choose_mi_best(R_paired)

    def choose_mi_best(self, R_paired):
        population = np.vstack([self.P, R_paired])
        eval_values = np.empty(population.shape[0])
        i = 0
        for individual in population:
            eval_values[i] = (self.J(individual[0:self.d, 0], self.nCEC) + self.J(individual[0:self.d, 1], self.nCEC)) / 2
            i = i+1

        sorted_population = population[np.argsort(eval_values)]
        #print(population[np.argsort(eval_values)])
        return sorted_population[-int(self.mi/2):]
        
    @staticmethod
    def crossover(f, m):
        """
        Function makes new individual by crossover on its parents f and m
        :param f: first parent, matrix sized 1*2d, first d rows are values of each individual and second d rows are
        coefficients for normal distribution for each individual's value
        :param m: second parent, matrix sized 1*2d, first d rows are values of each individual and second d rows are
        coefficients for normal distribution for each individual's value
        :return: new individual
        """
        x = (f + m) / 2
        return x

    def mutate(self, x):
        ksi = np.random.normal(0, 1)
        mutated_x = np.zeros(len(x))

        for i in range(self.d):
            ksi_i = np.random.normal(0, 1)
            mutated_x[self.d + i] = x[self.d + i] * np.exp(self.tau * ksi + self.tau_prim * ksi_i)

        for i in range(self.d):
            v_i = np.random.normal(0, 1)
            mutated_x[i] = x[i] + mutated_x[self.d + i] * v_i

        return mutated_x

    def run(self):
        for i in range(self.iter_count):
            self.iteration()
    
        return self.P[-1:]