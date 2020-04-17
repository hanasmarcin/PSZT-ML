import numpy as np


class EvolutionaryAlgorithm:

    def __init__(self, population, evaluation_function, lmbd):
        """
        Constructor for EvolutionaryAlgorithm
        :param population: array sized mi*2d, where mi is a population size and d is a dimension of population
        individuals, each row is one element of population, first d rows are values of each individual and second d rows
        are coefficients for normal distribution for each individual's value
        :param evaluation_function: function for evaluating every individual, whether it should be taken to next
        population
        :param lmbd: Size of temporary population, which will be reproduced
        """
        self.P = population
        self.J = evaluation_function
        self.lmbd = int(lmbd)
        self.mi = int(self.P.shape[0])
        self.d = int(self.P.shape[1]/2)
        self.tau = 1/np.sqrt(2*self.d)
        self.tau_prim = 1/np.sqrt(2*np.sqrt(self.d))

    def generate_T(self):
        """
        Function generates temporary population, which will be reproduced
        :return: temporary population
        """
        T = np.empty([self.lmbd, self.d * 2])
        # loop for sampling with replacement
        for i in range(self.lmbd):
            random_id = np.random.randint(low=0, high=self.mi-1)
            T[i, :] = self.P[random_id, :]

        return T

    def reproduce(self, T):
        """
        Function creates new individuals from T by crossover and mutation
        :return:
        """

        R = np.empty([self.lmbd, self.d * 2])

        for i in range(0, self.lmbd):
            x = self.crossover(T[i - 1], T[i])
            R[i, :] = self.mutate(x)

        return R

    def choose_new_population(self, R):
        self.choose_mi_best(R)

    def choose_mi_best(self, R):
        population = np.empty([self.P.shape[0] + R.shape[0], 2*self.d + 1])
        i = 0
        for individual in np.vstack([self.P, R]):
            population[i, 0] = self.J(individual[0:self.d])
            population[i, 1:] = individual
            i = i+1

        sorted_population = population[np.argsort(population[:, 0])]
        return sorted_population[-self.mi:, 1:]

    def iteration(self):
        T = self.generate_T()
        R = self.reproduce(T)
        self.P = self.choose_mi_best(R)
        print(self.P)

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


def evaluate(args):
    return sum(-(args + 5) ** 2)


pop = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10], [10, 11, 12, 13], [13, 14, 15, 16], [16, 17, 18, 19]])
evAlg = EvolutionaryAlgorithm(pop, evaluate, 10)
for i in range(100):
    evAlg.iteration()
