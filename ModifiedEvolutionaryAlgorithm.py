import numpy as np
import matplotlib.pyplot as plt

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
        """
        One iteration of unmodified evolutionary algorithm
        """
        T = self.generate_T()
        R = self.reproduce(T)
        self.P = self.choose_new_population(R)

    def generate_T(self):
        """
        Method generates temporary population, which will be reproduced
        :return: paired temporary population (array sized lambda/2 x 2 x 2d)
        """
        T = np.empty([int(self.lmbd/2), 2, self.d * 2])
        # loop for sampling (with replacement) lambda pairs to reproduce
        for i in range(int(self.lmbd/2)):
            random_id = np.random.randint(low=0, high=int(self.mi/2-1))
            T[i, :, :] = self.P[random_id, :, :]

        return T

    def reproduce(self, T):
        """
        Method creates new individuals from T by mutation
        :param T: paired temporary population (array sized lambda/2 x 2 x 2d)
        :return: unpaired children population (array sized lambda x 2d)
        """
        R = np.empty([self.lmbd, self.d * 2])

        for i in range(0, self.lmbd, 2):
            #print(self.lmbd)
            #print(i)
            R[i, :] = self.mutate(T[int(i/2), 0])
            R[i+1, :] = self.mutate(T[int(i/2), 1])

        return R

    def pair_children(self, R):
        """
        Method puts children in random pairs
        :param R: unpaired children population (array sized lambda x 2d)
        :return: paired children population (array sized lambda/2 x 2 x 2d)
        """
        R_paired = np.empty([int(self.lmbd/2), 2, self.d * 2])

        random_ids = np.random.randint(low=0, high=R.shape[0] - 1, size=R.shape[0])
        R = R[random_ids]
        for i in range(0, self.lmbd, 2):
            R_paired[int(i/2), 0] = R[i]
            R_paired[int(i/2), 1] = R[i+1]
        return R_paired

    def choose_new_population(self, R):
        """
        Method creates new population
        :param R: unpaired children population (array sized lambda x 2d)
        
        """
        R_paired = self.pair_children(R)
        return self.choose_mi_best(R_paired)

    def choose_mi_best(self, R_paired):
        """
        Method chooses mi best individuals from paired children and current population
        :param R_paired: paired children population (array sized lambda/2 x 2 x 2d)
        :return: new paired population (array sized lambda/2 x 2 x 2d)
        """
        population = np.vstack([self.P, R_paired])
        eval_values = np.empty(population.shape[0])
        i = 0
        for individual in population:
            # Calculating mean as a fitness function for both elements in pair
            eval_values[i] = -(self.J(individual[0, 0:self.d], self.nCEC) + self.J(individual[1, 0:self.d], self.nCEC)) / 2
            i = i+1

        sorted_population = population[np.argsort(eval_values)]
        return sorted_population[-int(self.mi/2):]

    def choose_best(self):
        """
        Method chooses one best individual to end an algorithm
        :return: best individual (array sized 1*d)
        """
        # Final population without pairs
        end_pop = np.empty([self.P.shape[0] * 2, 2 * self.d])
        i = 0
        for individual in self.P:
            end_pop[i] = individual[0, 0:2 * self.d]
            end_pop[i + 1] = individual[1, 0:2 * self.d]
            i = i + 2

        # Sorting final population
        population = np.empty([end_pop.shape[0], 2 * self.d + 1])
        i = 0
        for individual in end_pop:
            population[i, 0] = -self.J(individual[0:self.d], self.nCEC)
            population[i, 1:] = individual
            i = i + 1
        sorted_population = population[np.argsort(population[:, 0])]

        #print("Best from modif")
        #print(sorted_population[:, 1:self.d+1])
        #plt.plot(sorted_population[:, 1], sorted_population[:, 2], 'ro')
        #plt.show()
        return sorted_population[-1, 1:self.d+1]

    # @staticmethod
    # def crossover(f, m):
    #     """
    #     Function makes new individual by crossover on its parents f and m
    #     :param f: first parent, matrix sized 1*2d, first d rows are values of each individual and second d rows are
    #     coefficients for normal distribution for each individual's value
    #     :param m: second parent, matrix sized 1*2d, first d rows are values of each individual and second d rows are
    #     coefficients for normal distribution for each individual's value
    #     :return: new individual
    #     """
    #     x = (f + m) / 2
    #     return x

    def mutate(self, x):
        """
        Method makes new individual from another individual by mutation
        :param x: individual to mutate (array sized 1*2d)
        :return: new individual (array sized 1*2d)
        """
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
        """
        Method runs algorithm
        :return: best individual (array sized 1*d)
        """
        for i in range(self.iter_count):
            self.iteration()

        return self.choose_best()