"""
Testing module for evolutionary algorithms
PSZT 2020
Marcin Hanas
Radosław Tuzimek
"""
import numpy as np
from ModifiedEvolutionaryAlgorithm import ModifiedEvolutionaryAlgorithm 
from EvolutionaryAlgorithm import EvolutionaryAlgorithm 
from cec17_functions import cec17_test_func
import matplotlib.pyplot as plt
import time

# Evaluate function from CEC17 functions 
def evaluate(args,func_num):
    # nx: Number of dimensions
    nx = len(args)
    # mx: Number of objective functions
    mx = 1
    # Pointer for the calculated fitness
    f = [0]
    
    cec17_test_func(args, f, nx, mx, func_num)

    return f[0]

# Choosing CEC2017 function and number of dimensions
CEC_function_number = int(input("Enter the CEC 2017 function number: "))
dimensions = int(input("Enter number of dimensions: "))

while (dimensions != 2 and dimensions != 10 and dimensions != 20 and dimensions != 30 and dimensions != 50 and dimensions != 100):
    print("Error: Test functions are only defined for D=2,10,20,30,50,100.")
    dimensions = int(input("Please enter number of dimensions again: "))

lambdaa = 30
initial_population_size = 20
iteration_count = int(10000*dimensions/lambdaa)

# Variables for statistics
i_count = 25
best_ev = np.zeros([i_count, dimensions])
best_mod = np.zeros([i_count, dimensions])
best_ev_val = np.zeros(i_count)
best_mod_val = np.zeros(i_count)
time_average_ev = 0
time_average_mod = 0

for k in range(i_count):
    # TEST
    print("\n/////TEST: {}/////\n".format(k))
    # Population for one test
    ## Randomization of the initial population for modified evolutionary algorithm (pairs)
    m_random_initial_population = 10000 * np.random.rand(int(initial_population_size / 2), 2, 2 * dimensions)
    ## Initial population for evolutionary algorithm (no pairs, but the same for both algorithms)
    e_random_initial_population = np.empty([m_random_initial_population.shape[0] * 2, 2 * dimensions])
    i = 0
    for individual in m_random_initial_population:
        e_random_initial_population[i] = individual[0, 0:2 * dimensions]
        e_random_initial_population[i + 1] = individual[1, 0:2 * dimensions]
        i = i + 2

    # Evolutionary Algorithm
    evAlg = EvolutionaryAlgorithm(e_random_initial_population, evaluate, CEC_function_number, lambdaa, iteration_count)
    start = time.clock()
    best_ev[k] = evAlg.run()
    end = time.clock()
    time_average_ev = time_average_ev + end - start
    best_ev_val[k] = evaluate(best_ev[k], CEC_function_number)
    print("Evolutionary Algorithm:\n \n\t-Best:")
    print(best_ev[k])
    print("\n\t-Value of evaluate function:")
    print("\t{}".format(best_ev_val[k]))

    # Modified Evolutionary Algorithm
    modEvAlg = ModifiedEvolutionaryAlgorithm(m_random_initial_population, evaluate, CEC_function_number, lambdaa, iteration_count)
    start = time.clock()
    best_mod[k] = modEvAlg.run()
    end = time.clock()
    time_average_mod = time_average_mod + end - start
    best_mod_val[k] = evaluate(best_mod[k], CEC_function_number)
    print("\nModified Evolutionary Algorithm:\n \n\t-Best:")
    print(best_mod[k])
    print("\n\t-Value of evaluate function:")
    print("\t{}".format(best_mod_val[k]))

# Calculate averages
time_average_ev = time_average_ev/i_count
time_average_mod = time_average_mod/i_count

# Show statistics
print("\n///////AVERAGES///////\n")

print("Evolutionary Algorithm:\n \n\t-Average of bests:")

#plt.plot(best_ev[:, 0], best_ev[:, 1], 'ro')
#plt.show()
print(np.mean(best_ev, axis=0))
print(np.std(best_ev, axis=0))
print("\n\t-Average and standard deviation of evaluate function:")
print("\t{}".format(np.mean(best_ev_val)))
print("\t{}".format(np.std(best_ev_val)))
print("\n\t-Average time:")
print("\t{0:02f}s".format(time_average_ev))
plt.clf()
plt.plot(best_ev[:, 0], best_ev[:, 1], 'ro')
plt.xlabel("x2")
plt.ylabel("x1")
plt.title("Znalezione minima przez standardowy \nalgorytm ewolucyjny dla funkcji {}. CEC2017".format(CEC_function_number))
plt.savefig("std_{}_{}_{}_{}.pdf".format(CEC_function_number, dimensions, initial_population_size, lambdaa))


print("\nModified Evolutionary Algorithm:\n \n\t-Average of bests:")
#plt.plot(best_mod[:, 0], best_mod[:, 1], 'ro')
#plt.show()
print(np.mean(best_mod, axis=0))
print(np.std(best_mod, axis=0))
print("\n\t-Average and standard deviation of evaluate function:")
print("\t{}".format(np.mean(best_mod_val)))
print("\t{}".format(np.std(best_mod_val)))
print("\n\t-Average time:")
print("\t{0:02f}s".format(time_average_mod))
plt.clf()
plt.plot(best_mod[:, 0], best_mod[:, 1], 'ro')
plt.xlabel("x2")
plt.ylabel("x1")
plt.title("Znalezione minima przez zmodyfikowany \nalgorytm ewolucyjny dla funkcji {}. CEC2017".format(CEC_function_number))
plt.savefig("mod_{}_{}_{}_{}.pdf".format(CEC_function_number, dimensions, initial_population_size, lambdaa))
