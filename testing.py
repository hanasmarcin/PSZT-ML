import numpy as np
from ModifiedEvolutionaryAlgorithm import ModifiedEvolutionaryAlgorithm 
from EvolutionaryAlgorithm import EvolutionaryAlgorithm 
from cec17_functions import cec17_test_func

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

lambdaa = 10
initial_population_size = 6
iteration_count = int(1000*dimensions/lambdaa)

# Variables for statistics
average_best_mod = np.zeros(2*dimensions)
average_best_mod_val = 0
average_best_ev = np.zeros(2*dimensions)
average_best_ev_val = 0

for i in range(30):
    # TEST
    print("\n/////TEST: ", i,"/////\n")

    # Population for one test
    ## Randomization of the initial population for modified evolutionary algorithm (pairs)
    m_random_initial_population = 1000 * np.random.rand(int(initial_population_size / 2), 2, 2 * dimensions)
    ## Initial population for evolutionary algorithm (no pairs, but the same for both algorithms)
    e_random_initial_population = np.empty([m_random_initial_population.shape[0] * 2, 2 * dimensions])
    i = 0
    for individual in m_random_initial_population:
        e_random_initial_population[i] = individual[0, 0:2 * dimensions]
        e_random_initial_population[i + 1] = individual[1, 0:2 * dimensions]
        i = i + 2

    # Evolutionary Algorithm
    evAlg = EvolutionaryAlgorithm(e_random_initial_population, evaluate, CEC_function_number, lambdaa, iteration_count)
    best_ev = evAlg.run()
    best_ev_val = evaluate(best_ev[0:dimensions], CEC_function_number)
    average_best_ev = average_best_ev + best_ev
    average_best_ev_val = average_best_ev_val + best_ev_val
    print("Evolutionary Algorithm:\n \n\t-Best:")
    print("\t",best_ev)
    print("\n\t-Value of evaluate function:")
    print("\t",best_ev_val)

    # Modified Evolutionary Algorithm
    modEvAlg = ModifiedEvolutionaryAlgorithm(m_random_initial_population, evaluate, CEC_function_number, lambdaa, iteration_count)
    best_mod = modEvAlg.run()
    best_mod_val = evaluate(best_mod[0:dimensions], CEC_function_number)
    average_best_mod = average_best_mod + best_mod
    average_best_mod_val = average_best_mod_val + best_mod_val
    print("\nModified Evolutionary Algorithm:\n \n\t-Best:")
    print("\t",best_mod)
    print("\n\t-Value of evaluate function:")
    print("\t",best_mod_val)

# Show statistics
print("\n///////AVERAGES///////\n")

print("Evolutionary Algorithm:\n \n\t-Average of bests:")
print("\t",average_best_ev)
print("\n\t-Average of value of evaluate function:")
print("\t",average_best_ev_val)

print("\nModified Evolutionary Algorithm:\n \n\t-Average of bests:")
print("\t",average_best_mod)
print("\n\t-Average of value of evaluate function:")
print("\t",average_best_mod_val)
