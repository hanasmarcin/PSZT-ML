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

lambdaa = 60
initial_population_size = 40
iteration_count = int(1000*dimensions/lambdaa)
print("iteration count: {}".format(iteration_count))
# Variables for statistics
i_count = 10
best_ev = np.zeros([i_count, dimensions])
best_mod = np.zeros([i_count, dimensions])
best_ev_val = np.zeros(i_count)
best_mod_val = np.zeros(i_count)

for k in range(i_count):
    # TEST
    print("\n/////TEST: {}/////\n".format(k))

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
    best_ev[k] = evAlg.run()
    best_ev_val[k] = evaluate(best_ev[k], CEC_function_number)
    # print("Evolutionary Algorithm:\n \n\t-Best:")
    # print(best_ev[k])
    # print("\n\t-Value of evaluate function:")
    # print("\t{}".format(best_ev_val[k]))

    # Modified Evolutionary Algorithm
    modEvAlg = ModifiedEvolutionaryAlgorithm(m_random_initial_population, evaluate, CEC_function_number, lambdaa, iteration_count)
    best_mod[k] = modEvAlg.run()
    best_mod_val[k] = evaluate(best_mod[k], CEC_function_number)
    # print("\nModified Evolutionary Algorithm:\n \n\t-Best:")
    # print(best_mod[k])
    # print("\n\t-Value of evaluate function:")
    # print("\t{}".format(best_mod_val[k]))

# Show statistics
print("\n///////AVERAGES///////\n")

print("Evolutionary Algorithm:\n \n\t-Average of bests:")
print(np.mean(best_ev, axis=0))
print(np.std(best_ev, axis=0))
print("\n\t-Average and standard deviation of evaluate function:")
print("\t{}".format(np.mean(best_ev_val)))
print("\t{}".format(np.std(best_ev_val)))

print("\nModified Evolutionary Algorithm:\n \n\t-Average of bests:")
print(np.mean(best_mod, axis=0))
print(np.std(best_mod, axis=0))
print("\n\t-Average and standard deviation of evaluate function:")
print("\t{}".format(np.mean(best_mod_val)))
print("\t{}".format(np.std(best_mod_val)))
