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

# Randomization of the initial population for modified evolutionary algorithm
m_random_initial_population=1000*np.random.rand( int(initial_population_size/2), 2, 2*dimensions)
# 

for i in range(3):
    print("TEST: ", i)
    modEvAlg = ModifiedEvolutionaryAlgorithm(m_random_initial_population, evaluate, CEC_function_number, lambdaa, iteration_count)
    #evAlg = EvolutionaryAlgorithm(e_random_initial_population, evaluate, CEC_function_number, lambdaa, iteration_count)
    
    print(modEvAlg.run())

#print(modEvAlg.run())
#print(evAlg.run())