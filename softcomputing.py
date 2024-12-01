import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from deap import base, creator, tools, algorithms
import random

# Define fuzzy variables
temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')  # Room temperature (0 to 40)
setpoint = ctrl.Antecedent(np.arange(0, 41, 1), 'setpoint')  # Desired temperature (0 to 40)
heat = ctrl.Consequent(np.arange(0, 11, 1), 'heat')  # Heat required (0 to 10)

# Define fuzzy membership functions
temperature['cold'] = fuzz.trimf(temperature.universe, [0, 0, 20])
temperature['warm'] = fuzz.trimf(temperature.universe, [0, 20, 40])
temperature['hot'] = fuzz.trimf(temperature.universe, [20, 40, 40])

setpoint['low'] = fuzz.trimf(setpoint.universe, [0, 0, 20])
setpoint['medium'] = fuzz.trimf(setpoint.universe, [0, 20, 40])
setpoint['high'] = fuzz.trimf(setpoint.universe, [20, 40, 40])

heat['low'] = fuzz.trimf(heat.universe, [0, 0, 5])
heat['medium'] = fuzz.trimf(heat.universe, [0, 5, 10])
heat['high'] = fuzz.trimf(heat.universe, [5, 10, 10])

# Define fuzzy rules
rule1 = ctrl.Rule(temperature['cold'] & setpoint['high'], heat['high'])
rule2 = ctrl.Rule(temperature['warm'] & setpoint['medium'], heat['medium'])
rule3 = ctrl.Rule(temperature['hot'] & setpoint['low'], heat['low'])

# Control system
fuzzy_control = ctrl.ControlSystem([rule1, rule2, rule3])
fuzzy_system = ctrl.ControlSystemSimulation(fuzzy_control)

# Define a fitness function for Genetic Algorithm
def fitness(individual):
    temperature['cold'].mf = fuzz.trimf(temperature.universe, [0, individual[0], individual[1]])
    temperature['warm'].mf = fuzz.trimf(temperature.universe, [individual[1], individual[2], 40])
    temperature['hot'].mf = fuzz.trimf(temperature.universe, [individual[2], 40, 40])
    
    setpoint['low'].mf = fuzz.trimf(setpoint.universe, [0, individual[3], individual[4]])
    setpoint['medium'].mf = fuzz.trimf(setpoint.universe, [individual[4], individual[5], 40])
    setpoint['high'].mf = fuzz.trimf(setpoint.universe, [individual[5], 40, 40])

    fuzzy_system.input['temperature'] = 25  # Sample room temperature
    fuzzy_system.input['setpoint'] = 30  # Desired temperature
    
    fuzzy_system.compute()
    
    # The fitness is the inverse of the error (difference between desired heat and computed heat)
    error = abs(fuzzy_system.output['heat'] - 7)  # Assume optimal heat is 7
    return (error,)

# Genetic Algorithm setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize the error
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 20)  # Random floating point values for fuzzy parameters
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=6)  # 6 parameters
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=2, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", fitness)

# Create the population
population = toolbox.population(n=50)

# Run the Genetic Algorithm
algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=40, verbose=True)

# Output the best individual
best_individual = tools.selBest(population, 1)[0]
print("Best solution:", best_individual)
