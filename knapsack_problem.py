#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 23:29:45 2019

@author: kalyantulabandu
"""

import mlrose
import numpy as np

problem_size = [5,10,20,50,75,100,120,140]

weights_list = []
values_list = []
state_list = []


for each in problem_size:
    weights = np.random.randint(1,16,size=each)
    values = np.random.randint(1,20,size=each)
    state = np.random.randint(0,2,size=each)
    weights_list.append(weights)
    values_list.append(values)
    state_list.append(np.array(state))
    
print(len(weights_list))
print(len(values_list))
print(len(state_list))

max_attempts = 50
max_iters = 1500
max_weight_pct = 0.45

rhc_state = []
rhc_fitness = []
rhc_statistics = []
rhc_statistics_fn_evals = []
rhc_statistics_time = []
rhc_statistics_fitness = []

sa_statistics = []
sa_statistics_fn_evals = []
sa_statistics_time = []
sa_statistics_fitness = []
sa_state = []
sa_fitness = []

ga_state = []
ga_fitness = []
ga_statistics = []
ga_statistics_fn_evals = []
ga_statistics_time = []
ga_statistics_fitness = []

mimic_statistics = []
mimic_statistics_fn_evals = []
mimic_statistics_time = []
mimic_statistics_fitness = []
mimic_state = []
mimic_fitness = []

index = 0

#Random Hill Climbing
for each in problem_size:
    fitness_fn = mlrose.Knapsack(weights_list[index], values_list[index], max_weight_pct)
    problem = mlrose.DiscreteOpt(length=each,fitness_fn=fitness_fn,maximize=True,max_val=2)
    best_state, best_fitness, statistics = mlrose.random_hill_climb(problem=problem, 
                                                      max_attempts = max_attempts, max_iters = max_iters,restarts = 10,return_statistics=True)
    rhc_state.append(best_state)
    rhc_fitness.append(best_fitness)
    rhc_statistics_fn_evals.append(statistics['fitness_evals'])
    rhc_statistics_time.append(statistics['time'])
    rhc_statistics_fitness.append(best_fitness)
    rhc_statistics.append(statistics)
    index = index + 1
    

   
   
index = 0
#Simulated Annealing
for each in problem_size:
    fitness_fn = mlrose.Knapsack(weights_list[index], values_list[index], max_weight_pct)
    problem = mlrose.DiscreteOpt(length=each,fitness_fn=fitness_fn,maximize=True,max_val=2)
    best_state, best_fitness, statistics = mlrose.simulated_annealing(problem=problem,
                                                      max_attempts = max_attempts, max_iters = max_iters,
                                                      schedule = mlrose.GeomDecay(init_temp=2.2, decay=0.7, min_temp=1),
                                                      return_statistics=True)
    sa_state.append(best_state)
    sa_fitness.append(best_fitness)
    sa_statistics_fn_evals.append(statistics['fitness_evals'])
    sa_statistics_time.append(statistics['time'])
    sa_statistics_fitness.append(best_fitness)
    sa_statistics.append(statistics)
    index = index+1
 

index = 0
#Genetic Algorithm
for each in problem_size:
    fitness_fn = mlrose.Knapsack(weights_list[index], values_list[index], max_weight_pct)
    problem = mlrose.DiscreteOpt(length=each,fitness_fn=fitness_fn,maximize=True,max_val=2)
    best_state, best_fitness, statistics = mlrose.genetic_alg(problem=problem,pop_size=200,mutation_prob=0.4,
                                                      max_attempts = max_attempts, max_iters = max_iters, return_statistics = True)
    ga_state.append(best_state)
    ga_fitness.append(best_fitness)
    ga_statistics_fn_evals.append(statistics['fitness_evals'])
    ga_statistics_time.append(statistics['time'])
    ga_statistics_fitness.append(best_fitness)
    ga_statistics.append(statistics)
    index = index+1


index = 0
#MIMIC
for each in problem_size:
    fitness_fn = mlrose.Knapsack(weights_list[index], values_list[index], max_weight_pct)
    problem = mlrose.DiscreteOpt(length=each,fitness_fn=fitness_fn,maximize=True,max_val=2)
    best_state, best_fitness, statistics = mlrose.mimic(problem=problem,pop_size=200,keep_pct=0.2,
                                                      max_attempts = max_attempts, max_iters = max_iters,return_statistics = True )
    mimic_state.append(best_state)
    mimic_fitness.append(best_fitness)
    mimic_statistics_fn_evals.append(statistics['fitness_evals'])
    mimic_statistics_time.append(statistics['time'])
    mimic_statistics_fitness.append(best_fitness)
    mimic_statistics.append(statistics)
    index = index+1
    
print(len(ga_statistics_fitness))

print(ga_statistics_fn_evals[-1])
#Plot 1 - Problem Size vs Function Evaluations
import matplotlib.pyplot as plt
plt.plot(problem_size[:-1], rhc_statistics_fn_evals[:-1])
plt.plot(problem_size[:-1], sa_statistics_fn_evals[:-1])
plt.plot(problem_size[:-1], ga_statistics_fn_evals[:-1])
plt.plot(problem_size[:-1], mimic_statistics_fn_evals[:-1])
plt.title('Knapsack - Problem Size Vs No. of function evaluations')
plt.ylabel('# of function evaluations', fontsize = 14)
plt.xlabel('Problem Size', fontsize = 14)
plt.legend(['rhc','sa','ga','mimic'], loc='upper left')
plt.ylim(0,27000)
plt.show()

#Plot 2 - Problem Size vs Time
import matplotlib.pyplot as plt
plt.plot(problem_size[:-1], rhc_statistics_time[:-1])
plt.plot(problem_size[:-1], sa_statistics_time[:-1])
plt.plot(problem_size[:-1], ga_statistics_time[:-1])
plt.plot(problem_size[:-1], mimic_statistics_time[:-1])
plt.title('Knapsack - Problem Size Vs Time (in seconds)')
plt.ylabel('Time (in seconds)', fontsize = 14)
plt.xlabel('Problem Size', fontsize = 14)
plt.legend(['rhc','sa','ga','mimic'], loc='upper left')
plt.ylim(0.9,6)
plt.show()

#Plot 3 - Problem Size vs fitness value
import matplotlib.pyplot as plt
plt.plot(problem_size[:-1], rhc_statistics_fitness[:-1])
plt.plot(problem_size[:-1], sa_statistics_fitness[:-1])
plt.plot(problem_size[:-1], ga_statistics_fitness[:-1])
plt.plot(problem_size[:-1], mimic_statistics_fitness[:-1])
plt.title('Knapsack - Problem Size Vs Fitness Value')
plt.ylabel('Fitness Value', fontsize = 14)
plt.xlabel('Problem Size', fontsize = 14)
plt.legend(['rhc', 'sa','ga','mimic'], loc='upper left')
plt.ylim(6,900)
plt.show()


