import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import os
import io

currentExercise = 3



## EXERCISE 2
if currentExercise == 2:
    numbers = np.array([13, 15, 16, 16, 19, 20, 20, 21, 22, 22, 25, 25, 25, 25, 30, 33, 33, 35, 35, 35, 35, 36, 40, 45, 46, 52, 70])
    # a) What is the mean of the data? What is the median?
    mean = np.mean(numbers)
    print("mean: " + str(mean))

    median = np.median(numbers)
    print("median: " + str(median))

    # b) What is the mode of the data?
    mode = scipy.stats.mode(numbers)
    print("mode: " + str(mode))

    # c) What is the midrange of the data?
    maximum = max(numbers)
    minimum = min(numbers)
    midrange = (maximum + minimum)/2
    print("midrange: " + str(midrange))

    # d) Can you find (roughly) the first quartile  (Q1) and the third quartile (Q3) ofthe data?
    Q1 = np.percentile(numbers, 25)
    Q3 = np.percentile(numbers, 75)
    print("Q1: " + str(Q1) + ", Q3: " + str(Q3))

    # e) Give the five-number summary of the data.
    print("five number summary: " + "median: " + str(median) + ", 1st Quartile: " + str(Q1) + " , 3rd Quartile: " + str(Q3) + ", min: " + str(minimum) + ", max: " + str(maximum))

    # f) Show a boxplot of the data.
    plt.boxplot(numbers)
    plt.show()

## EXERCISE 3
if currentExercise == 3:


  #  with open(r'C:\Users\najam\PycharmProjects\dataScienceGamesCourse\Data-Science-for-Games-Spring-2019\assets\chapter2q3.csv') as csvfile:
   #     reader = csv.reader(csvfile)
    with open(r'C:\Users\najam\PycharmProjects\dataScienceGamesCourse\Data-Science-for-Games-Spring-2019\assets\chapter2q3.csv') as f:
      reader = csv.DictReader(f)

      for row in reader:
          print(row)

#Compute an approximate median value for the data.



