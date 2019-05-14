import numpy as np
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import csv
import pylab
from collections import defaultdict
import os
import io

currentExercise = 5


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


#Compute an approximate median value for the data.

## EXERCISE 3
if currentExercise == 3:

    import csv

    f = open('chapter2q3.csv')
    csv_f = csv.reader(f)

    fullArray = np.array([3194])

    i: int = 0
    rowName: int = 0
    tempNumberOfPeople: int = 0
    allNumbers = np.array([0,0,0,0,0,0,0,0,0,0,0,0]) # shouldn't be hardcoded
    numberInRow = 0

# get lowest and highest and find median
    for row in csv_f:
        if i > 0:
            s = str(row[0])
            c = ''
            fullNumber = ""

            for j in range(0, len(s)):
                c = s[j]

                if c != '-':
                    fullNumber += s[j]

                if c == '-':
                    age_int = int(fullNumber)
                    allNumbers[numberInRow] = age_int
                    numberInRow += 1
                    fullNumber = ""

                if j+1 == len(s):
                    age_int = int(fullNumber)
                    allNumbers[numberInRow] = age_int # if this happends, the counter should go up. else it takes the place of the last
                    numberInRow += 1
                    fullNumber = ""
        i += 1

    median = np.median(allNumbers)
    print("median: " + str(median))


## EXERCISE 4
if currentExercise == 4:
    print("exercise 4")

    ageAndFat = np.array(list(csv.reader(open("chapter2q4.csv", "r"), delimiter=";",))).astype("float")

    # A) Calculate the  mean, median, and standard deviation of age and %fat.
    # mean, median, and standard deviation of age
    print("AGE")
    mean = np.mean(ageAndFat[:,0 ])
    print("mean: " + str(mean))
    median = np.median(ageAndFat[:, 0])
    print("median: " + str(median))
    standardDeviation = np.std(ageAndFat[:,0 ], dtype=np.float64)
    print(str(standardDeviation))


    # mean, median, and standard deviation of %fat
    print("%FAT")
    mean = np.mean(ageAndFat[:, 1])
    print("mean: " + str(mean))
    median = np.median(ageAndFat[:, 1])
    print("median: " + str(median))
    standardDeviation = np.std(ageAndFat[:,1 ], dtype=np.float64)
    print(str(standardDeviation))

    # B) Draw the boxplots for age and %fat.
    plt.boxplot(ageAndFat)
    plt.show()

    # C) Draw a scatter plot and a q-q plot based on these two variables.
    stats.probplot(ageAndFat[:,0 ], dist="norm", plot=pylab)
    pylab.title("Q-Q PLOT OF AGE", fontdict=None, loc='center', pad=None,)
    pylab.show()

    stats.probplot(ageAndFat[:,1 ], dist="norm", plot=pylab)
    pylab.title("Q-Q PLOT OF %FAT", fontdict=None, loc='center', pad=None,)
    pylab.show()

    plt.scatter(ageAndFat[:,0 ], ageAndFat[:,1 ],)
    plt.title("SCATTER PLOT OF AGE AND %FAT")
    plt.xlabel("AGE")
    plt.ylabel("%FAT")
    plt.show()

# EXERCISE 5
if currentExercise == 5:
    print("exercise 5")
