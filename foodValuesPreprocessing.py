
import numpy as np
from sklearn.preprocessing import normalize
from decimal import getcontext, Decimal

# normalization for food values, carbs and calories respectivly
carbsArray = np.array([0.5, 0.5, 1, 0.1,0.6,0,17,2])
calorieArray = np.array([27, 16, 5, 1,26,8,107,9])
fruitnames = ["chicken","avocado","tomato","salad","walnut","egg","ryebread","apple"]

norm_carbs = carbsArray / np.linalg.norm(carbsArray)

norm_calories = calorieArray / np.linalg.norm(calorieArray)
#print("calories: ", str(norm_calories))

i = 0
for name in fruitnames:
    print("********")
    print(name)
    print("calories: ", Decimal(norm_calories[i]).quantize(Decimal('0.00')))
    print("carbs: ", Decimal(norm_carbs[i]).quantize(Decimal('0.00')))
    i = i +1