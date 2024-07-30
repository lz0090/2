import random
import numpy as np

def split_number(num, parts):
    numbers = [random.randint(1, num // parts) for i in range(parts - 1)]
    numbers.append(num - sum(numbers))
    return numbers

# test = np.load('E:/project/PPO/loc_delay/loc_delay_dynamic_psi.npy')
# print(test)
