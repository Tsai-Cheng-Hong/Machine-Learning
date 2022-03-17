import random
import numpy as np
import matplotlib.pyplot as plt  

def rand_seed(m, b, num=2):
    # create empty list
    x_coor = []
    y_coor = []
    label = []
    # positive and negtive point number
    pos_num = int(num / 2)
    neg_num = num - pos_num
    # random create point
    for i in range(pos_num):
        x = random.randint(0, 30)
        r = random.randint(1, 30)
        y = m * x + b - r
        # save the coordinate of x and y
        x_coor.append(x)
        y_coor.append(y)
        # save label, right=1, left=0
        label.append(1 if m >= 0 else -1)

    for i in range(neg_num):
        x = random.randint(0, 30)
        r = random.randint(1, 30)
        y = m * x + b + r
        x_coor.append(x)
        y_coor.append(y)
        label.append(-1 if m >= 0 else 1)
    return x_coor, y_coor, label

if __name__ == '__main__':
    # set value of m and b 
    m, b = 1, 2
    # plot the function curve
    x = np.arange(30)   # x = [0, 1,..., 29]
    y = m * x + b
    plt.plot(x, y)
    # plot the random point
    # blue for positive and red for negative
    x_coor, y_coor, label = rand_seed(m, b, num=30)
    plt.plot(x_coor[:15], y_coor[:15], 'o', color='blue')
    plt.plot(x_coor[15:], y_coor[15:], 'o', color='red')
    plt.show()