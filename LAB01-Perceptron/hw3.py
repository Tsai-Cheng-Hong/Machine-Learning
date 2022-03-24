import random
import numpy as np
import matplotlib.pyplot as plt
import datetime


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
        x = random.randint(0, 5000)
        r = random.randint(1, 5000)
        y = m * x + b - r
        # save the coordinate of x and y
        x_coor.append(x)
        y_coor.append(y)
        # save label, right=1, left=0
        label.append(1 if m >= 0 else -1)

    for i in range(neg_num):
        x = random.randint(0, 5000)
        r = random.randint(1, 5000)
        y = m * x + b + r
        x_coor.append(x)
        y_coor.append(y)
        label.append(-1 if m >= 0 else 1)
    return x_coor, y_coor, label


if __name__ == '__main__':
    # set value of m and b
    m, b = 5, 0
    # plot the function curve
    x = np.arange(30)  # x = [0, 1,..., 29]
    y = m * x + b
    plt.plot(x, y)
    # plot the random point
    # blue for positive and red for negative
    x_coor, y_coor, label = rand_seed(m, b, num=2000)

w_plot = np.array([[0, 0], [0, 0], [0, 0]], np.float)


def getDataSet(filename):
    # dataSet = open(filename, 'r')
    # dataSet = dataSet.readlines()
    num = len(x_coor)
    X = np.zeros((num, 2))
    Y = np.zeros((num, 1))
    for i in range(num):
        # data = dataSet[i].strip().split()
        X[i, 0] = np.float(x_coor[i])
        X[i, 1] = np.float(y_coor[i])
        Y[i, 0] = np.float(label[i])
        # print (data[0])
        # print (x_coor[i])
        # Color would be defined by its Label
        if Y[i, 0] == 1:
            color = 'blue'
        else:
            color = 'orange'
        plt.scatter(X[i, 0], X[i, 1], c=color)

    return X, Y


def sign(z, w):
    if np.dot(z, w) >= 0:
        return 1
    else:
        return -1


def PLA_Naive(X, Y, w, speed, updates):
    iterations = 0
    num = len(X)
    nump = 1000
    flag = True
    for i in range(updates):
        flag = True
        for j in range(num):
            if sign(X[j], w) != Y[j, 0]:
                flag = False
                w = w + speed * Y[j, 0] * np.matrix(X[j]).T
                w_plot.itemset((0, 0), 5000)
                w_plot.itemset((0, 1), (-1) * 5000 * w[0] / w[1])
                w_plot.itemset((1, 0), 0)
                w_plot.itemset((1, 1), 0)
                w_plot.itemset((2, 0), (0))
                w_plot.itemset((2, 1), (-1) * (0) * w[0] / w[1])
                break

            else:
                continue
        if flag == True:
            iterations = i
            break
        plt.plot(x_coor[:nump], y_coor[:nump], 'o', color='blue')
        plt.plot(x_coor[nump:], y_coor[nump:], 'x', color='orange')
    plt.plot(w_plot[:, 0], w_plot[:, 1])
    return w, flag, iterations


def error_rate(X, Y, w):
    error = 0.0
    for i in range(len(X)):
        if sign(X[i], w) != Y[i, 0]:
            error = error + 1.0
    return error / len(X)

def PLA_Pocket(X, Y, w, speed, updates):
    error = 0.0
    num = len(X)
    rand_sort = range(len(X))
    rand_sort = random.sample(rand_sort, len(X))
    for i in range(updates):
        for j in range(num):
            if sign(X[rand_sort[j]], w) != Y[rand_sort[j], 0]:
                wt = w + speed * Y[rand_sort[j], 0] * np.matrix(X[rand_sort[j]]).T
                error0 = error_rate(X, Y, w)
                error1 = error_rate(X, Y, wt)
                if error1 < error0:
                    w = wt
                    error = error1
                break
    return w, error


starttime_pla = datetime.datetime.now()

filename = 1
X, Y = getDataSet(filename)
plt.show()

w0 = np.zeros((2, 1))
speed = 1
updates = 10
updates_pla = 1000

w2, flag, iterations = PLA_Naive(X, Y, w0, speed, updates_pla)

endtime_pla = datetime.datetime.now()

print(' PLA execution time = ', endtime_pla - starttime_pla)

starttime_poc = datetime.datetime.now()

errorrate = []

for i in range(100):  # 輸出每次的錯誤率
    w1, error = PLA_Pocket(X, Y, w0, speed, updates)

    if error == 0.0:
        print("errorrate is zero")
        break

    errorrate.append(error)

endtime_poc = datetime.datetime.now()

print(' Pocket Algorithm execution time = ', endtime_poc - starttime_poc)