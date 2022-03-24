import matplotlib.pyplot as plt
import numpy as np
import random
#網路上找的dataset 可以線性分割

def dataset (m, b, num):
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
    return x_coor, y_coor, label, num


#判斷有沒有分類錯誤，並列印錯誤率

def check_error(w, x, y, label, num):
    result = None
    global  itr
    error = 0
    for i in range(num):
        aa = np.array((1,x[i],y[i])) #problem
        if int(np.sign(w.T.dot(aa))) != label[i]:
            result = aa, label[i]
            error += 1
    itr += 1
    print(itr)
    print("error=%s/%s" % (error, len(x)))
    return result

#PLA演算法實作

def pla(x,y,label,num):
    w = np.zeros(3) #w = [0,0,0]
    while check_error(w, x, y, label, num) is not None:
        x1, s = check_error(w, x, y, label, num)
        w += s * x1
    return w


#執行

x, y, label, num = dataset(1, 8, 30)
itr = 0
w = pla(x, y, label, num)

#畫圖
fig = plt.figure()
ax1 = fig.add_subplot(111)
#dataset前半後半已經分割好 直接畫就是

ax1.scatter([i for i in x[:int(num/2)]], [i for i in y[:int(num/2)]], s=10, c='b', marker="o", label='O')
ax1.scatter([i for i in x[int(num/2):]], [i for i in y[int(num/2):]], s=10, c='r', marker="x", label='X')
l = np.linspace(-50,50) #正副調整
a,b = -w[1]/w[2], -w[0]/w[2]
ax1.plot(l, a*l + b, 'b-')
plt.legend(loc='upper left');
plt.show()
