import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from mpl_toolkits import mplot3d

# multiple regression: y = b0 + b1x1 + b2x2
# x1: Income composition of resources
# x2: Average GDP of countries
# y: Life expectancy of citizens

file_name = "project_data.csv"
x1 = "Income composition of resources"
x2 = "GDP"
y = "Life expectancy"

df = pd.read_csv(file_name)

X1 = []
X2 = []
Y = []
final_list_x = []
final_list_y = []

sumx1 = sumx2 = sumy = sumx1_2 = sumx2_2 = sumy_2 = sumx1y = sumx2y = 0
b0 = b1 = b2 = 0
n = 0

for index, row in df.iterrows():
    sumx1 += row[x1]
    sumx2 += row[x2]
    sumy += row[y]
    sumx1_2 += row[x1]**2
    sumx2_2 += row[x2]**2
    sumy_2 += row[y]**2
    sumx1y += row[x1]*row[y]
    sumx2y += row[x2]*row[y]

for index, row in df.iterrows():
    X1.append(row[x1])
    X2.append(row[x2])
    Y.append(row[y])
    temp = [1, row[x1],row[x2]]
    temp1 = [row[y]]
    final_list_x.append(temp)
    final_list_y.append(temp1)
    n+=1

x = np.array(final_list_x)#X
y = np.array(final_list_y)#Y
transposedx = np.transpose(x)#X'

test_x = np.dot(transposedx,x)#(X'X)
test_x_inv = np.linalg.inv(test_x) #(X'X)^-1
test_y = np.dot(transposedx, y)#(X'Y)
b = np.dot(test_x_inv,test_y)#(X'X)^-1*(X'Y)

print(b)
print("----------------")
print("b0:", b[0][0])
print("b1:", b[1][0])
print("b2:", b[2][0])
b0 = b[0][0]
b1 = b[1][0]
b2 = b[2][0]
print("y = ", b0, " + ", b1, "* x1 + ", b2, "* x2")

delta_y = 0.0
y_hat = []

for index, row in df.iterrows():
    delta_y = (b0+b1*(row[x1])+b2*(row[x2]))#y-hat
    y_hat.append([delta_y])#y-hat

y_hat_np = np.array(y_hat)
print()
print("y: ",y)
print()
print("y_hat_np: ",y_hat_np)

error_e = y-y_hat_np
print("---------------------------")
print("error: ", error_e)

print("-------------------------------------")
SSE = np.dot(np.transpose(error_e), error_e)

print("SSE: ", SSE)

MSE = (SSE/(n-3))

print("MSE: ", MSE)

print("-------------------")

lamda1 = test_x_inv[0][0]
lamda2 = test_x_inv[1][1]
lamda3 = test_x_inv[2][2]
se_b0_hat = math.sqrt(MSE) * math.sqrt(lamda1)
se_b1_hat = math.sqrt(MSE) * math.sqrt(lamda2)
se_b2_hat = math.sqrt(MSE) * math.sqrt(lamda3)
print("se_b0_hat = ",se_b0_hat)
print("se_b1_hat = ",se_b1_hat)
print("se_b2_hat = ",se_b2_hat)

x1_bar = sumx1/n
x2_bar = sumx2/n
y_bar = sumy/n

print("---Coefficient of correlation bw x1 and y----")
r_x1y = (sumx1y - n*x1_bar*y_bar)/(math.sqrt(sumx1_2 - n*((x1_bar)**2)) * (math.sqrt(sumy_2) - n*((y_bar)**2)))
print("r_x1y = ", r_x1y)

print("---Coefficient of correlation bw x2 and y----")

r_x2y = (sumx2y - n*x2_bar*y_bar)/(math.sqrt(sumx2_2 - n*((x2_bar)**2)) * (math.sqrt(sumy_2) - n*((y_bar)**2)))
print("r_x2y = ", r_x2y)

print("y = ", b0, " + ", b1, "* x1 + ", b2, "* x2")

#-----Graph-------
minx1 = df[x1].min()
maxx1 = df[x1].max()
minx2 = df[x2].min()
maxx2 = df[x2].max()

x1_val = np.linspace(0.3, 0.9)
x2_val = np.linspace(10, 100000)
y_hat_val = b0+b1*(x1_val)+b2*(x2_val)

fig = plt.figure()
graph = plt.axes(projection='3d')
graph.scatter(X1, X2, Y, c=Y)
graph.set_xlabel("Income composition of resources")
graph.set_ylabel("GDP")
graph.set_zlabel("Life expectancy")
graph.plot(x1_val, x2_val, y_hat_val, label = 'y = B0 + B1(X1) + B2(X2)')
graph.legend()
fig.tight_layout()
plt.show()

#-----Hypothesis testing--------
t1 = b1/se_b1_hat
t2 = b2/se_b2_hat
print("t1 = ", t1)
print()
print("t2 = ", t2)