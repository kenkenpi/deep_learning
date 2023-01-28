import numpy as np
import matplotlib.pyplot as plt
import math

pi = math.pi
x1 = np.arange(0, pi, 0.01)
x2 = np.arange(pi, 2 * pi, 0.01)
num1 = np.size(x1)
noise1 = np.random.uniform(-0.1, 0.5, num1)
noise2 = np.random.uniform(-0.5, 0.1, num1) - 0.5
y1 = np.sin(x1) + noise1
y2 = np.sin(x2) + noise2 + 1
plt.scatter(x1, y1)
plt.scatter(x1 + pi / 2, y2)
plt.show()
#############################################################

# 子函数
# 设置激活函数
# def sigmoid(x):
# sigmoid=1/(1+np.exp(-1*x))
# return(sigmoid)
# print(sigmoid(0))
# https://www.cnblogs.com/zhhy236400/p/9873322.html
def sigmoid(inx):
    if inx >= 0:  # 对sigmoid函数的优化，避免了出现极大的数据溢出
        sigmoid = 1 / (1 + math.exp(-inx))
        return sigmoid
    else:
        sigmoid = math.exp(inx) / (1 + math.exp(inx))
        return sigmoid


print(sigmoid(1))


def dsigmoid(x):
    dsigmoid = sigmoid(x) * (1 - sigmoid(x))
    return dsigmoid


# 设置标签函数
def T(i):
    if i <= num1:
        T = 0
    else:
        T = 1
    return T
print(T(70))

#######################################################

X1 = np.array(x1)
Y1 = np.array(y1)
Y2 = np.array(y2)
X = [[X1, Y1], [X1, Y2]]
X = np.hstack(X)
XT = X.T
print(np.shape(XT))

# 初始化
N = 1000
# w1 = np.array([[0.01, -0.21], [0.03, -0.04]])  # w1:2x2
# w2 = np.array([[-0.01, 0.11], [-0.03, 0.3]])  # w2:2x2
# b1 = np.array([[0], [0]])  # b1:2x1
# b2 = np.array([[0], [0]])  # b2:2x1
# v = np.array([[0.001], [-0.002]])  # v:2x1
# c = 0

w1 = np.random.randn(2, 2) / np.sqrt(2)
w2 = np.random.randn(2, 2) / np.sqrt(2)
v = np.random.randn(2, 1) / np.sqrt(2)
b1 = np.array([[0], [0]])  # b1:2x1
b2 = np.array([[0], [0]])  # b2:2x1
c = 0


sum_w1 = np.zeros((2, 2))
sum_w2 = np.zeros((2, 2))
sum_b1 = np.zeros((2, 1))
sum_b2 = np.zeros((2, 1))
sum_v = np.zeros((2, 1))
sum_c = 0
h_w1 = np.array([[0.1, 0.1], [0.1, 0.1]])
h_w2 = np.array([[0.1, 0.1], [0.1, 0.1]])
h_b1 = np.array([[0.1, 0.1], [0.1, 0.1]])
h_b2 = np.array([[0.1, 0.1], [0.1, 0.1]])
h_v = np.array([[0.1, 0.1], [0.1, 0.1]])
h_c = 0.8
eta = 0.001
fp1 = np.zeros((2, 1))
fp2 = np.zeros((2, 1))

###################################################

for i in range(N):
    # 由前往后算各层的值
    for j in range(2 * num1):
        # 第一层
        p1 = np.dot(w1, X[..., j]) + b1
        p11 = p1[0, 0]
        p12 = p1[1, 0]
        fp1[0, 0] = sigmoid(p11)
        fp1[1, 0] = sigmoid(p12)

        # 第二层
        p2 = np.dot(w2, fp1) + b2
        p21 = p2[0, 0]
        p22 = p2[1, 0]
        fp2[0, 0] = sigmoid(p21)
        fp2[1, 0] = sigmoid(p22)

        # 第三层
        q = np.matmul(v.T, fp2) + c  # v:3x1,fp:1x3
        y = sigmoid(q)
        # 由后往前对权值进行更新
        # 第三层
        delta_v = (T(j) - y) * fp2
        delta_c = (T(j) - y)
        # 第二层
        dq_dp2_1 = v[0, 0] * dsigmoid(p21)
        dq_dp2_2 = v[1, 0] * dsigmoid(p22)
        dq_dp2 = np.array([[dq_dp2_1], [dq_dp2_2]])

        delta_b2 = delta_c * dq_dp2
        # print(np.shape(delta_b))
        # print(np.shape(XT[j,...]))
        delta_w2 = np.dot(delta_b2, fp1.T)

        # 第一层
        dp2_dp1_11 = w2[0, 0] * dsigmoid(p21)
        dp2_dp1_12 = w2[0, 1] * dsigmoid(p22)
        dp2_dp1_21 = w2[1, 0] * dsigmoid(p21)
        dp2_dp1_22 = w2[1, 1] * dsigmoid(p22)
        dp2_dp1 = np.array([[dp2_dp1_11, dp2_dp1_12], [dp2_dp1_21, dp2_dp1_22]])

        delta_b1 = delta_c * np.dot(dp2_dp1, dq_dp2)
        delta_w1 = np.dot(delta_b1, [XT[j, ...]])

        h_w1 = h_w1 + np.dot(delta_w1, delta_w1)
        h_w2 = h_w2 + np.dot(delta_w2, delta_w2)
        h_b1 = h_b1 + np.dot(delta_b1, delta_b1.T)
        h_b2 = h_b2 + np.dot(delta_b2, delta_b2.T)
        h_v = h_v + np.dot(delta_v, delta_v.T)
        h_c = h_c + np.dot(delta_c, delta_c)

        sum_w1 = sum_w1 + np.dot(1/np.sqrt(h_w1), delta_w1)
        sum_w2 = sum_w2 + np.dot(1/np.sqrt(h_w2), delta_w2)
        sum_b1 = sum_b1 + np.dot(1/np.sqrt(h_b1), delta_b1)
        sum_b2 = sum_b2 + np.dot(1/np.sqrt(h_b2), delta_b2)
        sum_v = sum_v + np.dot(1/np.sqrt(h_v), delta_v)
        sum_c = sum_c + 1/np.sqrt(h_c) * delta_c
        # print("fp1=", fp1)

    w1 = w1 + sum_w1 * eta
    w2 = w2 + sum_w2 * eta
    b1 = b1 + sum_b1 * eta
    b2 = b2 + sum_b2 * eta
    v = v + sum_v * eta
    c = c + sum_c * eta

print("w1=", w1)
print("w2=", w2)
print("b1=", b1)
print("b2=", b2)
# print(v)
# print(c)
# print(DELTA_W)

Y = np.zeros((1, 2 * num1))
YY = np.zeros((1, 2 * num1))
Q = np.zeros((1, 2 * num1))
FP = np.zeros((2 * num1, 2))

####################################################

# 分类结果
for j in range(2 * num1):
    # 第一层
    p1 = np.dot(w1, X[..., j]) + b1
    p11 = p1[0, 0]
    p12 = p1[1, 0]
    fp1[0, 0] = sigmoid(p11)
    fp1[1, 0] = sigmoid(p12)

    # 第二层
    p2 = np.dot(w2, fp1) + b2
    p21 = p2[0, 0]
    p22 = p2[1, 0]
    fp2[0, 0] = sigmoid(p21)
    fp2[1, 0] = sigmoid(p22)

    # 第三层
    q = np.matmul(v.T, fp2) + c  # v:2x1,fp:1x2
    y = sigmoid(q)
    Y[0, j] = y
    Q[0, j] = q
# print(Q)
print(Y)
# print(FP)

m1 = 0
m2 = 0
for k in range(0, num1 - 1, 1):
    if Y[0, k] < 0.5:
        m1 = m1 + 1
for l in range(0, num1 - 1, 1):
    if Y[0, l + num1] >= 0.5:
        m2 = m2 + 1
prec = (m1 + m2) / (2 * num1)
print("准确率为：", prec)
print(m1)
print(m2)
