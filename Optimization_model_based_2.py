#모델 기반 최적화 기초 (Convex, Unconstrained)
#Gradient 정보 이용

import numpy as np
from scipy.optimize import minimize

#Convex objective function model
def obj1(x):
    return (x[0]-100)**2 + (x[1]+20)**2

#Derivative of the objective function (Numerically)
def derivative_obj1(x):
    h = 10**(-5)
    der = np.zeros_like(x)
    der[0] = (obj1([x[0]+h, x[1]])-obj1([x[0]-h, x[1]]))/2/h
    der[1] = (obj1([x[0], x[1]+h])-obj1([x[0], x[1]-h]))/2/h
    return der

#초기값 (Local optimizing algorithm이므로)
initial = np.array([0, 0])

#Broyden-Fletcher-Goldfarb-Shanno algorithm으로 최적화
result_1 = minimize(obj1, initial, method='BFGS', jac=derivative_obj1, options={'disp':True})

#결과
print("종합 정보: \n", result_1)
print("Optimal input: \n", result_1.x)
print("Optimal output: \n", result_1.fun)

#Derivative 정보 안 줬을 때: 알아서 numerical하게 계산해 줌. 그러므로 사실은 derivative_obj_convex(x)도 정의할 필요 없었다.
result_2 = minimize(obj1, initial, method='BFGS', options={'disp':True})
print("종합 정보: \n", result_2)
print("Optimal input: \n", result_2.x)
print("Optimal output: \n", result_2.fun)