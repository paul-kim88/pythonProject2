#Pyomo modelin & optimization example 2_1 (Maximize)

#Concrete model

from pyomo.environ import *
from pyomo.opt import SolverFactory

#Concrete model 객체 생성
model = ConcreteModel()

#변수
model.x = Var([0,1,2],initialize=2)

#Objective function (Maximize)
model.obj = Objective(expr = -model.x[0]**2 - model.x[1]**2 - model.x[2]**2 + 3*model.x[0]*model.x[1] + model.x[2],
                     sense = maximize)

#Constraint
model.con1 = Constraint(expr = model.x[0] + model.x[1] + model.x[2] <= 40)
# model.con2 = Constraint(expr = -model.x[0]**2 - model.x[1] - model.x[2] <= -10)

#Solver 지정
opt = SolverFactory('ipopt')

#최적화 풀기
print(opt.solve(model))

#최적화 결과
# print("Objective function의 값은: \n", model.obj())
print("최적화 결과는: \n")
model.display()
