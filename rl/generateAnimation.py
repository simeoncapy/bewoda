import constantes as cst
from Motor import *
import numpy as np
import roboticstoolbox as rtb
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


robot = cst.ROBOT 

print(robot)


pyplot = rtb.backends.PyPlot.PyPlot()
#robot.plot([0, 0, 0], block=True)

fileName = "archive/motors-2022-09-29_18-36-58-565968(151_pts)_490"
f = open(fileName + ".traj", 'r')
temp = f.read().splitlines()
while temp[0][0] == "<":
    temp.pop(0)

input_list = []
for myline in temp:
    input_list.append(myline.split(cst.SEPARATOR))
    input_list[-1] = [float(x) for x in input_list[-1]]

f.close()


#pyplot.launch(realtime=True)
#pyplot.add(robot)
#for q in (input_list):
#    robot.q[:] = q[:]
#    # Step the simulator by 50 ms
#    pyplot.step(0.001)

#pyplot.hold()

print(np.array(input_list))
plt.plot(np.array(input_list))
plt.legend(["M1", "M2", "M3"])
plt.show()
# robot.plot(        
#         np.array(input_list),
#         backend='pyplot',
#         dt=0.001,
#         movie=fileName+"2.gif",
#         block=True
#     )

#pyplot.hold()