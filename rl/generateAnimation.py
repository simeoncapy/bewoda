import constantes as cst
from Motor import *
import numpy as np
import roboticstoolbox as rtb
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


robot = cst.ROBOT 

print(robot)


# pyplot = rtb.backends.PyPlot.PyPlot()
# robot.plot(cst.MOTOR_ORIGIN[cst.RADIAN], block=True)
# plt.show()
# pyplot.hold()
# exit()

fileName = "archive/motors-2022-10-10_23-40-45-814919(301_pts)_499"
f = open(fileName + ".traj", 'r')
temp = f.read().splitlines()
while temp[0][0] == "<":
    temp.pop(0)

input_list = []
color = []
for myline in temp:
    split = myline.split(cst.SEPARATOR)
    input_list.append(split[0:cst.NUMBER_OF_MOTOR])
    input_list[-1] = [float(x) for x in input_list[-1]]

    t = cst.PALETTE[split[cst.NUMBER_OF_MOTOR]].toList()
    t.append(int(split[cst.NUMBER_OF_MOTOR+1]))
    t2 = tuple(ti/255 for ti in t)
    color.append(t2)

f.close()


#pyplot.launch(realtime=True)
#pyplot.add(robot)
#for q in (input_list):
#    robot.q[:] = q[:]
#    # Step the simulator by 50 ms
#    pyplot.step(0.001)

# plt.show()
# pyplot.hold()

# print(np.array(input_list))
# plt.plot(np.array(input_list))
# plt.legend(["M1", "M2", "M3"])
# plt.show()

robot.plot(        
        np.array(input_list),
        backend='pyplot',
        dt=0.001,
        movie=fileName+".gif",
        block=True,
        color=color
    )

#pyplot.hold()