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

fileName = "archive/motors-2022-10-25_03-45-22-802181(301_pts)_0"
f = open(fileName + ".traj", 'r')
temp = f.read().splitlines()
while temp[0][0] == "<":
    temp.pop(0)

input_list = []
input_total = []

color = []
pad = []
for myline in temp:
    split = myline.split(cst.SEPARATOR)
    input_list.append(split[0:cst.NUMBER_OF_MOTOR])
    input_list[-1] = [float(x) for x in input_list[-1]]

    t = cst.PALETTE[split[cst.NUMBER_OF_MOTOR]].toList()
    t.append(int(split[cst.NUMBER_OF_MOTOR+1]))
    t2 = tuple(ti/255 for ti in t)
    color.append(t2)

    pad.append(split[cst.NUMBER_OF_MOTOR+2:])
    pad[-1] = [float(x) for x in pad[-1]]

    input_total.append(split[0:cst.NUMBER_OF_MOTOR] + [int(split[cst.NUMBER_OF_MOTOR+1])/255])
    input_total[-1] = [float(x) for x in input_total[-1]]


f.close()


#pyplot.launch(realtime=True)
#pyplot.add(robot)
#for q in (input_list):
#    robot.q[:] = q[:]
#    # Step the simulator by 50 ms
#    pyplot.step(0.001)

# plt.show()
# pyplot.hold()

graph = "pad"
if graph == "m":
    data = input_list
    legend = ["M1", "M2", "M3"]
elif graph == "pad":
    data = pad
    legend = ["P", "A", "D"]

#print(np.array(input_total))
plt.plot(np.array(data))
plt.legend(legend)
plt.savefig(fileName + "_" + "-".join(legend) + ".png")
plt.show()



#exit()

robot.plot(        
        np.array(input_list),
        backend='pyplot',
        dt=0.001,
        movie=fileName+".apng",
        block=True,
        color=color,
        printEach=True
    )

#pyplot.hold()