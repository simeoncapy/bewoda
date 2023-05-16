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

fileName = "data/motors-2023-04-20_12-58-14-339188(490_pts)_277"
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

graph = "m"
if graph == "m":
    data = input_list
    legend = ["M1", "M2", "M3"]
elif graph == "pad":
    data = pad
    legend = ["P", "A", "D"]

emotions_pad = []
emotions_remap_human = []
data = np.array(data)
for i in range(np.shape(data)[0]):
    emotions_pad.append(cst.padToEmotion(data[i]))
    emotions_remap_human.append(cst.remap_emotion(cst.padToEmotion(data[i])))

human_emotions = np.load("./data/human_emotions_2023-04-24_11-01-54-615176_146.npy")
human_emotions = [cst.EMOTION[i] for i in human_emotions]
print(len(human_emotions))
print(len(emotions_remap_human))
plt.figure()
plt.hist(human_emotions, density=True, bins=30)
plt.figure()
plt.hist(emotions_remap_human, density=True, bins=30)
plt.figure()
plt.hist(emotions_pad, density=True, bins=30)
plt.show()

#print(np.array(input_total))
plt.plot(data)
plt.legend(legend)
# plt.savefig(fileName + "_" + "-".join(legend) + ".png")
plt.show()



#exit()

robot.plot(        
        np.array(input_list),
        backend='pyplot',
        dt=0.001,
        movie=fileName+".gif",
        block=True,
        # color=color,
        # printEach=True
    )

#pyplot.hold()