import numpy as np
import random

a = ["neutral"] + [[0,1]]
print(a)

a[1] = [[3,4], tuple(a[1])]
print(a)

traj = []
for pt in a[1]: 
    traj += list(pt)

print(traj)

DYN_MAX = 4095
def dynToRad(val):    
    return val * ((2*np.pi)/DYN_MAX) - np.pi

def radtoDyn(val):    
    return (val + np.pi) * (DYN_MAX/(2*np.pi))

a = [dynToRad(val) for val in [17333.492729413185, 21219.64946234212, 26339.506745407227]]
print(a)

print([radtoDyn(val) for val in a])

init_dict = {}
for i in init_dict.keys():
    print(i)
print("after printing keys")