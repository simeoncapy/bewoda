from numpy import ones,vstack
from numpy.linalg import lstsq


def trajectoryPrediction(t1, t2):
    points = [t1,t2]

    x_coords, y_coords = zip(*points)
    A = vstack([x_coords,ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords)[0]
    #print("Line Solution is y = {m}x + {c}".format(m=m,c=c))

    dt = t2[0] - t1[0]
    return (int(t2[0]+dt), int(m * (t2[0]+dt) + c))

