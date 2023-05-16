import random
import math
import numpy as np

def kl_divergence(p, q):
    return sum(p[i] * math.log(p[i] / q[i]) for i in range(len(p)))

print(kl_divergence([1e-06,1e-05,0.5, 0.5], [0.25,0.25,0.25, 0.25]))
print(kl_divergence([0.25,0.25,0.25, 0.25], [1e-06,0.1,0.1, 0.8]))
a = [0.4200626959247649, 0.3072100313479624, 1e-06, 0.2727272727272727]
b = [0.30407523510971785, 0.30094043887147337, 0.21630094043887146, 0.1786833855799373]
c = a = [0.7000626959247649, 0.2072100313479624, 0.02, 0.1727272727272727]
print(kl_divergence(b, a))
print(kl_divergence(c, b))

d = [0.4200626959247649, 0.3072100313479624, 1e-06, 0.2727272727272727]
e = [0.30407523510971785, 0.30094043887147337, 0.21630094043887146, 0.1786833855799373]
print(kl_divergence(d, e))

d = [1e-6, 1e-6, 0.99, 1e-6]
e = [1e-6, 1e-6, 0.99, 1e-6]
print(kl_divergence(d, e))

d = [0.1, 1e-6, 0.9, 1e-6]
e = [1e-6, 0.1, 0.9, 0.1]
print(kl_divergence(d, e))

a = [[0,0,1,0],[0,9,8,7],[0,6,5,4],[0,3,2,1]]
print(np.shape(a))
print(np.array(a))
print(np.mean(np.array(a),axis=0))