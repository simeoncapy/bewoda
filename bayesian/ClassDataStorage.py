import numpy as np
from enum import Enum, auto

class WeightFonction(Enum):
    EXP = lambda x, a: np.exp(-x/a)
    TAN = lambda x, a: (-np.arctan(a[0] * x + a[1]) + 1.58 + abs(a[2]))/(2*1.58+abs(a[2]))

class ClassDataStorage:
    def __init__(self, className, weightFct, fctParam=None):
        self.data = {el:0 for el in className}
        self.weight = weightFct
        self.fctParam = fctParam

    def __len__(self):
        return sum([d for d in self.data.values()])

    def __str__(self) -> str:
        return str(self.data)

    def add(self, className):
        self.data[className] += 1

    append = add # alias

    def probability(self, weight=True, className=None):
        if weight:
            w = self.weight
            param = self.fctParam
        else:
            w = lambda x : 1
            param = None

        proba = {el:0 for el in self.data.keys()}
        leng = self.__len__()

        for cl, data in self.data.items():
            if className != None and className != cl:
                continue
            if param == None:
                proba[cl] = sum([w(i) for i in range(data)]) / leng
            else:
                proba[cl] = sum([w(i, param) for i in range(data)]) / leng

        if className == None:
            return proba
        else:
            return proba[className]


if __name__ == "__main__":
    temp = ClassDataStorage(["test", "test2", "last"], WeightFonction.TAN, [7, -10000, 0.2])
    temp.add("test")
    temp.add("test")
    temp.add("test")
    temp.add("test")
    temp.add("test")
    temp.add("test")
    temp.add("test")
    temp.add("test")
    temp.add("test2")
    temp.add("test2")
    temp.add("test2")
    temp.add("test2")
    temp.add("test2")
    temp.add("test2")

    print(temp)
    print(temp.probability())
    print(temp.probability(weight=False))