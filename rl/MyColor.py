class MyColor:
    def __init__ (self, red, green, blue):
        self.red = red
        self.green = green
        self.blue = blue

    def __mul__ (self, coef):
        copy = MyColor(round(coef * self.red), round(coef * self.green), round(coef * self.blue))
        return copy

    def __rmul__ (self, coef):
        return self.__mul__(coef)

    def toNum(self, white = 0):
        return (white << 24) | (self.red << 16)| (self.green << 8) | self.blue

    def toList(self):
        return [self.red, self.green, self.blue]
    
    def toTuple(self):
        return (self.red, self.green, self.blue)

    def __str__(self):
        return "RGB(" + str(self.red) + ", " + str(self.green) + ", " + str(self.blue) + ")"
    def __unicode__(self):
        return u"RGB(" + str(self.red) + ", " + str(self.green) + ", " + str(self.blue) + ")"