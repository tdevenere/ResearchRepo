import numpy as np

class point:
    def __init__(self, x, y, time=0):
        self.x = x
        self.y = y
        self.time = time

    @classmethod
    def fromlist(cls, arr):
        return cls(arr[0], arr[1], arr[2])


    def euclidianDistance(self, p2):
        return np.math.sqrt((self.y - p2.y) ** 2 + (self.x - p2.x) ** 2)

    # Calculate the angle between 3 points using law of cosines
    def angle(self, p2, p3):
        v1 = p2.sub(self)
        v2 = p3.sub(self)
        d1 = self.euclidianDistance(p2)
        d2 = self.euclidianDistance(p3)
        if d1 == 0 or d2 == 0:
            return 0

        # Dot Product of two vectors
        num = (v1.x * v2.x + v1.y * v2.y)

        # Magnitude of two vectors
        denom = (d1 * d2)

        # Theta = arccos ( a dot b / len(a) * len(b))
        return np.math.acos(max(min(num / denom, 1), -1))

    # Return the difference between two points in 2D space
    # also includes the time difference.
    def sub(self, p2):
        return point(p2.x - self.x, p2.y - self.y, p2.time - self.time)

    # Return the vector-addition of the two points
    # Time is not necessary for this function
    def add(self, p2):
        return point(self.x + p2.x, self.y + p2.y)

    def lerp(self, p2, amount, time=False):
        if time:
            timeamount = (amount - self.time) / (p2.time - self.time)
            x3 = self.x + timeamount * (p2.x - self.x)
            y3 = self.y + timeamount * (p2.y - self.y)
            return point(x3, y3, amount)
        else:
            x3 = self.x + amount * (p2.x - self.x)
            y3 = self.y + amount * (p2.y - self.y)
            t3 = self.time + amount * (p2.time - self.time)
            return point(x3, y3, t3)

    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ", " + str(self.time) + ")"