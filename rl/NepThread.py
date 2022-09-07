import nep
import nep_local_lan as nep_2
import constantes as cst
import time
import random
import string
import numpy as np
import threading

class NepThread():
    def __init__(self, topic=cst.NEP_TOPIC, ip="127.0.0.1", nodeName=False):
        rng = np.random.RandomState(int(time.time())) # to have random seed each time, without interfering with the rest of the code
        if nodeName == False:
            nodeName = ''.join(rng.choice(list(string.ascii_letters), size=10))
        self.node = nep.node(nodeName)
        self.master = self.node.new_sub(topic, 'json', nep_2.createConf(self.node, ip))
        self.ip = ip

        #init the data
        self.humanEmotionEstimate = cst.EMOTION.index(cst.EMOTION_NEUTRAL)
        self.robotPad = [0, 0, 0]
        self.humanPoseEstimate = [int(cst.CAMERA_X_SIZE/2), int(cst.CAMERA_Y_SIZE/2), int(cst.CAMERA_X_SIZE/2), int(cst.CAMERA_Y_SIZE/2)]

    def listen(self):
        while True:
            s, msg = self.master.listen()  # Read message

            if s:  # If there is a message
                self.humanEmotionEstimate = msg.get(cst.NEP_KEY_EMOTION_ESTIMATE, self.humanEmotionEstimate)
                self.robotPad = msg.get(cst.NEP_KEY_ROBOT_PAD, self.robotPad)
                self.humanPoseEstimate = msg.get(cst.NEP_KEY_BODY_ESTIMATE, self.humanPoseEstimate)
            time.sleep(0.1)

    def readData(self):
        return [self.humanEmotionEstimate, self.robotPad, self.humanPoseEstimate] # [self.humanEmotionEstimate] + self.robotPad + self.humanPoseEstimate


    def run(self):
        t1 = threading.Thread(target=self.listen)
        t1.start()