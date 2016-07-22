import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
class Visualizer():

    def __init__(self):
        self.recording = []
        self.target = None

    def set_target(self, x_f):
        self.x_f = np.reshape(x_f, (x_f.size))
        self.x_f = np.array([x_f[0], x_f[1]])
        

    def set_recording(self, recording):
        self.recording = recording
        return 

    def _animate(self, i):
        self.figure.autoscale(False)
        if i < len(self.recording):
            xar = [self.recording[i][0]]
            yar = [self.recording[i][1]]
            robo_size = 500
            indicator_size = 500
            self.figure.clear()
            self._draw_target(indicator_size)
            self.figure.scatter(xar, yar, s=robo_size)
            self.figure.set_xlim(-20, 20)
            self.figure.set_ylim(-20, 20)
            plt.title("Step " + str(i))
             
    def _draw_target(self, size):
        xs = [self.x_f[0]]
        ys = [self.x_f[1]]
        self.figure.scatter(xs, ys, s=size, c='g')

    def process_states(self):
        recording2 = []
        for state in self.recording:
            state = np.reshape(state, (state.size))
            state = np.array([state[0], state[1]])
            recording2.append(state)
        self.recording = recording2
            
    def show(self):
        self.process_states()
        fig, self.figure = plt.subplots()
        print len(self.recording)
        interval = float(10000) / float(len(self.recording))
        an = animation.FuncAnimation(fig, self._animate, interval=interval, repeat=False)
        plt.show(block=False)
        plt.pause(interval * (len(self.recording) + 1) / 1000)
        plt.close()
