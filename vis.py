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
        interval = float(10000) / float(len(self.recording))
        an = animation.FuncAnimation(fig, self._animate, interval=interval, repeat=False)
        plt.show(block=False)
        plt.pause(interval * (len(self.recording) + 1) / 1000)
        plt.close()




    def show_trajs(self, trajs, x_f, title, direc='/Users/JonathanLee/Desktop/'):
        self.x_f = np.reshape(x_f, (x_f.size))
        self.x_f = np.array([x_f[0], x_f[1]])
        
        size=500

        fig, figure = plt.subplots()
        figure.autoscale(False)

        for traj in trajs:
            xar = []
            yar = []
            for state in traj:
                state = np.reshape(state, (state.size))
                state = np.array([state[0], state[1]])
                xar.append(state[0])
                yar.append(state[1])

            robo_size=500
            indicator_size = 500        
            figure.scatter(xar, yar, s=robo_size, alpha=.3)
            figure.set_xlim(-20, 20)
            figure.set_ylim(-20, 20)
            
        xs = [x_f[0]]
        ys = [x_f[1]]
        figure.scatter(xs, ys, s=size, c='g')        

        plt.title(title)
        plt.savefig(direc + title + '.png')
        plt.clf()
        #plt.show()
        #plt.close()
    

    def compute_std_er_m(self, data):
        n = data.shape[0]
        std = np.std(data)
        return std/np.sqrt(n)

    def _compute_m(self, data):
        n = data.shape[0]
        return np.sum(data) / n


    def get_perf(self, data, color=None):
        
        iters = data.shape[1]
        mean = np.zeros(iters)
        err = np.zeros(iters)
        x = np.zeros(iters)
        self.iters = iters

        for i in range(iters):
            mean[i] = self._compute_m(data[:, i])
            x[i] = i
            err[i] = self.compute_std_er_m(data[:, i])

        if color is None:
            plt.errorbar(x, mean, yerr=err, linewidth=5.0)
        else:
            plt.errorbar(x, mean, yerr=err, linewidth=5.0, color=color)

        self.mean = mean
        self.err = err
        self.x = x
        return [mean, err]

    def plot(self, names, label, filename=None, ylims=None):
        plt.ylabel(label)
        plt.xlabel('Iterations')

        plt.legend(names, loc='upper center', prop={'size':10}, bbox_to_anchor=(.5, 1.12), fancybox=False, ncol=len(names))

        font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 22},

        axes = plt.gca()
        axes = plt.gca()
        axes.set_xlim([0, self.iters])
        if not ylims is None:
            axes.set_ylim(ylims)
        axes.set_yscale("log")

        if filename is not None:
            plt.savefig(filename, format='eps', dpi=1000)
        plt.clf()













