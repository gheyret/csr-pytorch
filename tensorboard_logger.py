from torch.utils.tensorboard import SummaryWriter
import os
import torch

class VisdomLogger(object):
    def __init__(self, id, num_epochs):
        from visdom import Visdom
        self.viz = Visdom()
        self.opts = dict(title=id, ylabel='', xlabel='Batch', legend=['Loss'])
        self.viz_window = None
        self.epochs = torch.arange(0, num_epochs)
        self.visdom_plotter = True
        self.epoch = 1
        self.losses = []

    def update(self, value):
        self.losses.append(value)
        x_axis = torch.arange(0, self.epoch)
        #x_axis = self.epochs[0:self.epoch]
        y_axis = torch.tensor(self.losses)  # torch.stack(self.losses,dim=1)
        self.viz_window = self.viz.line(
            X=x_axis,
            Y=y_axis,
            opts=self.opts,
            win=self.viz_window,
            update='replace' if self.viz_window else None
        )
        self.epoch += 1


class TensorboardLogger(object):

    def __init__(self):
        self.model_graph = False
        self.writer = SummaryWriter(flush_secs=120)
        self.epochs = 0


    def add_model_graph(self, model_in, args):
        if not self.model_graph:
            model = model_in.cpu()
            self.writer.add_graph(model, args)
            self.model_graph = True
            #self.writer.flush()
            #self.writer.close()
            model.cuda()

    def update_scalar(self, graph_name, value):
        self.writer.add_scalar(graph_name, value, self.epochs)
        self.epochs += 1

    def run_logger(self):
        #os.system('tensorboard --logdir=' + "runs")
        return None