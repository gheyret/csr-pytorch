from torch.utils.tensorboard import SummaryWriter
import os
import torch
import sys

class VisdomLogger(object):
    def __init__(self, id, num_epochs):
        from visdom import Visdom
        # print("python -m visdom.server")
        try:
            self.viz = Visdom()
        finally:
            print("Visdom server must be started, use: 'python -m visdom.server' in terminal")


        self.opts = dict(title=id, ylabel='', xlabel='Batch', legend=['Loss', 'per'])
        self.viz_window = None
        self.epochs = torch.arange(0, num_epochs)
        self.visdom_plotter = True
        self.epoch = 1
        self.values = dict()
        self.values["loss"] = []
        self.values["per"] = []
        self.losses = []

    def update(self, value_loss, value_per):
        self.values["loss"].append(value_loss)
        self.values["per"].append(value_per)
        # self.losses.append(value)
        x_axis = torch.arange(0, self.epoch)
        # x_axis = self.epochs[0:self.epoch]
        y_axis = torch.stack((torch.tensor(self.values["loss"]),
                              torch.tensor(self.values["per"])), dim=1)
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
            # self.writer.flush()
            # self.writer.close()
            model.cuda()

    def update_scalar(self, graph_name, value):
        self.writer.add_scalar(graph_name, value, self.epochs)
        self.epochs += 1

    def run_logger(self):
        # os.system('tensorboard --logdir=' + "runs")
        return None
