from torch.utils.tensorboard import SummaryWriter
import torch

class VisdomLogger(object):
    def __init__(self, title, keys, num_epochs_init):
        from visdom import Visdom
        # print("python -m visdom.server")
        try:
            self.viz = Visdom()
        finally:
            print("Visdom server must be started, use: 'python -m visdom.server' in terminal")

        self.opts = dict(title=title, ylabel='', xlabel='Batch', legend=keys)
        self.title = title
        self.viz_window = None
        self.epochs = torch.arange(0, num_epochs_init)
        self.visdom_plotter = True
        self.iteration = 1
        self.keys = keys
        self.values = dict()
        for key in keys:
            self.values[key] = []

    def add_value(self, keys, input_values):
        if type(keys) is list:
            for i, key in enumerate(keys):
                if not self.values[key]:
                    self.values[key] = [input_values[i]]
                else:
                    self.values[key].append(input_values[i])
        else:
            if not self.values[keys]:
                self.values[keys] =  [input_values]
            else:
                self.values[keys].append(input_values)

    def update(self):
        x_axis = torch.arange(0, self.iteration)
        torch_list = [torch.tensor(self.values[key]) for key in self.keys if self.values[key]]
        y_axis = torch.stack(torch_list, dim=1)
        self.viz_window = self.viz.line(
            X=x_axis,
            Y=y_axis,
            opts=self.opts,
            win=self.viz_window,
            update='replace' if self.viz_window else None
        )
        self.iteration += 1

    def save_data_to_file(self, file_path):
        import csv
        with open(file_path, 'w') as f:  # Just use 'w' mode in 3.x
            w = csv.DictWriter(f, self.values.keys())
            w.writeheader()
            w.writerow(self.values)

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
