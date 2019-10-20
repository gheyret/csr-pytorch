from torch.utils.tensorboard import SummaryWriter
import os

class TensorboardLogger(object):

    def __init__(self):
        self.model_graph = False
        self.writer = SummaryWriter(flush_secs=120)
        self.epochs = 0


    def add_model_graph(self, model_in, input_batch):
        if not self.model_graph:
            model = model_in.cpu()
            self.writer.add_graph(model, input_batch)
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