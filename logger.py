from tensorboardX import SummaryWriter
from datetime import datetime
import os
import numpy as np

class Logger(object):

    def __init__(self, log_dir):

        # check if logger directory exist
        if not os.path.exists('logs'):
            os.mkdir('./logs')

        self.log_dir = os.path.join('./logs', '{}'.format(datetime.now().strftime('%b%d_%H%M')), log_dir + '.log')

        # create tensorboardX summary writer
        self.summary_writer = SummaryWriter(log_dir=self.log_dir)
   

    def update_value(self, value_label, value, step):

        self.summary_writer.add_scalar(value_label, value, step)

    def update_values(self, label, values, step):

        self.summary_writer.add_scalars(label, values, step)

    def update_image(self, label, img, step):

        self.summary_writer.add_image(label, img, step) 
    
    def update_RGB_image(self, label, img, step):

        self.summary_writer.add_image(label, img, step,dataformats='HWC') 

    def update_histogram(self, label, matrix, step):

        self.summary_writer.add_histogram(label, matrix, step)
    
    def update_txt(self, label, text, step):

        self.summary_writer.add_text(label, text, step)

    def update_graph(self, model, inputs):

        self.summary_writer.add_graph(model,(inputs,))   

    def scalars_to_json(self, json):
        self.summary_writer.export_scalars_to_json(json)
    def close(self):
        self.summary_writer.close()