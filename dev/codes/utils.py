# authors_name = 'Preetham Ganesh'
# project_title = 'Digit Recognizer'
# email = 'preetham.ganesh2021@gmail.com'


import os
import logging
import warnings


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
warnings.filterwarnings("ignore")


import json
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import time

from model import DigitRecognition


