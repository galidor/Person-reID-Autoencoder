# -*- coding: utf-8 -*-

import sys
from os import popen

# Print iterations progress
def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    rows, columns = popen('stty size', 'r').read().split()
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    full_length = len(percents) + len(prefix) + len(suffix)
    if bar_length > (int(columns) - full_length):
        bar_length = int(columns) - full_length - 10
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    if bar_length < 10 and full_length < int(columns):
        sys.stdout.write('\r{} {}{} {}'.format(prefix, percents, '%', suffix))
    elif full_length > int(columns):
        sys.stdout.write('\r{} {}{} {}'.format(prefix, percents, '%', suffix)[:int(columns)])
    else:
        sys.stdout.write('\r{} |{}| {}{} {}'.format(prefix, bar, percents, '%', suffix))

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
