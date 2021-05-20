# coding=utf-8
import os
import multiprocessing

_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), ''))
_VAR = os.path.join(_ROOT, 'var')
_ETC = os.path.join(_ROOT, 'etc')

loglevel = 'info'
# errorlog = os.path.join(_VAR, 'log/api_v1-error.log')
# accesslog = os.path.join(_VAR, 'log/api_v1-access.log')
#errorlog = "-"
#accesslog = "-"
#capture_output = True

bind = '0.0.0.0:7777'

workers = 1

timeout = 30  # 30 seconds

