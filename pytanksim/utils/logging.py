# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:11:52 2024

@author: nextf
"""

import logging
import sys

logger = logging.getLogger('pytanksim')
logger.setLevel(logging.DEBUG)


if not len(logger.handlers):
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    