#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf8
import os
import copy
import numpy as np

import utils.tool
from utils.configure import Configure


class EvaluateTool(object):
    """
    The no evaluation
    """
    def __init__(self, meta_args):
        self.meta_args = meta_args

    def evaluate(self, preds, golds, section):
        return {}
