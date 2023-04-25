# -*-coding:utf-8-*-
from .senet import *



def get_model(config):
    return globals()[config.architecture](config.num_classes)
