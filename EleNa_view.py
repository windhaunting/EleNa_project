# -*- coding: utf-8 -*-


class EleNa_view(object):
     def __init__(self, key):
        self.key = key

     def get_input_from_user(self):
            #get input data from user
    		VALID_MODES = ['minimize', 'maximize']
