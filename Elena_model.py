# -*- coding: utf-8 -*-

class EleNa_Model(object):

	def _init_(self):
		self.VALID_MODES = ['minimize', 'maximize']
		self.origin = None
		self.dest = None
		self.overhead = None
		self.mode = None
		self.graph_projection = None
		self.algo = None
		self.bbox = None
		self.key = None

	#setters    
	def set_origin(self, o):
		self.origin = o

	def set_dest(self, d):
		self.dest = d
        
    def set_key(self, key):
		self.key = key