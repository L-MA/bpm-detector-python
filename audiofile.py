from scipy.io import wavfile
import numpy as np
import math
import logging

class AudioFile:
	def __init__(self):
		self.data = np.array([])
		self.fs = None

	def load(self, audio_file):
		""" Read in wav audio file and set 
		class attributes from gathered info
		"""
		logger.debug("Loading audio file...")
		self.fs, self.data = wavfile.read(audio_file)
		logger.debug("Succesfully loaded audio file.")

	def load_from_array(self, data, fs):
		self.data = data
		self.fs = fs

	def get_duration_in_seconds(self):
		if self.data.size > 0:
			return float(self.data.shape[0]/self.fs)
		else:
			raise ValueError('Audio data not loaded.')

	def get_duration_in_string(self):
		minutes = math.floor(self.get_time_in_seconds()/60.0)
		seconds = self.get_time_in_seconds()%60
		return "%i:%i" % (minutes, seconds)

	def num_channels(self):
		if self.data.size > 0:
			return self.data.shape[1]
		else:
			raise ValueError('Audio data not loaded.')

	def num_samples(self):
		if self.data.size > 0:
			return self.data.shape[0]
		else:
			raise ValueError('Audio data not loaded.')

	def bit_depth(self):
		if self.data.size > 0:
			if self.data.dtype == np.dtype(np.float32):
				return 32
			elif self.data.dtype == np.dtype(np.int32):
				return 32
			elif self.data.dtype == np.dtype(np.int16):
				return 16
			elif self.data.dtype == np.dtype(np.uint8):
				return 8
			else:
				raise ValueError('Unknown bit depth')
		else:
			raise ValueError('Audio data not loaded.')

	def max_sample_value(self):
		if self.data.size > 0:
			if self.data.dtype == np.dtype(np.float32):
				return 1.0
			elif self.data.dtype == np.dtype(np.int32):
				return 2147483647
			elif self.data.dtype == np.dtype(np.int16):
				return 32767
			elif self.data.dtype == np.dtype(np.uint8):
				return 255
			else:
				raise ValueError('Unknown bit depth')
		else:
			raise ValueError('Audio data not loaded.')

	def min_sample_value(self):
		if self.data.size > 0:
			if self.data.dtype == np.dtype(float32):
				return -1.0
			elif self.data.dtype == np.dtype(int32):
				return -2147483648
			elif self.data.dtype == np.dtype(int16):
				return -32768
			elif self.data.dtype == np.dtype(uint8):
				return 0
			else:
				raise ValueError('Unknown bit depth')
		else:
			raise ValueError('Audio data not loaded.')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stdout_hdlr = logging.StreamHandler()
stdout_hdlr.setLevel(logging.DEBUG)
logger.addHandler(stdout_hdlr)