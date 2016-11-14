from audiofile import AudioFile 
import numpy as np
from scipy import signal
#import matplotlib.pyplot as plt
from operations import *
import logging

class Detector:
	def __init__(self):
		self.raw_data = None
		self.ch_data = {}
		self.num_channels = 0
		self.bandlimits = [200, 400, 800, 1600, 3200]
		self.N = 0
		self.fs = 0
		self._sample_index = 0
		self.bpm_test_range = range(60, 180)

	def load_audio(self, filename):
		"""Load wav audio data from filename into class

		Reads audio file into the raw_data attribute. 

		The raw_data persists and remains unmutated for the lifecycle
		of the class.

		The raw_data attribute contains an AudioFile() object, so we are
		able to easily access attributes such as sampling rate, bit depth, etc.
		"""
		logger.info("Loading audio file.")
		audio = AudioFile()
		audio.load(filename)
		self.raw_data = audio
		self.N = self.raw_data.num_samples()
		self.fs = self.raw_data.fs
		self.num_channels = self.raw_data.num_channels()

	def split_channels(self):
		"""Split raw_data into separate data streams in ch_data

		Each channel in raw_data is split into its own numpy array
		in the dictionary object ch_data.

		Use int index from 0 to number of channels to access audio data
		in ch_data dictionary.
		"""
		# Determine number of channels in raw_audio file
		num_channels = self.raw_data.num_channels()
		logger.info("Splitting audio into %i channels." % num_channels)

		
		# Split the data into a list of arrays using numpy's split function
		new_data = np.split(self.raw_data.data, num_channels, 1)

		# Set the split data in the ch_data dictionary
		for i in range(0, num_channels):
			self.ch_data[i] = [new_data[i].flatten()]

		logger.info("Successfully split audio.")

	def window(self, seconds, start_location = -1):
		if start_location == -1:
			start_location = self._sample_index

		try:
			self.ch_data[0]
		except KeyError as e:
			log.error("Cannot window. Raw data has a not been split into channels yet.")
			raise e

		num_samples = seconds * self.raw_data.fs
		end_sample = start_location + num_samples
		logger.info("Windowing data to %i second segment starting at sample %i and " \
			"ending at sample %i" % (seconds, start_location, end_sample))

		for i in range(0, self.num_channels):
			for j in range(0, len(self.ch_data[i])):
				self.ch_data[i][j] = self.ch_data[i][j][start_location:end_sample]
		
		self._sample_index = end_sample
		self.N = num_samples

		logger.info("Successfully windowed data.")

	def normalize(self):
		logger.info("Normalizing data.")
		bit_depth = self.raw_data.bit_depth()
		for i in range(0, self.num_channels):
			for j in range(0, len(self.ch_data[i])):
				self.ch_data[i][j] = np.array(self.ch_data[i][j], dtype=np.float32)
				self.ch_data[i][j] = self.ch_data[i][j]/(2**bit_depth - 1)
		logger.info("Sucessfully normalized data.")

	def filterbank(self):
		"""Filter ch_data based on bandlimits attribute

		This method uses butterworth filters to filter the ch_data, audio data, into 
		band limited audio data. The number of bands and the limits are defined in the 
		bandlimits attribute. 

		First run split_channels() method to get the audio data split into left and
		right channels and set into the ch_data attribute.
		"""
		num_bands = len(self.bandlimits)
		logger.info("Breaking signal into %i bandlimits" % num_bands)

		# Each channel will be broken down into 6 bandpassed signals.
		# First duplicate data for each channel into six signals.
		logger.debug("Duplicating data before being filtered")
		for i in range(0, self.num_channels):
			duplicate_data = np.copy(self.ch_data[i][0])
			for j in range(0, num_bands):
				self.ch_data[i].append(duplicate_data)

		# Then filter each channel based on the bandlimits attribute
		logger.debug("Filtering data.")
		for i in range(0, self.num_channels):
			self.ch_data[i][0] = butter_lowpass_filter(self.ch_data[i][0], self.bandlimits[0], self.fs)
			for j in range(0, num_bands-1):
				self.ch_data[i][j+1] = butter_bandpass_filter(self.ch_data[i][j+1], self.bandlimits[j], self.bandlimits[j+1], self.fs)
			self.ch_data[i][num_bands] = butter_highpass_filter(self.ch_data[i][num_bands], self.bandlimits[num_bands-1], self.fs)

		logger.info("Successfully bandlimited data.")

	def full_wave_rectify(self):
		logger.info("Full wave rectifying all signal data")
		for i in range(0, self.num_channels):
			for j in range(0, len(self.ch_data[i])):
				self.ch_data[i][j] = np.abs(self.ch_data[i][j])
		logger.info("Successfully full wave rectified all signal data")

	def half_wave_rectify(self):
		logger.info("Half wave rectifying all signal data")
		for i in range(0, self.num_channels):
			for j in range(0, len(self.ch_data[i])):
				for k in range(0, len(self.ch_data[i][j])):
					if self.ch_data[i][j][k] < 0:
						self.ch_data[i][j][k] = 0
		logger.info("Successfully half wave rectified all signal data")

	def differentiate(self):
		logger.info("Differentiating all signal data.")
		output = np.zeros(self.N, dtype=np.float32)
		for i in range(0, self.num_channels):
			for j in range(0, len(self.ch_data[i])):
				for k in range(1, self.N):
					output[k] = self.ch_data[i][j][k] - self.ch_data[i][j][k-1]
				self.ch_data[i][j] = output
		logger.info("Successfully differentiated all signal data.")

	def smooth(self):
		logger.info("Smoothing signal data.")
		for i in range(0, self.num_channels):
			for j in range(0, len(self.ch_data[i])):
				self.ch_data[i][j] = smooth(self.ch_data[i][j])
		logger.info("Successfully smoothed signal data.")

	def comb_filter_convolution(self):
		# Define len of impulse train / comb filter to be used later
		impulse_train_len = self.N

		# Initialize energy buffer based on the bpm test range, defined in init
		test_range_len = len(self.bpm_test_range)
		energy_buffer = np.zeros(test_range_len, dtype=np.float32)
		logger.debug("Created energy buffer.")

		# For bpm in bpm_range
		for i, bpm in enumerate(self.bpm_test_range):

			# Create impulse train array of impulse_train_len
			impulse_train = np.zeros(impulse_train_len)

			# Determine periodicity of impulses
			periodicity = np.floor(120.0/bpm * self.fs)

			# Set impulses 
			for j in range(0, impulse_train_len):
				if j%periodicity == 0:
					impulse_train[j] = 1
			logger.debug("Created impulse train for bpm %i" % bpm)

			# For each channel
			for j in range(0, self.num_channels):

				# For each signal in the channel
				for k in range(0, len(self.bandlimits)):

					# Convolve impulse train with signal
					conv_result = signal.convolve(self.ch_data[j][k], impulse_train)
					logger.debug("Convoled impulse trian with signal %i in channel %i" % (k, j))

					# Sum total energy in convolved signal
					energy = np.sum(conv_result)
					logger.debug("Summed result to get %f" % energy)

					# Add to total energy range under bpm index
					energy_buffer[i] += energy
					logger.debug("Added result to energy buffer to get %f" % energy_buffer[i])

		# Determine winning bpm based on greatest value in the energy buffer
		winning_bpm = np.argmax(energy_buffer) + self.bpm_test_range[0]
		logger.info("Determined winning bpm to be %i" % winning_bpm)

		return winning_bpm


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stdout_hdlr = logging.StreamHandler()
stdout_hdlr.setLevel(logging.DEBUG)
logger.addHandler(stdout_hdlr)

if __name__ == "__main__":
	try:
		dt = Detector()
		dt.load_audio("fire.wav")
		dt.split_channels()
		dt.window(2)
		dt.filterbank()
		# dt.normalize()
		dt.full_wave_rectify()
		dt.smooth()
		dt.differentiate()
		dt.half_wave_rectify()
		bpm = dt.comb_filter_convolution()

		print("winning bpm is %i" % bpm)
		
	except Exception as e:
		logger.error("ERROR: %s" % e.message)
		exit(1)
		

