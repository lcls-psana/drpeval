import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import argparse

WIDTH = 388
HEIGHT = 185

def visualize_image(data_file, peak_file, image_idx):

	if data_file != "":
		data = np.fromfile(data_file, np.float32)
		data = np.reshape(data,(data.shape[0]/WIDTH/HEIGHT, HEIGHT, WIDTH))
		data = data[image_idx,:,:] / data[image_idx,:,:].max()
		plt.imshow(data)
		plt.savefig('data{0}.png'.format(image_idx))

	if peak_file != "":
		peak = np.fromfile(peak_file, np.float32)
		print('num of images:{0}'.format(peak.shape[0]/WIDTH/HEIGHT))
		peak = np.reshape(peak,(peak.shape[0]/WIDTH/HEIGHT, HEIGHT, WIDTH))
		peak = peak[image_idx,:,:]
		plt.imshow(peak)
		plt.savefig('peak{0}.png'.format(image_idx))
		print('save peak{0}.png'.format(image_idx))


parser = argparse.ArgumentParser()
parser.add_argument('-image', type=int, default=9)
parser.add_argument('-regent', action='store_true')
args = parser.parse_args()
if args.regent:
	visualize_image("", "peaks.regent.img", args.image)
else:
	visualize_image("", "peaks.img", args.image)

