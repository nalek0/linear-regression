from random import random

import numpy
from PIL import Image, ImageDraw


def sigmoid(x):
	return 1 / (1 + numpy.exp(-x))


def pr_sigmoid(x):
	fx = sigmoid(x)
	return fx * (1 - fx)


class Net:
	def __init__(self, number_of_neyrons, number_of_inputs):
		self.n = number_of_neyrons
		self.i = number_of_inputs
		self.w = list(map(lambda _: random(), [0.0] * (self.n * self.i + self.n)))
		self.b = list(map(lambda _: random(), [0.0] * (self.n + 1)))

	def get_answer(self, inp):
		h = []
		for i in range(self.n):
			sum_h = 0
			for j in range(self.i):
				sum_h += inp[j] * self.w[i * self.i + j]
			sum_h += self.b[i]
			h.append(sigmoid(sum_h))
		sum_o = 0
		for i in range(self.n):
			sum_o += h[i] * self.w[self.i * self.n + i]
		sum_o += self.b[self.n]
		return sigmoid(sum_o)

	def get_errors(self, data_in, data_out):
		error = 0
		for i in range(len(data_in)):
			error += (data_out[i] - self.get_answer(data_in[i])) ** 2
		return error / len(data_in)

	def train(self, inputs, outputs, interactions, step):
		for it in range(interactions):
			for jj in range(len(inputs)):
				current_input = inputs[jj]
				h = []
				sum_h = []
				for i in range(self.n):
					sum_h_now = self.b[i]
					for j in range(self.i):
						sum_h_now += current_input[j] * self.w[i * self.i + j]
					sum_h.append(sum_h_now)
					h.append(sigmoid(sum_h_now))
				sum_o = self.b[self.n]
				for i in range(self.n):
					sum_o += h[i] * self.w[self.i * self.n + i]
				y_predp = sigmoid(sum_o)
				y_real = outputs[jj]

				pr_dL_dy = -2 * (y_real - y_predp)
				
				pr_dy_dw = [0] * len(self.w)
				for i in range(self.n):
					pr_dy_dw[len(self.w) - i - 1] = sum_h[self.n - i - 1] * pr_sigmoid(sum_o)
				pr_dy_dh = [0] * self.n
				for i in range(self.n):
					pr_dy_dh[i] = self.w[self.n * self.i + i] * pr_sigmoid(sum_o)
				pr_dy_db = pr_sigmoid(sum_o)

				pr_dh_dw = [[0] * self.i] * self.n
				for i in range(self.n):
					for j in range(self.i):
						pr_dh_dw[i][j] = current_input[j] * pr_sigmoid(sum_h[i])
				pr_dh_db = [0] * self.n
				for i in range(self.n):
					pr_dh_db[i] = pr_sigmoid(sum_h[i])

				for i in range(self.n):
					for j in range(self.i):
						self.w[i * self.i + j] -= step * pr_dL_dy * pr_dy_dh[i] * pr_dh_dw[i][j]
				for i in range(self.n):
					self.w[self.n * self.i + i] -= step * pr_dL_dy * pr_dy_dw[self.n * self.i + i]
				for i in range(self.n):
					self.b[i] -= step * pr_dL_dy * pr_dy_dh[i] * pr_dh_db[i]
				self.b[self.n] -= step * pr_dL_dy * pr_dy_db
			print(f"Interaction: {str(it).rjust(4, '0')}, Error: {self.get_errors(inputs, outputs)}")


if __name__ == "__main__":
	img = Image.open('points.png')
	width = img.size[0]
	height = img.size[1]
	pix = img.load()
	data_inputs = []
	data_outputs = []
	for y in range(height):
		for x in range(width):
			a, b, c = pix[x, y][:3]
			if a == 255 and b == 0 and c == 0:
				data_inputs.append([x / width, y / height])
				data_outputs.append(0.0)
			elif a == 0 and b == 255 and c == 0:
				data_inputs.append([x / width, y / height])
				data_outputs.append(1.0)
	net = Net(1, 2)
	print('Start!')
	net.train(data_inputs, data_outputs, 1000, 0.25)
	print('Done!')
	image = Image.open('points2.png')
	draw = ImageDraw.Draw(image)
	for x in range(width):
		for y in range(height):
			ans = net.get_answer([x / width, y / height])
			ans = int(ans * 255)
			draw.point((x, y), (ans, ans, ans))

	for y in range(height):
		for x in range(width):
			a, b, c = pix[x, y][:3]
			if a == 255 and b == 0 and c == 0:
				draw.point((x, y), (a, b, c))
			elif a == 0 and b == 255 and c == 0:
				draw.point((x, y), (a, b, c))
	image.save('points2.png', 'PNG')
	del draw
