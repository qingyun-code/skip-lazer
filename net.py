import torch
import torch.nn as nn

class net(nn.Module):
	def __init__(self):
		super(net, self).__init__()

		self.Conv_layers1 = nn.Sequential(
				nn.Conv2d(3, 128, 10, 2),
				nn.LeakyReLU(),
				nn.MaxPool2d(2, 2)
			)

		self.Conv_layers2 = nn.Sequential(
				nn.Conv2d(128, 256, 6, 2),
				nn.LeakyReLU(),
				nn.MaxPool2d(2, 2)
			)

		self.Conv_layers3 = nn.Sequential(
				nn.Conv2d(256, 512, 3, 2),
				nn.LeakyReLU(),
				nn.MaxPool2d(2, 2)
			)

		self.Conn_layers1 = nn.Sequential(
				nn.Linear(15 * 15 * 512, 2000),
				nn.LeakyReLU()
			)

		self.Conn_layers2 = nn.Sequential(
				nn.Linear(2000, 5),
				nn.Sigmoid()
			)

	def forward(self, input):
		input = self.Conv_layers1(input)
		input = self.Conv_layers2(input)
		input = self.Conv_layers3(input)
		input = input.view(input.size()[0], -1)
		input = self.Conn_layers1(input)
		input = self.Conn_layers2(input)
		output = input.reshape(-1, 5)

		return output

if __name__ == '__main__':
	x = torch.randn((1, 3, 1000, 1000))
	net = net()
	output = net(x)
	print(output.size())