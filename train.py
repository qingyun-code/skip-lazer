import torch
import dataset as D
import net as N
import loss_function as LF
# import predict as P
from torch.utils.data import Dataset, DataLoader

epoch = 10# 设置迭代次数
batch_size = 10# 设置批量的大小
lr = 0.01# 设置学习率

train_data = D.dataset(is_train = True)
data = DataLoader(train_data, batch_size = batch_size, shuffle = True)# 加载数据
# for i, (inputs, labels) in enumerate(data):
# 	print(i)
# 	print(inputs.shape)
# 	print(labels)

net = N.net()# 初始化一个网络对象

# 随机梯度下降
# optimizer = torch.optim.SGD(net.parameters(), lr = lr)
optimizer = torch.optim.SGD(net.parameters(), lr = lr, momentum = 0.9, weight_decay = 0.0005)

for e in range(epoch):
	# net.train()# *************************test
	for i, (inputs, labels) in enumerate(data):
		# inputs = torch.from_numpy(inputs)
		# labels = torch.from_numpy(labels)
		inputs = inputs.float()
		#print(inputs.size())
		#print(inputs)
		predict = net(inputs)
		# labels = labels.float()
		loss = LF.loss_function(predict, labels)
		print("epoch={},i={},loss={}".format(e, i, str(loss)))
		if e == 9:
			if loss > 0.005:
				print("labels={}".format(str(labels)))
		optimizer.zero_grad()
		# print('aaaaaaaaaaaaaaa')
		loss.backward()
		# print('bbbbbbbbbbbbbbb')
		optimizer.step()
		# print('mmmmmmmmmmmmmmm')

	if (e + 1) % 2 == 0:# ***************************test
		torch.save(net, "weight1/weight" + str(e + 1) + ".pkl")# ************test

# # P.predict('80', 'Image/', net)
# # 	print(loss)