import torch

def loss_function(predict, labels):
	'''
	功能：计算损失值
	参数：
	——predict：net的预测结果(batchsize,5)，其中predict为tensor
	——labels：样本数据(batchsize,5)，其中labels为tensor
	返回值：
	——loss：平均损失
	'''
	# 设置初始损失值
	loss = 0.

	# 获取batchsize
	batch = labels.size()[0]

	# 计算每个批次的总损失
	for batch_size in range(batch):
		if labels[batch_size, 4] == 1:
			loss = loss + torch.sum((predict[batch_size, 0:5] - labels[batch_size, 0:5]) ** 2)
		else:
			loss = loss + 5 * torch.sum((predict[batch_size, 4] - labels[batch_size, 4]) ** 2)

	# 求平均损失
	loss = loss / batch

	return loss