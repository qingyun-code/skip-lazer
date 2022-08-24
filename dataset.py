import cv2
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class dataset(Dataset):
	def __init__(self, is_train = True):
		self.file_names = []# 设置存储着文件名的列表

		# 如果是要进行训练读取训练数据文档，否则读取验证数据文档
		if is_train:
			with open("ImageSets/Main/train.txt", 'r') as f:# 打开训练文档
				self.file_names = [x.strip() for x in f]# 将训练数据的文档名存储到文件列表中,.strip()去除字符串前后的空格或换行符
		else:
			with open("ImageSets/Main/val.txt", 'r') as f:# 打开验证数据文档
				self.file_names = [x.strip() for x in f]# 将验证数据的文档名存储到文件列表中

		self.img_path = "Image/"# 图片存储路径
		self.label_path = "labels/"# label数据文档存储数据

	def __len__(self):
		'''
		功能：返回文件名的个数
		'''
		return len(self.file_names)# 返回文件名的个数

	def __getitem__(self, item):
		'''
		功能：对图像数据进行规范化处理
		参数：
		——picture_index：图片在文件名当中的索引
		'''
		img = cv2.imread(self.img_path + self.file_names[item] + '.jpg')# 读取需要图像处理和数据转换的图像

		img = img.transpose(2, 1, 0)

		img = img / 255.

		with open(self.label_path+self.file_names[item] + ".txt") as f:

			# 将.txt文档中的内容放入列表中
			line = [x.split() for x in f]
			label = [float(x) for y in line for x in y]

		#label = transforms.ToTensor()(label)
		labels = np.zeros((5))

		labels[0:5] = np.array([label[0], label[1], label[2], label[3], label[4]])

		return img, labels

if __name__ == '__main__':
	voc = dataset()
	# voc.image_process(0)
	img, labels = voc.__getitem__(20)
	print(labels)
	# print(img)
	# print(img.shape)
	# dr.show_labels_img2(img, bbox)
	#print('img={}\nlabels={}'.format(img, labels))
	# print(voc.__len__())