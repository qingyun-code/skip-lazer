import cv2
import utils as U

def convert_annotation(image_id):
	'''
	功能：将Annotation中的txt文档中的坐标进行归一化然后
	存入labels中。
	参数：
	——image_id：Annotations中的.txt文档的文件名例如'1.txt'
	'''
	with open('Annotations/%s' % (image_id)) as in_file:

		# 将文件名以'.'隔开取前面数字字符串放入image_id中
		image_id = image_id.split('.')[0]

		# 将.txt文档中的内容以每个整行数值的格式方式列表中
		line = [x.split() for x in in_file]
		pixel = [int(x) for y in line for x in y]

		# 读入图片数据并将pixel内的坐标点放入归一化函数进行归一化
		img = cv2.imread('Image/' + image_id + '.jpg')
		normal = U.normal(pixel[0], pixel[1] - 300, pixel[2],
					pixel[3] - 300, img.shape[1], img.shape[0])

		# 将归一化后的数值放入pixel中
		for i in range(4):
			pixel[i] = normal[i]

		# 将归一化后的pixel内的数值存入labels中的.txt文档内
		with open('labels/%s.txt' % (image_id), 'w') as out_file:
			out_file.write(" ".join([str(a) for a in pixel]) + '\n')

	# 打印出文件名id和归一化后的坐标
	print('image_id={}, pixel={}'.format(image_id, pixel))

if __name__ == '__main__':
	for i in range(1000):
		convert_annotation(str(i + 1) + '.txt')