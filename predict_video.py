# 利用训练好的权重导入到网络中并测试视屏效果
# 而且能够在视屏中实时用绿线表示出显示后的效果
import cv2
import torch
import numpy as np
import net as N

def predict_video(video_path, result_video_path):
	batch_size = 1

	# 初始化文件名
	video = "laser.mp4"
	result_video = "result_laser.mp4"

	# 加载权重
	net = torch.load("weight/weight10.pkl")

	# 读取视屏
	cap = cv2.VideoCapture(video_path + video)

	# 获取视屏帧率
	fps_video = cap.get(cv2.CAP_PROP_FPS)
	print("fps_video={}".format(fps_video))

	# 设置写入视屏的编码格式
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	print("fourcc={}".format(fourcc))

	# 获取视屏宽度和高度
	frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	print("frame_width={},frame_height{}".format(frame_width, frame_height))

	# 设置写视屏的对象
	videoWriter = cv2.VideoWriter(result_video_path + result_video, fourcc, fps_video, (frame_width, frame_height))

	while (cap.isOpened()):
		ret, frame = cap.read()

		# 如果视屏没有结束
		if ret == True:
			# 将图片进行剪切
			tailor_frame = frame[300:1300, 0:1000]

			# 初始化输入网络的numpy数组
			inputs = np.zeros((1, 3, 1000, 1000))

			# 将每一帧的图片像素值赋值给输入矩阵
			for wide in range(1000):
				for high in range(1000):
					for channel in range(3):
						inputs[0, channel, wide, high] = tailor_frame[high, wide, channel] / 255.0

			# 进行输入数据类型转换
			inputs = torch.from_numpy(inputs)
			inputs = inputs.float()

			# 得到预测数据
			predict = net(inputs)

			# 如果为箕斗上来的图片时对图片进行处理
			if float(predict[0, 4]) > 0.6:
				# 初始化一个tensor向量
				predict_conversion = torch.zeros((5))

				# 将预测值中的坐标赋值给tensor向量
				for i in range(4):
					predict_conversion[i] = predict[0, i]

				# 将预测值进行坐标换算，横坐标不变纵坐标加三百
				predict_conversion[0] = (predict_conversion[0] * 1000)
				predict_conversion[2] = (predict_conversion[2] * 1000)
				predict_conversion[1] = (predict_conversion[1] * 1000 + 300)
				predict_conversion[3] = (predict_conversion[3] * 1000 + 300)

				# 绘制两个坐标点
				pt1 = (int(predict_conversion[0]), int(predict_conversion[1]))
				pt2 = (int(predict_conversion[2]), int(predict_conversion[3]))
				print('pt1 = {}, pt2 = {}'.format(pt1, pt2))

				# 在每一帧图片上画出预测的绿线
				cv2.line(frame, pt1, pt2, (0, 250, 0, 2))

			# 显示视频
			cv2.imshow('Frame',frame)

			# 刷新视频
			cv2.waitKey(10)

			# 按q键退出
			if cv2.waitKey(25) & 0xFF == ord('q'):
				break

			# 将图片写入视屏
			videoWriter.write(frame)
		else:
			# 写入视屏结束
			videoWriter.release()
			break

if __name__ == '__main__':
	predict_video("video/", "result_video/")