import cv2
import os
#要提取视频的文件名，隐藏后缀
sourceFileName = 'laser'
#在这里把后缀接上
video_path = os.path.join("video/", sourceFileName + '.mp4')
times = 0
#提取视频的频率，每１帧提取一个
frameFrequency = 1
#输出图片到当前目录vedio文件夹下
outPutDirName = 'Image2/'
if not os.path.exists(outPutDirName):
    #如果文件目录不存在则创建目录
    os.makedirs(outPutDirName)
camera = cv2.VideoCapture(video_path)
num = 0
a = 300
while True:
    num = num + 1
    res, image = camera.read()
    if not res:
        print('not res , not image')
        break
    if (num > 200) and (num < 251) :
        if num % frameFrequency == 0:
            a = a + 1
            cv2.imwrite(outPutDirName + str(int(a)) + '.jpg', image)
            print(outPutDirName + str(int(a)) + '.jpg')
print('图片提取结束')
camera.release()