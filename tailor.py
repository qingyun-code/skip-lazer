import cv2

def tailor(item):
	img = cv2.imread("Image/" + item + ".jpg")
	tailor_image = img[300:1300, 0:1000]
	cv2.imwrite("Image3/" + item + ".jpg", tailor_image)

if __name__ == '__main__':
	for item in range(1000):
		tailor(str(item + 1))