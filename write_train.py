def write_train(train_data_number):
	with open("ImageSets/Main/train.txt", 'w') as f:
		for num in range(train_data_number):
				f.write(str(num + 1) + '\n')

if __name__ == '__main__':
	write_train(1000)