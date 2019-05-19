import os
import random
import torch
import numpy as np
import PIL.Image as Image

from torch.utils.data import Dataset
from torchvision import transforms, utils

class loadedDataset(Dataset):
	def __init__(self, root_dir, transform=None,num_of_image_per_video=40):
		self.root_dir = root_dir# 根目录
		self.transform = transform# 是否进行变换，如归一化等操作
		self.num_of_image_per_video = num_of_image_per_video
		self.classes = sorted(os.listdir(self.root_dir))# 获得真实的标签总数量，50类
		self.count = [len(os.listdir(self.root_dir + '/' + c)) for c in self.classes]# 获得不同动作视频数量
		self.acc_count = [self.count[0]]#
		for i in range(1, len(self.count)):
				self.acc_count.append(self.acc_count[i-1] + self.count[i])#count的累加
		# self.acc_count = [self.count[i] + self.acc_count[i-1] for i in range(1, len(self.count))]

	def __len__(self):
		l = np.sum(np.array([len(os.listdir(self.root_dir + '/' + c)) for c in self.classes]))
		return l

	def __getitem__(self, idx):
		for i in range(len(self.acc_count)):# 不同的类别，对每个类别进行便利
			if idx < self.acc_count[i]:
				label = i
				break

		class_path = self.root_dir + '/' + self.classes[label] 

		if label:# 如果有label的话，返回这个label下的那个文件
			file_path = class_path + '/' + sorted(os.listdir(class_path))[idx-self.acc_count[label]]
		else:
			file_path = class_path + '/' + sorted(os.listdir(class_path))[idx]

		_, file_name = os.path.split(file_path)

		frames = []

		# print os.listdir(file_path) 文件夹下的所有视频帧
		file_list = sorted(os.listdir(file_path))
		# print file_list
		step = int(float(len(file_list)) / float(self.num_of_image_per_video))
		# 为了多batch训练，抽帧出来用
		if step == 0:
		    # Supplement 少于阈值，把最后的图重复几遍
		    file_list += [file_list[-1]] * (self.num_of_image_per_video - len(file_list))
		if step == 1:
		    # cut from the middle 读取中间17张帧
		    start_num = np.floor(float(len(file_list) - self.num_of_image_per_video) / 2)
		    start = 1 if start_num == 0 else start_num
		    file_list = file_list[int(start - 1):int(self.num_of_image_per_video + start - 1)]
		if step > 1:
		    # 每间隔step个图片取一次，再取中间的17个
		    file_list = file_list[step - 1::step]
		    start_num = np.floor(float(len(file_list) - self.num_of_image_per_video) / 2)
		    start = 1 if start_num == 0 else start_num
		    file_list = file_list[int(start - 1):int(self.num_of_image_per_video + start - 1)]
		# v: maximum translation in every step
		v = 2
		offset = 0 # 偏移量为0
		for i, f in enumerate(file_list):
			frame = Image.open(file_path + '/' + f)#读取对应帧
			#translation
			offset += random.randrange(-v, v)
			offset = min(offset, 3 * v)
			offset = max(offset, -3 * v)
			frame = frame.transform(frame.size, Image.AFFINE, (1, 0, offset, 0, 1, 0))# 仿射变换
			if self.transform is not None:
				frame = self.transform[0](frame)
			frames.append(frame)

		frames = torch.stack(frames)# 拼接
		#frames = frames[: -1] - frames[1:]

		return frames, label, file_name
