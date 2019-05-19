import argparse
import os

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, utils
from torch.autograd import Variable

import dataset
from model import *

parser = argparse.ArgumentParser(description = 'Predicting')
# 用法：python3 predict.py data/model/model_best.pth data/predict_data/

parser.add_argument('model', metavar = 'DIR', help = 'path to model')
parser.add_argument('data', metavar = 'DIR', help = 'path to dataset')


def predict(predict_loader, model):
	# switch to evaluate mode 将模型设置成验证模式
	model.eval()
	
	result = [''] * 1500

	classes = []
	# 打开class文件读取每一行作为动作分类结果
	with open('./data/classes.txt', 'r') as cfile:
		classes = cfile.readlines()

	for i, (input, _, name) in enumerate(predict_loader):
		input_var = torch.autograd.Variable(input).cuda()

		# # compute output
		# #  计算网络输出，计算权重
		# output = model(input_var[0])	
		# weight = Variable(torch.Tensor(range(output.shape[0])) / sum(range(output.shape[0]))).cuda().view(-1,1).repeat(1, output.shape[1])
		# # 将网络输出与权重相乘
		# output = torch.mul(output, weight)
		# # 求均值
		# output = torch.mean(output, dim=0).unsqueeze(0).data.cpu()
		# 计算输出
		output = model(input_var)
		#print("计算输出",output.size())
		weight = Variable(torch.Tensor(range(output.shape[1])) / sum(range(output.shape[1]))).cuda().view(-1,1).unsqueeze(0).repeat(output.shape[0],1, output.shape[2])
		output = torch.mul(output, weight)
		output = torch.mean(output, dim=1).data.cpu()
		# 获得最大的可能
		_, pred = output.topk(1, 1, True, True)
		pred = pred.t()
		print('File Name: ' + name[0])
		print('Predict: ' + classes[pred[0][0].numpy()])# 最大的分类名称
		
	# 	result[int(name[0])] = classes[pred[0][0].numpy()]

	# file= open('data/predict.txt', 'w')  
	# for fp in result:
	# 	file.write(str(fp))
	# file.close()

def main():
	args = parser.parse_args()

	predictdir = args.data

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									std=[0.339, 0.224, 0.225])

	transform = (transforms.Compose([
									transforms.Resize(224),
									transforms.CenterCrop(224),
									transforms.ToTensor(),
									normalize]
									),
				transforms.Compose([
									transforms.Resize(224),
									transforms.CenterCrop(224),
									transforms.ToTensor()]
									)
				)

	predict_loader = torch.utils.data.DataLoader(
		dataset.loadedDataset(predictdir, transform),
		batch_size=1, shuffle=False,
		num_workers=8, pin_memory=True)

	if os.path.exists(args.model):
		# load existing model加载模型
		model_info = torch.load(args.model)
		# 加载模型，并返回卷积提取网络
		print("==> loading existing model '{}' ".format(model_info['arch']))
		original_model = models.__dict__[model_info['arch']](pretrained=False)
		model = LSTMModel(original_model, model_info['arch'],
			model_info['num_classes'], model_info['lstm_layers'], model_info['hidden_size'], model_info['fc_size'])
		print(model)
		# 模型转化到显卡，并从pth中加载权重
		model.cuda()
		model.load_state_dict(model_info['state_dict'])
	else:
		print("Error: load model failed!")
		return

	predict(predict_loader, model)

if __name__ == '__main__':
	main()
