import os
import shutil
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, utils
from torch.autograd import Variable
import torch.nn.functional as F

import dataset
from model import *
from train_options import parser

def train(train_loader, model, criterion, optimizer, epoch):
	losses = AverageMeter()# 用AverageMeter来管理更新等
	top1 = AverageMeter()
	top5 = AverageMeter()

	model.train()	# 切换到训练模式

	for i, (input, target, _) in enumerate(train_loader):

		# 输入和目标转化成Variable
		input_var = torch.autograd.Variable(input)
		target_var = torch.autograd.Variable(target)
		input_var, target_var = input_var.cuda(), target_var.cuda()

		# 计算输出
		output = model(input_var)
		#print("计算输出",output.size())
		weight = Variable(torch.Tensor(range(output.shape[1])) / sum(range(output.shape[1]))).cuda().view(-1,1).unsqueeze(0).repeat(output.shape[0],1, output.shape[2])
		output = torch.mul(output, weight)
		output = torch.mean(output, dim=1)#.unsqueeze(1)
		#print("输出",output.size())
		#print('目标输出',target_var.size())
		loss = criterion(output, target_var).mean()
		losses.update(loss.item(), input.size(0))

		# 计算精确度
		prec1, prec5 = accuracy(output.data.cpu(), target, topk=(1, 5))
		top1.update(prec1[0], input.size(0))
		top5.update(prec5[0], input.size(0))

		# 梯度清零
		optimizer.zero_grad()

		# 计算梯度，更新
		loss.backward()
		optimizer.step()
		# 每10个epoch 展示
		if i % 10 == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				'lr {lr:.5f}\t'
				'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				'Top1 {top1.val:.3f} ({top1.avg:.3f})\t'
				'Top5 {top5.val:.3f} ({top5.avg:.3f})'.format(
				epoch, i, len(train_loader),
				lr=optimizer.param_groups[-1]['lr'],
				loss=losses,
				top1=top1,
				top5=top5))


def validate(val_loader, model, criterion):
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# 切换到评估模式
	model.eval()

	for i, (input, target, _) in enumerate(val_loader):

		# target = target.cuda(async=True)
		input_var = torch.autograd.Variable(input)
		target_var = torch.autograd.Variable(target)
		input_var, target_var = input_var.cuda(), target_var.cuda()

		# compute output
		output = model(input_var)
		weight = Variable(torch.Tensor(range(output.shape[1])) / sum(range(output.shape[1]))).cuda().view(-1,1).unsqueeze(0).repeat(output.shape[0],1, output.shape[2])
		output = torch.mul(output, weight)
		output = torch.mean(output, dim=1)#output = torch.mean(output, dim=0).unsqueeze(0)
		loss = criterion(output, target_var)
		losses.update(loss[0].item(), input.size(0))

		# compute accuracy
		prec1, prec5 = accuracy(output.data.cpu(), target, topk=(1, 5))
		top1.update(prec1[0], input.size(0))
		top5.update(prec5[0], input.size(0))

		if i % 10 == 0:
			print ('Test: [{0}/{1}]\t'
					'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					'Top1 {top1.val:.3f} ({top1.avg:.3f})\t'
					'Top5 {top5.val:.3f} ({top5.avg:.3f})'.format(
					i, len(val_loader),
					loss=losses,
					top1=top1,
					top5=top5))

	return (top1.avg, top5.avg)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	# 保存模型
	torch.save(state, os.path.join(args.data, 'save_model', filename))
	if is_best:
		shutil.copyfile(os.path.join(args.data, 'save_model', filename), os.path.join(args.data, 'save_model/model_best.pth.tar'))

class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
	# 90%下降式学习率下降
	if not epoch % args.lr_step and epoch:
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group['lr'] * 0.1
	return optimizer


def accuracy(output, target, topk=(1,)):
	# 计算精确度
	maxk = max(topk)
	# 获得batchsize
	batch_size = target.size(0)
	# 
	_, pred = output.topk(maxk, 1, True, True)# 按照第一维度，得到maxk，返回最大值，并排序
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))# 使用eq计算

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


def main():
	#################
	# 需要输入的参数有：
	# args.data 训练集和验证集存在的文件
	# args.batch-size 训练的batchsize
	# args.workers 并行读取数据
	# arg.model 预训练模型的模型
	# arg.arch CNN模型的选择，默认是alexnet
	# arg.lstm-layers LSTM层的数量，默认是1
	# arg.hidden-size 隐藏层的数量，默认是512
	# arg.fc-size LSTM前的全连接层数量，默认是1024
	# arg.epochs epoch的数量，默认是150
	# arg.lr 学习率 默认是0.01 
	# arg.optim 优化器的选择,默认是sgd
	# arg.momentum 动量选择，默认0.9
	# arg.lr-step 学习率递减频率，默认是50
	# arg.weight-decay 默认1e-4
	###
	global args

	best_prec = 0# 最佳预测
	args = parser.parse_args()

	# Data Transform and data loading
	traindir = os.path.join(args.data, 'train_data')# data文件
	valdir = os.path.join(args.data, 'valid_data')

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
	# transform = transforms.Compose([
	# 								transforms.Resize(224),
	# 								transforms.CenterCrop(224),
	# 								transforms.ToTensor(),
	# 								normalize]
	# 								)
	num_of_image_per_video = args.fpv			

	train_dataset = dataset.loadedDataset(traindir, transform)

	train_loader = torch.utils.data.DataLoader(#train_dataset,
		dataset.loadedDataset(traindir, transform,num_of_image_per_video),
		batch_size=args.batch_size, shuffle=True,
		num_workers=args.workers, pin_memory=True)

	val_loader = torch.utils.data.DataLoader(
		dataset.loadedDataset(valdir, transform,num_of_image_per_video),
		batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True)



	if os.path.exists(args.model):
		# load existing model
		model_info = torch.load(args.model)
		print("==> loading existing model '{}' ".format(model_info['arch']))
		original_model = models.__dict__[model_info['arch']](pretrained=False)
		model = LSTMModel(original_model, model_info['arch'],
			model_info['num_classes'], model_info['lstm_layers'], model_info['hidden_size'], model_info['fc_size'])
		print(model)
		model.cuda()
		model.load_state_dict(model_info['state_dict'])
		best_prec = model_info['best_prec']
		cur_epoch = model_info['epoch']
	else:
		# load and create model
		print("==> creating model '{}' ".format(args.arch))

		original_model = models.__dict__[args.arch](pretrained=True)
		model = LSTMModel(original_model, args.arch,
			len(train_dataset.classes), args.lstm_layers, args.hidden_size, args.fc_size)
		print(model)
		model.cuda()
		cur_epoch = 0

	# loss criterion and optimizer
	criterion = nn.CrossEntropyLoss(reduction='none')
	criterion = criterion.cuda()

	if args.optim == 'sgd':
		optimizer = torch.optim.SGD([{'params': model.fc_pre.parameters()},
									{'params': model.rnn.parameters()},
									{'params': model.fc.parameters()}],
									lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

	elif args.optim == 'rmsprop':
		optimizer = torch.optim.RMSprop([{'params': model.fc_pre.parameters()},
									{'params': model.rnn.parameters()},
									{'params': model.fc.parameters()}],
									lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

	elif args.optim == 'adam':
		optimizer = torch.optim.Adam([{'params': model.fc_pre.parameters()},
									{'params': model.rnn.parameters()},
									{'params': model.fc.parameters()}],
									lr=args.lr, weight_decay=args.weight_decay)


	# Training on epochs
	for epoch in range(cur_epoch, args.epochs):

		optimizer = adjust_learning_rate(optimizer, epoch)

		print("---------------------------------------------------Training---------------------------------------------------")

		# train on one epoch
		train(train_loader, model, criterion, optimizer, epoch)

		print("--------------------------------------------------Validation--------------------------------------------------")

		# evaluate on validation set
		prec1, prec5 = validate(val_loader, model, criterion)

		print("------Validation Result------")
		print("   Top1 accuracy: {prec: .2f} %".format(prec=prec1.item()))
		print("   Top5 accuracy: {prec: .2f} %".format(prec=prec5.item()))
		print("-----------------------------")

		# remember best top1 accuracy and save checkpoint
		is_best = prec1 > best_prec
		best_prec = max(prec1, best_prec)
		save_checkpoint({
			'epoch': epoch + 1,
			'arch': args.arch,
			'num_classes': len(train_dataset.classes),
			'lstm_layers': args.lstm_layers,
			'hidden_size': args.hidden_size,
			'fc_size': args.fc_size,
			'state_dict': model.state_dict(),
			'best_prec': best_prec,
			'optimizer' : optimizer.state_dict(),}, is_best)

if __name__ == '__main__':
	main()