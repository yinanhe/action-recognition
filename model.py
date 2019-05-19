import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, utils
from torch.autograd import Variable


class LSTMModel(nn.Module):
	def __init__(self, original_model, arch, num_classes, lstm_layers, hidden_size, fc_size):
		super(LSTMModel, self).__init__()
		self.hidden_size = hidden_size
		self.num_classes = num_classes
		self.fc_size = fc_size

		# select a base model
		if arch.startswith('alexnet'):
			self.features = original_model.features
			for i, param in enumerate(self.features.parameters()):
				param.requires_grad = False# 将预训练的模型参数设置不更新
			self.fc_pre = nn.Sequential(nn.Linear(9216, fc_size), nn.Dropout())# 9126 -》 全联接 -〉 单层lstm
			self.rnn = nn.LSTM(input_size = fc_size,
						hidden_size = hidden_size,
						num_layers = lstm_layers,
						batch_first = True)
			self.fc = nn.Linear(hidden_size, num_classes)# lstm输出到结果
			self.modelName = 'alexnet_lstm'

		elif arch.startswith('resnet18'):
			self.features = nn.Sequential(*list(original_model.children())[:-1])
			for i, param in enumerate(self.features.parameters()):
				param.requires_grad = False
			self.fc_pre = nn.Sequential(nn.Linear(512, fc_size), nn.Dropout())
			self.rnn = nn.LSTM(input_size = fc_size,
						hidden_size = hidden_size,
						num_layers = lstm_layers,
						batch_first = True)
			self.fc = nn.Linear(hidden_size, num_classes)
			self.modelName = 'resnet18_lstm'

		elif arch.startswith('resnet50'):
			self.features = nn.Sequential(*list(original_model.children())[:-1])
			for i, param in enumerate(self.features.parameters()):
				param.requires_grad = False
			self.fc_pre = nn.Sequential(nn.Linear(2048, fc_size), nn.Dropout())
			self.rnn = nn.LSTM(input_size = fc_size,
						hidden_size = hidden_size,
						num_layers = lstm_layers,
						batch_first = True)
			self.fc = nn.Linear(hidden_size, num_classes)
			self.modelName = 'resnet50_lstm'

		else:
			raise Exception("This architecture has not been supported yet")

	def init_hidden(self, num_layers, batch_size):
		return (Variable(torch.zeros(num_layers, batch_size, self.hidden_size)).cuda(),
				Variable(torch.zeros(num_layers, batch_size, self.hidden_size)).cuda())

	def forward(self, inputs, hidden=None, steps=0):
		length = len(inputs) # 获得输入的尺寸
		fs = Variable(torch.zeros(length, inputs.size(1),self.rnn.input_size)).cuda()#生成lstm的输入
		for i in range(length):
			f = self.features(inputs[i])
			f = f.view(f.size(0), -1)
			f = self.fc_pre(f)# 得到lstm前的fc结果
			fs[i] = f
		#fs = fs.unsqueeze(0)

		outputs, hidden = self.rnn(fs, hidden)
		outputs = self.fc(outputs)
		return outputs
