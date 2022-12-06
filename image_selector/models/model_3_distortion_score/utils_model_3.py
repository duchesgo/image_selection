import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
from image_selector.models.model_3_distortion_score.preprocessing_model_3 import Image_load
from PIL import Image
import os
import torchvision.transforms as T

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device('cpu')
WEIGHT_PATH=os.environ.get("WEIGHT_MODEL_3_PATH")
WEIGHT_PATH=os.environ.get("LOCAL_PROJECT_PATH")+"registry/model_3_distortion_score/weights_model_3.pt"

class ResNet(nn.Module):

	def __init__(self):
		super().__init__()
		self.backbone = torchvision.models.resnet50(pretrained=False)
		fc_feature = self.backbone.fc.in_features
		self.backbone.fc = nn.Linear(fc_feature, 1, bias=True)

	def forward(self, x):
		result = self.backbone(x)
		# result = self.backbone.fc(result)
		return result

def predict_quality(model, image):

	prepare_image = Image_load(size=512, stride=224)
	image = prepare_image(Image.open(image).convert("RGB")).to(DEVICE)


	pred = model(image)

	print(pred.shape)
	print(pred.mean())
	return(float(pred.mean()))

def predict_quality_from_pil(model,image):

	prepare_image = Image_load(size=512, stride=224)
	image = prepare_image(image).to(DEVICE)

	pred = model(image)

	return(float(pred.mean()))

def update_model(model):
	model.backbone.fc = nn.Linear(model.backbone.fc.in_features, 6, bias=True)
	return model


def return_score_torch(image):

	to_pil_image=T.ToPILImage()
	pil_image=to_pil_image(image)

	DEVICE = torch.device('cpu')

	model = ResNet()
	model.to(DEVICE)
	model.eval()

	checkpoint = torch.load(WEIGHT_PATH, map_location=DEVICE)
	model.load_state_dict(checkpoint['state_dict'])

	#model = update_model(model)
	prediciton=predict_quality_from_pil(model, pil_image)

	return prediciton

if __name__ == '__main__':

	pass
