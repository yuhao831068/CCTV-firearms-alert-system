from PIL import Image
from models.SRCNN import SRCNN
from models.BLIP2 import blip2_model, processor
from models.MultiModalModel import MultiModalModel
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import os
import pickle
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# load my Tf-Idf vectorizer
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


def sr_image(image_path):
    print("Super resolution starts.")
    # open image
    img = Image.open(image_path).convert('YCbCr')

    # convert to YCbCr
    y, cb, cr = img.split()

    # define model
    srcnn_model = SRCNN()

    # load weight
    weight = os.path.join('models/SRCNN.pth')
    srcnn_model.load_state_dict(torch.load(weight, map_location=torch.device('cpu'), weights_only=True))

    # convert Y channel to tensor and resize to [0,1]
    input_tensor = transforms.ToTensor()(y).view(1, 1, y.size[1], y.size[0])

    # input tensor to SRCNN
    with torch.no_grad():
        output = srcnn_model(input_tensor)

    # transfer output to RGB image, and resize to [0, 255]
    output_img_y = output[0].cpu().detach().numpy().squeeze(0)
    output_img_y = (output_img_y * 255.0).clip(0, 255).astype('uint8')
    output_img_y = Image.fromarray(output_img_y)

    # combine YCbCr and transfer to RGB
    sr_img = Image.merge('YCbCr', [output_img_y, cb, cr]).convert('RGB')
    print("Super resolution is finished.")

    return sr_img


def get_description(sr_img):
    # input super resolution image.
    inputs = processor(sr_img, return_tensors="pt")

    print("Generating description...")
    with torch.no_grad():
        out = blip2_model.generate(**inputs)

    print("Decoding output...")
    result = processor.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
    print('Description is generated')
    return result


def generate_description_vector(description):
    description_vector = vectorizer.transform([description]).toarray()[0]
    description_tensor = torch.FloatTensor(description_vector).unsqueeze(0)
    return description_tensor


def run_multi_modal_model(image_path):
    vgg19_weights = models.VGG19_Weights.DEFAULT
    vgg19_transforms = vgg19_weights.transforms()

    multimodal_model = MultiModalModel(284, 2)
    multimodal_model_weights = os.path.join(
        'models/multimodal best weights.pth')
    multimodal_model.load_state_dict(torch.load(multimodal_model_weights, map_location=torch.device('cpu'), weights_only=True))

    sr_img = sr_image(image_path)
    resize_sr_img = sr_img.resize((224,224))
    img_tensor = vgg19_transforms(resize_sr_img).unsqueeze(0)
    description = get_description(sr_img)
    description_tensor = generate_description_vector(description)

    multimodal_model.eval()
    with torch.no_grad():
        output = multimodal_model(img_tensor, description_tensor)
        max_value, predicted_class = torch.max(output, 1)

        return predicted_class.item(), resize_sr_img, description
