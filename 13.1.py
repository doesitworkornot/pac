import cv2
import numpy as np
import torch.nn.functional
import torchvision
from PIL import Image


data = None


def get_feature_map(module, inputs, output):
    global data
    data = output


def normalize_image(img):
    transform_pipe = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            size=(224, 224)
        ),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    img_tensor = transform_pipe(img)
    batch = img_tensor[None, :, :, :]
    return batch


def calc_impact(col):
    global data
    data = data.squeeze()
    col = col[None, :]
    ans = col @ data.reshape(2048, 49)
    ans = ans.reshape(7, 7).detach().numpy()
    return ans


def scale_img_by_mask(image, imp):
    b, g, r = cv2.split(image)
    b = b * imp
    g = g * imp
    r = r * imp
    merge_image = cv2.merge([b, g, r])
    merge_image = merge_image.astype(np.uint8)
    return merge_image


def parse_img(image, masked_image, label):
    im_to_show = cv2.hconcat([masked_image, image])
    h, w, _ = im_to_show.shape
    scale = h/w
    w = 1920
    h = int(w*scale)
    padding = 1010 - h
    im_to_show = cv2.resize(im_to_show, [w, h])
    border = cv2.copyMakeBorder(im_to_show, padding, 0, 0, 0, cv2.BORDER_CONSTANT)

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 100)
    fontScale = 3
    fontColor = (255, 255, 255)
    thickness = 1
    lineType = 2

    with_text = cv2.putText(border, label,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)

    return with_text


def main():
    # Creating model
    model = torchvision.models.resnet50(weights=True)
    global data
    model.layer4.register_forward_hook(get_feature_map)
    model.eval()

    # Loading categories
    with open("data/lab13/images_classes.txt") as f:
        categories = [s.strip() for s in f.readlines()]

    cap = cv2.VideoCapture('data/lab13/roomtour.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, image = cap.read()
        if image is None:
            break
        # Preparing image
        image = Image.fromarray(image)
        batch = normalize_image(image)

        result = model(batch)

        # Calculating importance factors
        ind_predicted = torch.argmax(result)
        col = model.fc.weight[ind_predicted]
        importance_mask = calc_impact(col)

        # Normalizing our factors and scaling by them
        importance_mask = importance_mask / np.max(importance_mask)
        importance_mask = np.where(importance_mask > 0, importance_mask, 0)
        importance_mask = cv2.resize(importance_mask, image.size)

        # Scaling image using mask
        image = np.array(image)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        masked_image = scale_img_by_mask(image, importance_mask)

        # Parsing output
        label = categories[ind_predicted]
        img_to_show = parse_img(image, masked_image, label)

        # Showing data
        cv2.imshow('Highlighting', img_to_show)
        if cv2.waitKey(int(1000 / fps)) == ord('q'):
            break


if __name__ == '__main__':
    main()
