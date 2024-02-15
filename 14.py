import torch.nn.functional
import torchvision
import cv2
import numpy as np


def slice_im(path):
    im = cv2.imread(path)
    h, w, _ = im.shape
    im = cv2.resize(im, [(w//10)*10, (h//10)*10])
    im_arr = torch.tensor(np.zeros((((h-224)//10), ((w-224)//10)+1, 1, 3, 224, 224)))
    for height in range(0, h-224, 10):
        for width in range(0, w-224, 10):
            ind_h = height//10
            ind_w = width//10
            new_im = normalize_img(im[height:height+224, width:width+224])
            im_arr[ind_h][ind_w] = new_im
    return im_arr


def normalize_img(im):
    transform_pipe = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    batch = transform_pipe(im)
    batch = batch[None, :, :, :]
    return batch


def main():
    model = torchvision.models.resnet50(weights=True)
    model.fc = torch.nn.Identity()
    model.eval()

    base_img = cv2.imread('source/defdin.jpg')
    base_img = cv2.resize(base_img, [224, 224])
    base_img = normalize_img(base_img)

    get_ims = slice_im('source/abc.png')

    base_vector = model(base_img)
    print('Base vector: ', base_vector.shape, base_vector.dtype)
    print('Multiple img: ', get_ims.shape, get_ims.dtype)
    print('Base img: ', base_img.shape, base_img.dtype)
    h, w, _, _, _, _ = get_ims.shape
    im_vector = torch.zeros([h, w, 1, 2048], dtype=torch.float32)
    for i in range(h):
        for j in range(w):
            im_vector[i][j] = model(get_ims[i][j].to(torch.float32))


    cv2.waitKey()


if __name__ == '__main__':
    main()
