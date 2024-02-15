import torch.nn.functional
import torchvision
import cv2
import numpy as np
import time


def slice_im(path):
    image = cv2.imread(path)
    height, width = image.shape[:2]
    shift = 75
    im_arr = []
    for y in range(0, height - 223, shift):
        for x in range(0, width - 223, shift):
            sub_image = image[y:y + 224, x:x + 224]
            new_im = normalize_img(sub_image)
            im_arr.append(new_im)
    sub_images_tensor = torch.tensor(np.array(im_arr))
    return sub_images_tensor


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
    cuda0 = torch.device('cuda:0')
    model.to(device=cuda0)
    base_img = cv2.imread('data/lab14/defdin.jpg')
    base_img = cv2.resize(base_img, [224, 224])
    base_img = normalize_img(base_img).to(device=cuda0)

    get_ims = slice_im('data/lab14/abc.jpg').to(device=cuda0)

    base_vector = model(base_img)
    print('Base vector: ', base_vector.shape, base_vector.dtype)
    print('Multiple img: ', get_ims.shape, get_ims.dtype)
    print('Base img: ', base_img.shape, base_img.dtype)
    h = get_ims.shape[0]
    im_vector = torch.zeros([h, 2048], dtype=torch.float32, device=cuda0)
    time_start = time.perf_counter()
    for i in range(h):
        im_vector[i] = model(get_ims[i])
    time_end = time.perf_counter()
    print(f'It took {time_end-time_start} seconds!')
    base_norm = base_vector / torch.linalg.vector_norm(base_vector)
    sin_sim = torch.zeros(h, dtype=torch.float32, device=cuda0)
    for i in range(h):
        vector_norm = im_vector[i]/torch.linalg.vector_norm(im_vector[i])
        sin_sim[i] = torch.inner(vector_norm, base_norm)
    print(h)
    print(torch.argmax(sin_sim), sin_sim.shape)
    most_sim = torch.argmax(sin_sim)
    most_im = get_ims[most_sim].squeeze()
    most_im = most_im.to('cpu').permute(1, 2, 0).numpy()
    cv2.imshow('Most similar', most_im)
    cv2.waitKey()


if __name__ == '__main__':
    main()
