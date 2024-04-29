from share import *
import config

import os
import cv2
import einops
import numpy as np
import torch
import random
from tqdm import tqdm

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from fashion_dataset import MyDataset
from torch.utils.data import DataLoader

model = create_model('./models/cldm_v15.yaml').cpu()
checkpoint_path = './checkpoints/latest-model-v1.ckpt'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
model.load_state_dict(checkpoint['state_dict'])
# model.load_state_dict(load_state_dict('./models/control_sd15_scribble.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = np.zeros_like(img, dtype=np.uint8)
        detected_map[np.min(img, axis=2) < 127] = 255

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [255 - detected_map] + results


def main():
    dataset = MyDataset()

    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    batch_size = 1

    # Split the dataset
    torch.manual_seed(0)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create data loaders to handle batching
    train_loader = DataLoader(train_dataset,num_workers=0, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, num_workers=0, batch_size=batch_size, shuffle=False)

    num_samples = 1
    image_resolution = 512
    strength = 1.0
    guess_mode = False
    ddim_steps = 20
    scale = 9.0
    seed = np.random.randint(-1, 2147483647 + 1)
    eta = 0.0
    a_prompt = 'best quality, extremely detailed'
    n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

    def process_images(dataloader, dir):
        count = 0
        for x_dict in tqdm(dataloader):
            # print(x_dict)
            input_image = x_dict['hint'][0]
            # source_img = x_dict['jpg'][0]
            prompt = x_dict['txt'][0]
            filename = x_dict['filename'][0]
            output = process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, 
                             ddim_steps, guess_mode, strength, scale, seed, eta)
            
            # cv2.imwrite(f'./Results/{dir}/original/{filename}', source_img)
            cv2.imwrite(f'./Results/{dir}/reproduced/{filename}', output[1])
            count += 1

            if count == 100:
                break


    # process_images(train_loader, 'Train')
    process_images(test_loader, 'Test')
    

if __name__ == '__main__':
    main()