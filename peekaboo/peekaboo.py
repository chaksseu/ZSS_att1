from typing import Union, List, Optional
import numpy as np
import rp
import torch
import torch.nn as nn
from easydict import EasyDict

import peekaboo.stable_diffusion as sd
from peekaboo.mask_generation import (BilateralProxyBlur,
                                    LearnableImageFourier,
                                    LearnableImageFourierBilateral,
                                    LearnableImageRaster,
                                    LearnableImageRasterBilateral)

#Importing this module loads a stable diffusion model. Hope you have a GPU
sd_iff = sd.StableDiffusion(device='cuda',
                            checkpoint_path='CompVis/stable-diffusion-v1-4',
                            variant="fp16",
                            token='_________',)
device = sd_iff.device


#==========================================================================================================================

def make_learnable_image(height, width, num_channels, foreground=None, bilateral_kwargs=None, representation='fourier'):
    "이미지의 파라미터화 방식을 결정하여 학습 가능한 이미지를 생성."
    bilateral_kwargs = bilateral_kwargs or {}
    bilateral_blur = BilateralProxyBlur(foreground, **bilateral_kwargs)
    if representation == 'fourier bilateral':
        return LearnableImageFourierBilateral(bilateral_blur, num_channels)
    elif representation == 'raster bilateral':
        return LearnableImageRasterBilateral(bilateral_blur, num_channels)
    elif representation == 'fourier':
        return LearnableImageFourier(height, width, num_channels)
    elif representation == 'raster':
        return LearnableImageRaster(height, width, num_channels)
    else:
        raise ValueError(f'Invalid method: {representation}')

def blend_torch_images(foreground, background, alpha):
    '주어진 foreground와 background 이미지를 alpha 값에 따라 블렌딩합니다.'
    C, H, W = foreground.shape
    return foreground * alpha + background * (1 - alpha)

def make_image_square(image: np.ndarray, method='crop') -> np.ndarray:
    """
    주어진 이미지를 512x512(x3) 크기의 정사각형으로 변환. 
    method는 'crop' 또는 'scale' 중 하나를 사용할 수 있다."""
    
    try:
        image = rp.as_rgb_image(image)  # 이미지가 3채널 RGB인지 확인 및 변환
        height, width = rp.get_image_dimensions(image)
        min_dim = min(height, width)
    except:
        height, width = rp.get_image_dimensions(image)
        min_dim = min(height, width)

    if method == 'crop':
        # 중앙에서 자르고 스케일링하여 정사각형 이미지 생성
        return make_image_square(rp.crop_image(image, min_dim, min_dim, origin='center'), 'scale')
    
    # 'scale' 메서드일 경우 이미지 크기를 512x512로 조정
    return rp.resize_image(image, (512, 512))

#==========================================================================================================================

class PeekabooResults(EasyDict):
    pass

class BaseLabel:
    def __init__(self, name:str, embedding:torch.Tensor):
        self.name=name
        self.embedding=embedding

    def __repr__(self):
        return f"{type(self).__name__}(name={self.name})"
        
class SimpleLabel(BaseLabel):
    def __init__(self, name:str):
        super().__init__(name, sd_iff.get_text_embeddings(name).to(device))

#==========================================================================================================================

def save_peekaboo_results(results, new_folder_path):
    import json
    rp.make_folder(new_folder_path)

    with rp.SetCurrentDirectoryTemporarily(new_folder_path):
        print(f"\nSaving PeekabooResults to {new_folder_path}")
        params = {}

        for key, value in results.items():
            if rp.is_image(value):
                rp.save_image(value, f'{key}.png')  # 단일 이미지 저장
            elif isinstance(value, np.ndarray) and rp.is_image(value[0]):
                rp.make_directory(key)  # 이미지 폴더 저장
                with rp.SetCurrentDirectoryTemporarily(key):
                    [rp.save_image(img, f'{i}.png') for i, img in enumerate(value)]
            elif isinstance(value, np.ndarray):
                np.save(f'{key}.npy', value)  # 일반 Numpy 배열 저장
            else:
                try:
                    json.dumps({key: value})  # JSON으로 변환 가능한 값 저장
                    params[key] = value
                except Exception:
                    params[key] = str(value)  # 변환 불가한 값은 문자열로 저장

        rp.save_json(params, 'params.json', pretty=True)
        print(f"Done saving PeekabooResults to {new_folder_path}!")

#==========================================================================================================================

class PeekabooSegmenter(nn.Module):
    '이미지 분할을 위한 PeekabooSegmenter 클래스.'
    
    def __init__(self, 
                 image: np.ndarray, 
                 labels: List['BaseLabel'], 
                 size: int = 256,
                 channel: int = 3,
                 name: str = 'Untitled', 
                 bilateral_kwargs: dict = None, 
                 representation: str = 'fourier bilateral', 
                 min_step=None, 
                 max_step=None):
        super().__init__()     

        self.height = self.width = size  #We use square images for now
        self.channel = channel
        self.labels = labels
        self.name = name
        self.representation = representation
        self.min_step = min_step
        self.max_step = max_step
        
        # 이미지 전처리
        image = rp.cv_resize_image(image, (self.height, self.width))  # np (256,256,3)
        # 채널에 맞게 이미지 변환
        if self.channel == 3:
            image = rp.as_rgb_image(image)  # 3채널로 변환 (RGB)
        image = rp.as_float_image(image)  # 값의 범위를 0과 1 사이로 변환
        self.image = image  # np (256,256,3) & norm
        
        # 이미지를 Torch 텐서로 변환 (CHW 형식)
        self.foreground = rp.as_torch_image(image).to(device)  # torch (3,256,256) & norm
        
        # 배경은 단색으로 설정
        self.background = torch.zeros_like(self.foreground)
        
        # 학습 가능한 알파 값 생성
        bilateral_kwargs = bilateral_kwargs or {}
        self.alphas = make_learnable_image(self.height,
                                           self.width,
                                           num_channels=len(labels), 
                                           foreground=self.foreground, 
                                           representation=self.representation, 
                                           bilateral_kwargs=bilateral_kwargs)
            
    @property
    def num_labels(self):
        return len(self.labels)
            
    def set_background_color(self, color):
        r,g,b = color
        self.background[0] = r
        self.background[1] = g
        self.background[2] = b

    def default_background(self):
        self.background[0] = 0
        self.background[1] = 0
        self.background[2] = 0
        
    def forward(self, alphas=None, return_alphas=False):        
        # alpha 값이 없으면, 학습된 alpha 생성
        alphas = alphas if alphas is not None else self.alphas()

        # alpha 값을 이용하여 각 라벨에 대한 이미지를 생성
        output_images = [blend_torch_images(self.foreground, self.background, alpha) for alpha in alphas]
        output_images = torch.stack(output_images)

        return (output_images, alphas) if return_alphas else output_images

def run_peekaboo(name: str,
                 image: Union[str, np.ndarray],
                 label: Optional['BaseLabel'] = None,

                 GRAVITY=1e-1/2,
                 NUM_ITER=300,
                 LEARNING_RATE=1e-5, 
                 BATCH_SIZE=1,   
                 GUIDANCE_SCALE=100,
                 bilateral_kwargs=dict(
                     kernel_size=3,
                     tolerance=0.08,
                     sigma=5,
                     iterations=40
                     ),
                 square_image_method='crop', 
                 representation='fourier bilateral',
                 min_step=None, 
                 max_step=None) -> PeekabooResults:
    """
    Peekaboo Hyperparameters:
        GRAVITY: prompt에 따라 tuning이 제일 필요함. 주로 1e-2, 1e-1/2, 1e-1, 1.5*1e-1에서 잘 됨.
        NUM_ITER: 300이면 대부분 충분
        LEARNING_RATE: neural neural textures 아닐 경우, 값 키워도 됨
        BATCH_SIZE: 큰 차이 없음. 배치 1 키우면 vram만 잡아먹음
        GUIDANCE_SCALE=100: DreamFusion 논문의 고정 값.
        bilateral_kwargs = (kernel_size=3,tolerance=.08,sigma=5,iterations=40)
        square_image_method: input image를 정사각형화 하는 두 가지 방법. (crop / scale)
        representation: (fourier bilateral / raster bilateral / fourier / raster)
    """
    
    # 레이블이 없을 경우 기본 레이블 생성
    label = label or SimpleLabel(name)  # label이 갖고 있는 embedding dim은 (2,77,768)

    # 이미지 로드 및 전처리
    image_path = image if isinstance(image, str) else '<No image path given>'
    image = rp.load_image(image_path) if isinstance(image, str) else image  # np (500,500,3)
    image = rp.as_rgb_image(rp.as_float_image(make_image_square(image, square_image_method)))  # np (512,512,3)
    
    # PeekabooSegmenter 생성
    pkboo=PeekabooSegmenter(
                        image, labels=[label], name=name, 
                        bilateral_kwargs=bilateral_kwargs, 
                        representation=representation, 
                        min_step=min_step, 
                        max_step=max_step
                        ).to(device)

    # 옵티마이저 설정
    params = list(pkboo.parameters())
    optim = torch.optim.SGD(params, lr=LEARNING_RATE)

    # 학습 반복 설정
    global iter_num
    iter_num = 0
    timelapse_frames=[]
    preview_interval = max(1, NUM_ITER // 10)  # 10번의 미리보기를 표시

    try:
        display_eta = rp.eta(NUM_ITER)
        for _ in range(NUM_ITER):
            display_eta(_)
            iter_num += 1

            alphas = pkboo.alphas()
            for __ in range(BATCH_SIZE):
                pkboo.default_background()
                composites = pkboo()
                for label, composite in zip(pkboo.labels, composites):
                    sd_iff.train_step(label.embedding, composite[None], guidance_scale=GUIDANCE_SCALE)

            ((alphas.sum()) * GRAVITY).backward()
            optim.step()
            optim.zero_grad()

            with torch.no_grad():
                if not _ % preview_interval: 
                    timelapse_frames.append(pkboo.display())

    except KeyboardInterrupt:
        print("Interrupted early, returning current results...")

  