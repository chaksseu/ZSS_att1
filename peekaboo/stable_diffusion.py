from typing import Union, List, Optional
from transformers import logging
from diffusers import PNDMScheduler
from diffusers import StableDiffusionPipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
import rp

# Suppress partial model loading warning
logging.set_verbosity_error()


class StableDiffusion(nn.Module):
    def __init__(self, token, device='cuda', checkpoint_path="CompVis/stable-diffusion-v1-4", variant="fp16"):
        super().__init__()

        self.device = torch.device(device)
        self.num_train_timesteps = 1000

        # Timestep ~ U(0.02, 0.98) to avoid very high/low noise levels
        self.min_step = int(self.num_train_timesteps * 0.02)  # aka 20
        self.max_step = int(self.num_train_timesteps * 0.98)  # aka 980

        # Unlike the original code, I'll load these from the pipeline. This lets us use DreamBooth models.
        pipe = StableDiffusionPipeline.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16, ##
                use_auth_token=token, ##
                variant=variant, ##
                low_cpu_mem_usage=True) ##

        pipe.safety_checker = None                
        pipe.scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=self.num_train_timesteps) #Error from scheduling_lms_discrete.py
        
        self.pipe         = pipe
        self.vae          = pipe.vae.to(self.device)
        self.tokenizer    = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.to(self.device)
        self.unet         = pipe.unet.to(self.device)
        self.scheduler    = pipe.scheduler

        self.uncond_text = ''
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience
        print(f'[INFO] sd.py: loaded stable diffusion!')


    def get_text_embeddings(self,
                            prompts: Union[str, List[str]]
                            ) -> torch.Tensor:

        if isinstance(prompts, str):
            prompts = [prompts]

        text_input = self.tokenizer(prompts,
                                    padding='max_length',
                                    max_length=self.tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors='pt'
                                    ).input_ids    # (1,77)
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.to(self.device))[0]  # (1,77,768)

        uncond_input = self.tokenizer([self.uncond_text] * len(prompts),
                                      padding='max_length',
                                      max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt'
                                      ).input_ids
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.to(self.device))[0]

        output_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return output_embeddings  # (2,77,768)

    
    def train_step(self,
                   text_embeddings: torch.Tensor,
                   pred_rgb: torch.Tensor,
                   guidance_scale: float = 100,
                   t: Optional[int] = None):
        '이 메서드는 dream-loss gradients을 생성하는 역할.'

        # interp to 512x512 to be fed into vae
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False).to(torch.float16) ##

        if t is None:
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # VAE를 사용하여 이미지를 latents로 인코딩 / grad 필요.
        latents = self.encode_imgs(pred_rgb_512)

        # unet으로 noise residual 예측 / NO grad.
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t.cpu())
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

        # guidance 수행 (high scale from paper)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2   
        w = (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        # grad에서 item을 생략하고 자동 미분 불가능하므로, 수동 backward 수행.
        latents.backward(gradient=grad, retain_graph=True)
        return 0  # dummy loss value

    def encode_imgs(self, imgs: torch.Tensor) -> torch.Tensor:
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs)
        if hasattr(posterior, 'latent_dist'):
            latents = posterior.latent_dist.sample() * 0.18215
        else:
            latents = posterior.sample() * 0.18215
        return latents
