from __future__ import annotations

import os
import pathlib
import random
import sys
from typing import Any

import cv2
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import tqdm.auto
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel

HF_TOKEN = os.getenv('HF_TOKEN')

repo_dir = pathlib.Path(__file__).parent
submodule_dir = repo_dir / 'ELITE'
snapshot_download('ELITE-library/ELITE',
                  repo_type='model',
                  local_dir=submodule_dir.as_posix(),
                  token=HF_TOKEN)
sys.path.insert(0, submodule_dir.as_posix())

from train_local import (Mapper, MapperLocal, inj_forward_crossattention,
                         inj_forward_text, th2image)


def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [T.ToTensor()]
    if normalize:
        transform_list += [
            T.Normalize((0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711))
        ]
    return T.Compose(transform_list)


def process(image: np.ndarray, size: int = 512) -> torch.Tensor:
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
    image = np.array(image).astype(np.float32)
    image = image / 127.5 - 1.0
    return torch.from_numpy(image).permute(2, 0, 1)


class Model:
    def __init__(self):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        (self.vae, self.unet, self.text_encoder, self.tokenizer,
         self.image_encoder, self.mapper, self.mapper_local,
         self.scheduler) = self.load_model()

    def download_mappers(self) -> tuple[str, str]:
        global_mapper_path = hf_hub_download('ELITE-library/ELITE',
                                             'global_mapper.pt',
                                             subfolder='checkpoints',
                                             repo_type='model',
                                             token=HF_TOKEN)
        local_mapper_path = hf_hub_download('ELITE-library/ELITE',
                                            'local_mapper.pt',
                                            subfolder='checkpoints',
                                            repo_type='model',
                                            token=HF_TOKEN)
        return global_mapper_path, local_mapper_path

    def load_model(
        self,
        scheduler_type=LMSDiscreteScheduler
    ) -> tuple[UNet2DConditionModel, CLIPTextModel, CLIPTokenizer,
               AutoencoderKL, CLIPVisionModel, Mapper, MapperLocal,
               LMSDiscreteScheduler, ]:
        diffusion_model_id = 'CompVis/stable-diffusion-v1-4'

        vae = AutoencoderKL.from_pretrained(
            diffusion_model_id,
            subfolder='vae',
            torch_dtype=torch.float16,
        )

        tokenizer = CLIPTokenizer.from_pretrained(
            'openai/clip-vit-large-patch14',
            torch_dtype=torch.float16,
        )
        text_encoder = CLIPTextModel.from_pretrained(
            'openai/clip-vit-large-patch14',
            torch_dtype=torch.float16,
        )
        image_encoder = CLIPVisionModel.from_pretrained(
            'openai/clip-vit-large-patch14',
            torch_dtype=torch.float16,
        )

        # Load models and create wrapper for stable diffusion
        for _module in text_encoder.modules():
            if _module.__class__.__name__ == 'CLIPTextTransformer':
                _module.__class__.__call__ = inj_forward_text

        unet = UNet2DConditionModel.from_pretrained(
            diffusion_model_id,
            subfolder='unet',
            torch_dtype=torch.float16,
        )
        inj_forward_crossattention
        mapper = Mapper(input_dim=1024, output_dim=768)

        mapper_local = MapperLocal(input_dim=1024, output_dim=768)

        for _name, _module in unet.named_modules():
            if _module.__class__.__name__ == 'CrossAttention':
                if 'attn1' in _name:
                    continue
                _module.__class__.__call__ = inj_forward_crossattention

                shape = _module.to_k.weight.shape
                to_k_global = nn.Linear(shape[1], shape[0], bias=False)
                mapper.add_module(f'{_name.replace(".", "_")}_to_k',
                                  to_k_global)

                shape = _module.to_v.weight.shape
                to_v_global = nn.Linear(shape[1], shape[0], bias=False)
                mapper.add_module(f'{_name.replace(".", "_")}_to_v',
                                  to_v_global)

                to_v_local = nn.Linear(shape[1], shape[0], bias=False)
                mapper_local.add_module(f'{_name.replace(".", "_")}_to_v',
                                        to_v_local)

                to_k_local = nn.Linear(shape[1], shape[0], bias=False)
                mapper_local.add_module(f'{_name.replace(".", "_")}_to_k',
                                        to_k_local)

        #global_mapper_path, local_mapper_path = self.download_mappers()
        global_mapper_path = submodule_dir / 'checkpoints/global_mapper.pt'
        local_mapper_path = submodule_dir / 'checkpoints/local_mapper.pt'

        mapper.load_state_dict(
            torch.load(global_mapper_path, map_location='cpu'))
        mapper.half()

        mapper_local.load_state_dict(
            torch.load(local_mapper_path, map_location='cpu'))
        mapper_local.half()

        for _name, _module in unet.named_modules():
            if 'attn1' in _name:
                continue
            if _module.__class__.__name__ == 'CrossAttention':
                _module.add_module(
                    'to_k_global',
                    mapper.__getattr__(f'{_name.replace(".", "_")}_to_k'))
                _module.add_module(
                    'to_v_global',
                    mapper.__getattr__(f'{_name.replace(".", "_")}_to_v'))
                _module.add_module(
                    'to_v_local',
                    getattr(mapper_local, f'{_name.replace(".", "_")}_to_v'))
                _module.add_module(
                    'to_k_local',
                    getattr(mapper_local, f'{_name.replace(".", "_")}_to_k'))

        vae.eval().to(self.device)
        unet.eval().to(self.device)
        text_encoder.eval().to(self.device)
        image_encoder.eval().to(self.device)
        mapper.eval().to(self.device)
        mapper_local.eval().to(self.device)

        scheduler = scheduler_type(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule='scaled_linear',
            num_train_timesteps=1000,
        )
        return (vae, unet, text_encoder, tokenizer, image_encoder, mapper,
                mapper_local, scheduler)

    def prepare_data(self,
                     image: PIL.Image.Image,
                     mask: PIL.Image.Image,
                     text: str,
                     placeholder_string: str = 'S') -> dict[str, Any]:
        data: dict[str, Any] = {}

        data['text'] = text

        placeholder_index = 0
        words = text.strip().split(' ')
        for idx, word in enumerate(words):
            if word == placeholder_string:
                placeholder_index = idx + 1

        data['index'] = torch.tensor(placeholder_index)

        data['input_ids'] = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors='pt',
        ).input_ids[0]

        image = image.convert('RGB')
        mask = mask.convert('RGB')
        mask = np.array(mask) / 255.0

        image_np = np.array(image)
        object_tensor = image_np * mask
        data['pixel_values'] = process(image_np)

        ref_object_tensor = PIL.Image.fromarray(
            object_tensor.astype('uint8')).resize(
                (224, 224), resample=PIL.Image.Resampling.BICUBIC)
        ref_image_tenser = PIL.Image.fromarray(
            image_np.astype('uint8')).resize(
                (224, 224), resample=PIL.Image.Resampling.BICUBIC)
        data['pixel_values_obj'] = get_tensor_clip()(ref_object_tensor)
        data['pixel_values_clip'] = get_tensor_clip()(ref_image_tenser)

        ref_seg_tensor = PIL.Image.fromarray(mask.astype('uint8') * 255)
        ref_seg_tensor = get_tensor_clip(normalize=False)(ref_seg_tensor)
        data['pixel_values_seg'] = F.interpolate(ref_seg_tensor.unsqueeze(0),
                                                 size=(128, 128),
                                                 mode='nearest').squeeze(0)

        device = torch.device('cuda:0')
        data['pixel_values'] = data['pixel_values'].to(device)
        data['pixel_values_clip'] = data['pixel_values_clip'].to(device).half()
        data['pixel_values_obj'] = data['pixel_values_obj'].to(device).half()
        data['pixel_values_seg'] = data['pixel_values_seg'].to(device).half()
        data['input_ids'] = data['input_ids'].to(device)
        data['index'] = data['index'].to(device).long()

        for key, value in list(data.items()):
            if isinstance(value, torch.Tensor):
                data[key] = value.unsqueeze(0)

        return data

    @torch.inference_mode()
    def run(
        self,
        image: dict[str, PIL.Image.Image],
        text: str,
        seed: int,
        guidance_scale: float,
        lambda_: float,
        num_steps: int,
    ) -> PIL.Image.Image:
        data = self.prepare_data(image['image'], image['mask'], text)

        uncond_input = self.tokenizer(
            [''] * data['pixel_values'].shape[0],
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            return_tensors='pt',
        )
        uncond_embeddings = self.text_encoder(
            {'input_ids': uncond_input.input_ids.to(self.device)})[0]

        if seed == -1:
            seed = random.randint(0, 1000000)
        generator = torch.Generator().manual_seed(seed)
        latents = torch.randn(
            (data['pixel_values'].shape[0], self.unet.in_channels, 64, 64),
            generator=generator,
        )

        latents = latents.to(data['pixel_values_clip'])
        self.scheduler.set_timesteps(num_steps)
        latents = latents * self.scheduler.init_noise_sigma

        placeholder_idx = data['index']

        image = F.interpolate(data['pixel_values_clip'], (224, 224),
                              mode='bilinear')
        image_features = self.image_encoder(image, output_hidden_states=True)
        image_embeddings = [
            image_features[0],
            image_features[2][4],
            image_features[2][8],
            image_features[2][12],
            image_features[2][16],
        ]
        image_embeddings = [emb.detach() for emb in image_embeddings]
        inj_embedding = self.mapper(image_embeddings)

        inj_embedding = inj_embedding[:, 0:1, :]
        encoder_hidden_states = self.text_encoder({
            'input_ids':
            data['input_ids'],
            'inj_embedding':
            inj_embedding,
            'inj_index':
            placeholder_idx,
        })[0]

        image_obj = F.interpolate(data['pixel_values_obj'], (224, 224),
                                  mode='bilinear')
        image_features_obj = self.image_encoder(image_obj,
                                                output_hidden_states=True)
        image_embeddings_obj = [
            image_features_obj[0],
            image_features_obj[2][4],
            image_features_obj[2][8],
            image_features_obj[2][12],
            image_features_obj[2][16],
        ]
        image_embeddings_obj = [emb.detach() for emb in image_embeddings_obj]

        inj_embedding_local = self.mapper_local(image_embeddings_obj)
        mask = F.interpolate(data['pixel_values_seg'], (16, 16),
                             mode='nearest')
        mask = mask[:, 0].reshape(mask.shape[0], -1, 1)
        inj_embedding_local = inj_embedding_local * mask

        for t in tqdm.auto.tqdm(self.scheduler.timesteps):
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            noise_pred_text = self.unet(latent_model_input,
                                        t,
                                        encoder_hidden_states={
                                            'CONTEXT_TENSOR':
                                            encoder_hidden_states,
                                            'LOCAL': inj_embedding_local,
                                            'LOCAL_INDEX':
                                            placeholder_idx.detach(),
                                            'LAMBDA': lambda_
                                        }).sample
            latent_model_input = self.scheduler.scale_model_input(latents, t)

            noise_pred_uncond = self.unet(latent_model_input,
                                          t,
                                          encoder_hidden_states={
                                              'CONTEXT_TENSOR':
                                              uncond_embeddings,
                                          }).sample
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        _latents = 1 / 0.18215 * latents.clone()
        images = self.vae.decode(_latents).sample
        return th2image(images[0])
