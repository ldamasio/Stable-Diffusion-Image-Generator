# image_generator.py
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List, Dict
from config import Config
import json


class ImageGenerator:
    """Classe para gerar imagens usando Stable Diffusion"""
    
    def __init__(self, config: Config):
        self.config = config
        self.pipeline = self._setup_pipeline()
        
    def _setup_pipeline(self) -> StableDiffusionPipeline:
        """Configura o pipeline do Stable Diffusion"""
        # Carrega o modelo
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            cache_dir=str(self.config.CACHE_DIR),
            use_auth_token=self.config.HUGGINGFACE_TOKEN
        )
        
        # Configura o scheduler para melhor qualidade/velocidade
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        
        # Move para GPU se disponível
        if torch.cuda.is_available():
            pipeline = pipeline.to(self.config.DEVICE)
            
        # Habilita attenção em slicing para economizar memória
        pipeline.enable_attention_slicing()
        
        return pipeline
    
    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_images: int = 1,
        steps: int = None,
        width: int = None,
        height: int = None,
        guidance_scale: float = None,
        seed: Optional[int] = None
    ) -> List[Image.Image]:
        """Gera imagens a partir de um prompt"""
        try:
            # Define valores padrão
            steps = steps or self.config.DEFAULT_STEPS
            width = width or self.config.DEFAULT_WIDTH
            height = height or self.config.DEFAULT_HEIGHT
            guidance_scale = guidance_scale or self.config.DEFAULT_GUIDANCE_SCALE
            negative_prompt = negative_prompt or self.config.DEFAULT_NEGATIVE_PROMPT
            
            # Define seed para reprodutibilidade
            if seed is not None:
                torch.manual_seed(seed)
                
            # Gera as imagens
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                num_inference_steps=steps,
                width=width,
                height=height,
                guidance_scale=guidance_scale
            )
            
            return result.images
            
        except Exception as e:
            raise Exception(f"Erro na geração de imagem: {str(e)}")
    
    def modify_image(
        self,
        image: Union[str, Path, Image.Image],
        prompt: str,
        strength: float = 0.8,
        **kwargs
    ) -> List[Image.Image]:
        """Modifica uma imagem existente usando Stable Diffusion"""
        try:
            # Carrega a imagem se necessário
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert("RGB")
            
            # Redimensiona a imagem se necessário
            width = kwargs.get('width', self.config.DEFAULT_WIDTH)
            height = kwargs.get('height', self.config.DEFAULT_HEIGHT)
            if image.size != (width, height):
                image = image.resize((width, height), Image.LANCZOS)
            
            # Gera variações da imagem
            result = self.pipeline(
                prompt=prompt,
                image=image,
                strength=strength,
                **kwargs
            )
            
            return result.images
            
        except Exception as e:
            raise Exception(f"Erro na modificação de imagem: {str(e)}")
    
    def save_images(
        self,
        images: List[Image.Image],
        base_name: str,
        output_dir: Optional[Union[str, Path]] = None,
        metadata: Optional[Dict] = None
    ) -> List[Path]:
        """Salva as imagens geradas com metadados"""
        output_dir = Path(output_dir or self.config.OUTPUT_DIR)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths = []
        
        for i, image in enumerate(images):
            # Gera nome do arquivo
            filename = f"{base_name}_{timestamp}_{i+1}.png"
            filepath = output_dir / filename
            
            # Salva a imagem
            image.save(filepath, "PNG")
            
            # Salva metadados se fornecidos
            if metadata:
                metadata_file = filepath.with_suffix('.json')
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            saved_paths.append(filepath)
        
        return saved_paths