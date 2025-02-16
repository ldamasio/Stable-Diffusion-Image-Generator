# main.py
from config import Config
from image_generator import ImageGenerator
import argparse
from pathlib import Path
import sys
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Gerador de Imagens com Stable Diffusion')
    
    # Argumentos básicos
    parser.add_argument('prompt', help='Prompt para geração da imagem')
    parser.add_argument('--output', '-o', help='Diretório de saída')
    parser.add_argument('--num-images', '-n', type=int, default=1,
                        help='Número de imagens a gerar')
    
    # Argumentos avançados
    parser.add_argument('--steps', type=int, help='Número de passos de inferência')
    parser.add_argument('--width', type=int, help='Largura da imagem')
    parser.add_argument('--height', type=int, help='Altura da imagem')
    parser.add_argument('--guidance-scale', type=float, help='Escala de orientação')
    parser.add_argument('--seed', type=int, help='Seed para reprodutibilidade')
    
    # Argumentos para modificação de imagem
    parser.add_argument('--input-image', '-i', help='Imagem de entrada para modificação')
    parser.add_argument('--strength', type=float, default=0.8,
                        help='Força da modificação (0-1)')
    
    args = parser.parse_args()
    
    try:
        # Inicializa configurações e gerador
        config = Config()
        generator = ImageGenerator(config)
        
        # Prepara metadados
        metadata = {
            'prompt': args.prompt,
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'steps': args.steps or config.DEFAULT_STEPS,
                'width': args.width or config.DEFAULT_WIDTH,
                'height': args.height or config.DEFAULT_HEIGHT,
                'guidance_scale': args.guidance_scale or config.DEFAULT_GUIDANCE_SCALE,
                'seed': args.seed
            }
        }
        
        print("Iniciando geração de imagem...")
        
        if args.input_image:
            # Modo de modificação de imagem
            images = generator.modify_image(
                args.input_image,
                args.prompt,
                strength=args.strength,
                num_images_per_prompt=args.num_images,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale
            )
            base_name = f"modified_{Path(args.input_image).stem}"
        else:
            # Modo de geração de imagem
            images = generator.generate_image(
                args.prompt,
                num_images=args.num_images,
                steps=args.steps,
                width=args.width,
                height=args.height,
                guidance_scale=args.guidance_scale,
                seed=args.seed
            )
            base_name = "generated"
        
        # Salva as imagens
        saved_paths = generator.save_images(
            images,
            base_name,
            args.output,
            metadata
        )
        
        print("\nGeração concluída com sucesso!")
        for path in saved_paths:
            print(f"Imagem salva em: {path}")
        
    except Exception as e:
        print(f"Erro: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
