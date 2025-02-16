# Stable-Diffusion-Image-Generator
This project explores the fascinating world of generative AI by implementing a Stable Diffusion model for image generation. Users can provide text prompts to generate unique and creative images, modify existing images, or even create variations of images.

##
Projeto completo de geração de imagens usando Stable Diffusion, permitindo gerar e modificar imagens a partir de prompts em texto.

### Funcionalidades Principais:
- Geração de imagens a partir de prompts em texto
- Modificação de imagens existentes
-Suporte a múltiplas imagens por prompt
-Controle fino sobre parâmetros de geração
-Salvamento automático com metadados


### Como usar

```
# 1. Instale as dependências
pip install -r requirements.txt

# 2. Configure o token da Hugging Face (opcional)
# Crie um arquivo .env com:
HUGGINGFACE_TOKEN=seu_token_aqui

# 3. Gere imagens
python main.py "uma paisagem tropical ao pôr do sol" --num-images 3

# Exemplos de uso:
# Geração básica
python main.py "gato siamês dormindo" 

# Geração com parâmetros personalizados
python main.py "cidade futurista" --steps 75 --width 768 --height 512

# Modificação de imagem existente
python main.py "versão anime" --input-image foto.jpg --strength 0.7
```

### Recursos Avançados
-Suporte a GPU para processamento rápido
-Controle de seed para reprodutibilidade
-Prompts negativos para melhor controle
-Metadados detalhados das gerações
-Otimizações de memória

