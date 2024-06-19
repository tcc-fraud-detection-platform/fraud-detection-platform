# PUC Minas – 17/04/2024
# Modelo de detecção de imagem criado por IA
# Desenvolvedor: Weverson Euzébio Forbes Silva

# Importação de Bibliotecas, Componentes
import argparse
import io
import psycopg2
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageFilter
from utils.utils import get_network, str2bool, to_cuda
from decouple import config


DB_NAME = config('DB_NAME')
DB_USER = config('DB_USER')
DB_PASSWORD = config('DB_PASSWORD')
DB_HOST = config('DB_HOST')
DB_PORT = config('DB_PORT')

# Conexão com o banco de dados
# Aqui você configura a conexão com o banco de dados PostgreSQL usando a biblioteca psycopg2.

conn = psycopg2.connect(dbname=DB_NAME,
                        user=DB_USER,
                        password=DB_PASSWORD,
                        host=DB_HOST,
                        port=DB_PORT)
cur = conn.cursor()

# Seleciona imagens não processadas
# Esta consulta SQL seleciona todas as imagens que ainda não foram processadas e que estão no fluxo 'image_fraud'.
select_query = """
    SELECT 
        file, operation_id 
    FROM
        public.tcc_2
    WHERE
        was_processed is false 
        AND ai_flow = 'image_fraud'
"""
cur.execute(select_query)
rows = cur.fetchall()

def apply_filters(image):
    # Convertendo para o modo RGB se não estiver no modo correto
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    
    # Aplicar o filtro GaussianBlur
    image = image.filter(ImageFilter.GaussianBlur(radius=1))
  
    # Ajustar a saturação
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(0.5)
    
    # Aumentar o contraste
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    

        # Aplicar detecção de bordas
    image = image.filter(ImageFilter.FIND_EDGES)

      # Aplicar Box Blur
    image = image.filter(ImageFilter.BoxBlur(radius=2))
    
    return image



def predict_image(row):
    image_data, operation_id = row
    print(row)
    # Abre a imagem a partir dos dados binários armazenados no banco de dados.
    img = Image.open(io.BytesIO(image_data))

    if img is None:
        print(f"Não foi possível abrir a imagem para operação {operation_id}.")
        return

    # Aplicar os filtros
    img = apply_filters(img)

    # Aplica as transformações necessárias na imagem.
    img = trans(img)
    if args.aug_norm:
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    in_tens = img.unsqueeze(0)
    if not args.use_cpu:
        in_tens = in_tens.cpu()

    # Faz a predição com o modelo carregado.
    with torch.no_grad():
        prob = 1 - model(in_tens).sigmoid().item()
        if prob == 1:
            prob -= 0.015
        elif prob == 0:
            prob += 0.015

        print(f"Probabilidade de ser Real para operação {operation_id}: {prob:.4f}")
        print('<<{"result": 1 , "score": '+ str(prob) + '}>>')
        # Atualiza o banco de dados para marcar a imagem como processada e registrar a pontuação da predição.
        update_query = f"""
        UPDATE 
            public.tcc_2 
        SET 
            was_processed = true,
            score = {prob},
            is_real = 1
        WHERE
            operation_id = {operation_id}
        """
        cur.execute(update_query)
        conn.commit()

# Configura os argumentos para o script.
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "-m",
    "--model_path",
    type=str,
    default="data/exp/ckpt/model_epoch_best.pth",
)
parser.add_argument("--use_cpu", action="store_true", help="usar CPU por padrão, ative para usar GPU")
parser.add_argument("--arch", type=str, default="resnet50")
parser.add_argument("--aug_norm", type=str2bool, default=True)

args = parser.parse_args()

# Carrega o modelo de IA
model = get_network(args.arch)
state_dict = torch.load(args.model_path, map_location="cpu")
if "model" in state_dict:
    state_dict = state_dict["model"]
model.load_state_dict(state_dict)
model.eval()
if not args.use_cpu:
    model.cpu()

# Define as transformações que serão aplicadas às imagens antes de fazer a predição.
trans = transforms.Compose(
    (
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    )
)

# Itera sobre as imagens não processadas e realiza a predição.
for row in rows:
    predict_image(row)

# Fecha a conexão com o banco de dados.
cur.close()
conn.close()
