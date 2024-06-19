# PUC Minas – 17/04/2024
# Treinamento para gerar um modelo de detecção de imagem criado por IA
# Desenvolvedor: Weverson Euzébio Forbes Silva

# Importação de Bibliotecas, Componentes e Configurações

import os
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils.config import cfg  # Importa a configuração
from utils.datasets import create_dataloader
from utils.earlystop import EarlyStopping
from utils.eval import get_val_cfg, validate
from utils.trainer import Trainer
from utils.utils import Logger

# Verifica se o script está sendo executado diretamente
if __name__ == "__main__":
    # Obtém a configuração de validação
    val_cfg = get_val_cfg(cfg, split="val", copy=True)
    # Define o caminho do conjunto de dados de treinamento
    cfg.dataset_root = os.path.join(cfg.dataset_root, "train")
    # Cria o dataloader
    data_loader = create_dataloader(cfg)
    # Calcula o tamanho do conjunto de dados
    dataset_size = len(data_loader)

    # Inicializa o logger
    log = Logger()
    log.open(cfg.logs_path, mode="a")
    log.write("Num de imagens de treinamento = %d\n" % (dataset_size * cfg.batch_size))
    log.write("Configuração:\n" + str(cfg.to_dict()) + "\n")

    # Inicializa o SummaryWriter para o treinamento e validação
    train_writer = SummaryWriter(os.path.join(cfg.exp_dir, "treino"))
    val_writer = SummaryWriter(os.path.join(cfg.exp_dir, "val"))

    # Inicializa o treinamento
    treinador = Trainer(cfg)
    # Inicializa o EarlyStopping
    parada_antecipada = EarlyStopping(patience=cfg.earlystop_epoch, delta=-0.001, verbose=True)
    for epoca in range(cfg.nepoch):
        tempo_inicio_epoca = time.time()
        iter_tempo_dados = time.time()
        iteração_epoca = 0

        # Loop pelos dados de treinamento
        for dados in tqdm(data_loader, dynamic_ncols=True):
            treinador.total_steps += 1
            iteração_epoca += cfg.batch_size

            treinador.set_input(dados)
            treinador.optimize_parameters()

            # Adiciona a perda ao tensorboard
            train_writer.add_scalar("perda", treinador.loss, treinador.total_steps)

            # Salva o modelo mais recente
            if treinador.total_steps % cfg.save_latest_freq == 0:
                log.write(
                    "salvando o modelo mais recente %s (época %d, modelo.total_steps %d)\n"
                    % (cfg.exp_name, epoca, treinador.total_steps)
                )
                treinador.save_networks("latest")

        # Salva o modelo no final da época
        if epoca % cfg.save_epoch_freq == 0:
            log.write("salvando o modelo no final da época %d, iterações %d\n" % (epoca, treinador.total_steps))
            treinador.save_networks("latest")
            treinador.save_networks(epoca)

        # Validação
        treinador.eval()
        resultados_val = validate(treinador.model, val_cfg)
        val_writer.add_scalar("AP", resultados_val["AP"], treinador.total_steps)
        val_writer.add_scalar("ACC", resultados_val["ACC"], treinador.total_steps)
        log.write(f"(Val @ época {epoca}) AP: {resultados_val['AP']}; ACC: {resultados_val['ACC']}\n")

        # Verifica a parada antecipada
        if cfg.earlystop:
            parada_antecipada(resultados_val["ACC"], treinador)
            if parada_antecipada.early_stop:
                if treinador.adjust_learning_rate():
                    log.write("Taxa de aprendizado reduzida em 10, continuando treinamento...\n")
                    parada_antecipada = EarlyStopping(patience=cfg.earlystop_epoch, delta=-0.002, verbose=True)
                else:
                    log.write("Parada antecipada.\n")
                    break
        # Verifica se é período de aquecimento
        if cfg.warmup:
            treinador.scheduler.step()
        treinador.train()
