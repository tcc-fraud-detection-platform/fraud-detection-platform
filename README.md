# Plataforma Unificada para Detecção de Fraudes em Três Tipos de Dados Críticos

## Projeto Final de Graduação - Pontifícia Universidade Católica de Minas Gerais

**Autores:**
- Bernardo Ragonezi Silva Lopes
- Ian Rafael de Souza Oliveira
- Pedro Henrique Ramos Loura
- Rafaella Cristina de Oliveira Vitorino
- Weverson Euzebio Forbes Silva

**Orientador:**
- Gabriel Barbosa da Fonseca

## Introdução

Este projeto visa desenvolver uma plataforma de análise da veracidade de arquivos, utilizando algoritmos de inteligência artificial (IA) para enfrentar o desafio da desinformação na era digital.

## Contexto e Motivação

No Brasil, 40% das pessoas afirmam receber notícias falsas todos os dias, com um índice de preocupação que atinge 65% dos brasileiros. A disseminação acelerada de informações digitais aumenta a ameaça de fraudes e compromete a integridade visual e auditiva das informações.

## Objetivos

- Enfrentar desafios críticos em detecção precisa de fraudes
- Abranger detecção em texto, imagem e áudio
- Desenvolver uma resposta integrada
- Preservar a integridade da informação em ambiente digital dinâmico

## Metodologia

### Plataforma
A plataforma SaaS foi projetada para operar por meio de diversos módulos independentes ou integrados, oferecendo alta usabilidade, segurança da informação e credibilidade nos resultados.

### Análise de Fake News
- Utilização do conjunto de dados WEL-Fake
- Tokenização e remoção de "stopwords"
- Modelo pré-treinado BERT para análise e classificação textual
- Uso de Regressão Logística como classificador

### Análise de Áudios
- Entrada de áudio convertida em espectrogramas
- Modelos de Deep Learning como MobileNet, ResNet e VGG16
- Técnicas de Explainable AI

### Detecção de Fraudes em Imagens
- Arquitetura do modelo utilizando ResNet-50
- Tratamento de imagens com técnicas como desfoque Gaussiano e detecção de bordas
- Treinamento do modelo com dados otimizados

## Resultados

### Análise de Fake News
- Alta precisão na identificação correta de notícias verdadeiras e falsas
- Equilíbrio entre precisão e recall
- Alto F1 Score

### Análise de Áudio
- Resultados de testes em conjuntos de dados 'Fake or Real' e 'In the Wild'
- Acurácia inferior para 'In the Wild' devido a características distintas dos áudios

### Detecção de Imagens Falsas
- Avaliação quantitativa e qualitativa do desempenho do modelo
- Necessidade de melhorias contínuas para aumentar a precisão

## Conclusões

- Geração de relatórios
- Adoção por outras plataformas
- Desafios na integração dos módulos
- Inclusão constante de novos dados
- Refinamentos contínuos para resultados mais precisos e confiáveis

## Referências Bibliográficas

- ASLAM, F. The benefits and challenges of customization within saas cloud solutions. American Journal of Data, Information and Knowledge Management, 2023.
- CNN Brasil. 4 em cada 10 brasileiros afirmam receber fake news diariamente. [Link](https://www.cnnbrasil.com.br/nacional/4-em-cada-10-brasileiros-afirmam-receber-fake-news-diariamente/)
- HAI, H.; SAKODA, S. Saas and integration best practices. Fujitsu Scientific and Technical Journal, 2009.
- STATISTA. Share concerned about what is real and fake on the internet. [Link](https://www.statista.com/chart/18343/share-concerned-about-what-is-real-and-fake-on-the-internet/)

## Contato

Para mais informações, visite nosso [repositório no GitHub](#).

