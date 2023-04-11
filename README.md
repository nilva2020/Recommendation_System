# Sistema de Recomendação de Artista para Estabelecimentos

___
![artist](img/artist.png)
___
## Contéudo
Sistema de recomendação são algoritmos que utiliza variadas técnicas computacionais combinadas que procura prever a "avaliação" e/ou "preferência" de itens ao usuário. Este sistema de recomendação são utilizados nas mais diversas áreas.

___
## Objetivo
Recomendação de Artista
___
## Objetivo Específico
Recomendação de Artistas para Bares.
___

## Propósito
    Nosso modelo de Algoritmo tem a pretensão de estimar/predizer se um Artista do nosso portfolio tem os mesmos qualitativos, features, para ser uma recomendação assertiva de continuação do projeto musical do bar.
    Nas recomendações dos Artistas para Bares ocorrerão uma renovação sutil para manter o perfil do estabelecimento e a rotatividade dos artistas.

___

### Análise Exploratória de dados:
        * Quantidade de estabelecimento;
        * Quantidade de Artistas;
        * Quantidade de Propostas;
        * Estabelecimento com maior número de propostas;
        * Estilo musical mais tocado.
___
### Insight
* Análise dos estilos musicais mais tocados:
        Top -  Estilo prinicipal e secundário.
* Análise dos melhores artistas por estilo.
* Análise de artistas similares.

___

### Dataset

![dataset](img/dataset.png)

___
### Projeto:


Sistema de recomendação:
* Filtragem baseada em conteúdo LightFM .

Essa abordagem utiliza uma série de característica discretas de um item (estabelecimento) para recomendar (artistas) com propriedades semelhantes.

Análises:

1 - Realiza a análise do estilo musical de cada estabelecimento;
2 - Realiza a análise do estilo musical de cada artista;
3 - Algoritmo prever as similaridades entre os dois itens e faz a devolutiva com a recomendação. 

![similar](img/similares.png)


### Tecnologia:

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

### Dependências:
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)


___

### Deploy
![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)

![artist](img/deploy.png)
___
### Referências Bibliograficas
KNN  - https://scikit-learn.org/stable/modules/neighbors.html

LIGHTFM - https://making.lyst.com/lightfm/docs/home.html

Sistema de recomendação: https://www.supero.com.br

___
### Código fonte:
 * Recomendador de artista com filtragem colaborativa baseada em itens KNN;
 * Recomendador de artista com  filtragem baseada em conteúdo baseada em LightFM
___
### Orientador
 * Wagner Maurício Nunes dos Santos

___
## Equipe desenvolvedora:
* Adgelson Gomes 
* Bruno Farias  
* Eduardo Iwasaki
* Nilva Pires


___
<p text-align="center">🔸Projeto Integrador - Digital House 🔸Ciências de Dados 🔸2023</p>