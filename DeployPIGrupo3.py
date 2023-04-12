#from sklearn import *
import pandas as pd
import numpy as np
#from sklearn import tree
import seaborn as sns
from matplotlib import pyplot as plt
#%matplotlib inline
sns.set_style("whitegrid")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import streamlit as st

from sklearn.preprocessing import StandardScaler

from joblib import dump, load

from scipy import stats
from scipy import sparse

import warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')
#print("No Warning Shown")


# Import surprise modules
from surprise import Dataset
from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
from surprise import SVD, SVDpp, NMF, SlopeOne, CoClustering, KNNBaseline, KNNWithZScore, KNNWithMeans, KNNBasic, BaselineOnly, NormalPredictor
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from lightfm import LightFM
from lightfm.evaluation import precision_at_k

from joblib import dump, load
import os


#########################################################################################################

st.sidebar.image('FourShowLogo.png')

#st.title('Projeto Integrador\n',style='text-align: center)

st.markdown("<h1 style='text-align: center; color: darkblue;'>Sistema de Recomendação de Artistas</h1>", unsafe_allow_html=True)
#st.markdown('Recomendação de Artistas\n')




##########################################################################################################


#Importar Base Proposta
Base_Propostas = pd.read_excel(r"BASE_PROPOSTAS_GRANDE_SP.xlsx")

substituto = {'Qualidade Musical': 'Música Aprovada pelo público', 
                   'Material amador': 'Autoria do Artista', 
                   'Material amador com boa qualidade': 'Autoria do Artista',
                   'Material Profissional': 'Música Aprovada pelo público',
                   'Músicas Internacionais': 'Músicas Internacionais',
                   'Violão/Guitarra': 'Instrumental',
                   'Acústico': 'Voz e violão',
                   'Para dançar': 'Música Suave',
                   'Vídeo fora do youtube': 'Sucesso do Cinema',
                   'Para cantar junto': 'Sucesso de Todos os Tempos',
                   'Músicas Autorais': 'Autoria do Artista',
                   'Para relaxar':'Música Suave',
                   'Animação':'Músicas para início de evento',
                   'Presença de palco':'Músicas para início de evento',
                   'Teclado/Piano':'Instrumental',
                   'Show ao vivo':'Música Aprovada pelo público',
                   'Sax/Metais':'Instrumental',
                   'Material em casamentos':'Sucesso de Todos os Tempos',
                   'Elegante':'Música Suave',
                   'Sanfona':'Instrumental',
                   'Violino/Violoncelo':'Instrumental',
                   'Material em eventos grandes':'Música Aprovada pelo público',
                   'Instrumental':'Instrumental',
                   'Romântico':'Música Suave',
                   'Material com qualidade ruim':'Música rejeitada pelo público',
                   'Modão/Sertanejo Raiz ':'Músicas em Português',
                   'Cover/Tributo':'Músicas Internacionais',
                   'Projeto Famoso':'Música Aprovada pelo público',
                   'Músicas Típicas':'Músicas em Português',
                   'música aprovada pelo público':'Música Aprovada pelo público',
                   '':'Música Aprovada pelo público'
                  }  
                  
Base_Propostas["ESTILO_SECUNDARIO"] = Base_Propostas["ESTILO_SECUNDARIO"].replace(substituto)

#Criar uma base apenas com propostas "Aceita", "Checkin Realizado", "Checkout Realizado" e "Confirmada"
Base_Propostas_Aceitas = Base_Propostas.loc[Base_Propostas['DESCRICAO'] == "Aceita"]
Base_Propostas_Checkin = Base_Propostas.loc[Base_Propostas['DESCRICAO'] == "Checkin Realizado"]
Base_Propostas_Checkout = Base_Propostas.loc[Base_Propostas['DESCRICAO'] == "Checkout Realizado"]
Base_Propostas_Confirmada = Base_Propostas.loc[Base_Propostas['DESCRICAO'] == "Confirmada"]
Base_Propostas_Resumo_Aceitas = pd.concat([Base_Propostas_Aceitas, Base_Propostas_Checkin,Base_Propostas_Checkout,Base_Propostas_Confirmada])

#Retirando as linhas com Estilos_Secundarios nulos
Base_Propostas_Resumo_Aceitas['ESTILO_SECUNDARIO_NULL'] = Base_Propostas_Resumo_Aceitas['ESTILO_SECUNDARIO'].isnull()
Base_Propostas_Resumo_Aceitas = Base_Propostas_Resumo_Aceitas.loc[(Base_Propostas_Resumo_Aceitas['ESTILO_SECUNDARIO_NULL'] == False)]

#Retirando as linhas com Estilos_Principais nulos
Base_Propostas_Resumo_Aceitas['ESTILO_PRINCIPAL_NULL'] = Base_Propostas_Resumo_Aceitas['ESTILO_PRINCIPAL'].isnull()
Base_Propostas_Resumo_Aceitas = Base_Propostas_Resumo_Aceitas.loc[(Base_Propostas_Resumo_Aceitas['ESTILO_PRINCIPAL_NULL'] == False)]

###Acrescentar o campo de Formação
Base_Propostas_Resumo_Aceitas["VALOR_LIQUIDO"] = pd.to_numeric(Base_Propostas_Resumo_Aceitas.VALOR_LIQUIDO, errors='coerce')
def formacao(x):
    if x < 350:
        return 'Solo'
    elif x < 549:
        return 'Dupla'
    elif x < 849:
        return 'Trio'
    else:
        return 'Banda'
       
Base_Propostas_Resumo_Aceitas['FORMACAO'] = Base_Propostas_Resumo_Aceitas.apply(lambda x: formacao(x['VALOR_LIQUIDO']), axis=1)

###CONTAGEM SHOW
TB_QTD_SHOW = Base_Propostas_Resumo_Aceitas.groupby(['NOME_BAR', 'NOME_ARTISTA']).nunique()

#DataSet com as infos das Propostas
TB_QTD_SHOW=TB_QTD_SHOW.drop([
    #'PROPOSTAS_ID'
    'DATA_INICIO',
    'DATA_FIM',
    'VALOR_BRUTO',
    'VALOR_LIQUIDO',
    'DESCRICAO',
    'LATITUDE_BAR',
    'LONGITUDE_BAR',
    'CIDADE_BAR',
    'CEP_ARTISTA',
    'LOGRADOURO_ARTISTA',
    'CIDADE_ARTISTA',
    'UF_ARTISTA',
    'ESTILO_SECUNDARIO',
    'ESTILO_PRINCIPAL',
    'FORMACAO_PRINCIPAL_ARTISTA',
    'AVALIACOES',
    'ESTILO_SECUNDARIO_NULL',
    'ESTILO_PRINCIPAL_NULL',
    'FORMACAO'],axis=1)

TB_QTD_SHOW = TB_QTD_SHOW.rename(columns ={"PROPOSTAS_ID":"QTD_SHOW"})

###BASE BARES E TAG
Base_Propostas_Resumo_Aceitas_2 = Base_Propostas_Resumo_Aceitas.merge(TB_QTD_SHOW, on=['NOME_BAR','NOME_ARTISTA'], how='left')

#Eliminando  QTD_SHOW de Shown nos bares
Base_Propostas_Resumo_Aceitas_2 = Base_Propostas_Resumo_Aceitas_2.loc[(Base_Propostas_Resumo_Aceitas_2['QTD_SHOW'] >= 5)]

# Tabela Bares e Tag
Base_Propostas_Resumo_Aceitas_2_Pivot= Base_Propostas_Resumo_Aceitas_2.pivot_table(index = ('NOME_BAR','ESTILO_SECUNDARIO'), 
                 aggfunc = {'NOME_BAR':'count'}
                )
                
# Tabela Bares e Estilo
Base_Propostas_Resumo_Aceitas_3_Pivot= Base_Propostas_Resumo_Aceitas_2.pivot_table(index = ('NOME_BAR','ESTILO_PRINCIPAL'), 
                 aggfunc = {'NOME_BAR':'count'}
                )
                
# Tabela Bares, Estilo e TAG
Base_Propostas_Resumo_Aceitas_4_Pivot= Base_Propostas_Resumo_Aceitas_2.pivot_table(index = ('NOME_BAR','ESTILO_PRINCIPAL','ESTILO_SECUNDARIO'), 
                 aggfunc = {'NOME_BAR':'count'}
                )              
                
                
### Encoding
le = LabelEncoder()
Base_Propostas_Resumo_Aceitas_2['NUM_ESTILO_SECUNDARIO'] = le.fit_transform(Base_Propostas_Resumo_Aceitas_2['ESTILO_SECUNDARIO'])
Base_Propostas_Resumo_Aceitas_2['NUM_ESTILO_PRINCIPAL'] = le.fit_transform(Base_Propostas_Resumo_Aceitas_2['ESTILO_PRINCIPAL'])
Base_Propostas_Resumo_Aceitas_2['NUM_FORMACAO'] = le.fit_transform(Base_Propostas_Resumo_Aceitas_2['FORMACAO'])

#DataSet com as infos das Propostas
Base_Propostas_Resumo_Aceitas_3=Base_Propostas_Resumo_Aceitas_2.drop([
    #'NOME_BAR',
    'PROPOSTAS_ID',
    #'NOME_ARTISTA',
    'DATA_INICIO',
    'DATA_FIM',
    'VALOR_BRUTO',
    'VALOR_LIQUIDO',
    'DESCRICAO',
    'LATITUDE_BAR',
    'LONGITUDE_BAR',
    'CIDADE_BAR',
    'CEP_ARTISTA',
    'LOGRADOURO_ARTISTA',
    'CIDADE_ARTISTA',
    'UF_ARTISTA',   
    #'ESTILO_SECUNDARIO',
    #'ESTILO_PRINCIPAL',
    'FORMACAO_PRINCIPAL_ARTISTA',
    'ESTILO_SECUNDARIO_NULL',
    'ESTILO_PRINCIPAL_NULL',
    #'FORMACAO',
    'QTD_SHOW',
    #'AVALIACOES',
    #'NUM_ESTILO_SECUNDARIO',   
    #'NUM_ESTILO_PRINCIPAL',
    #'NUM_FORMACAO'
    ],axis=1)
    
    
###Scaling
from sklearn.preprocessing import MinMaxScaler
Funscaling = MinMaxScaler()
Base_Propostas_Resumo_Aceitas_3[['AVALIACAO_ARTISTA','NUM_ESTILO_SECUNDARIO','NUM_ESTILO_PRINCIPAL','NUM_FORMACAO']] =Funscaling.fit_transform(Base_Propostas_Resumo_Aceitas_3[['AVALIACOES','NUM_ESTILO_SECUNDARIO','NUM_ESTILO_PRINCIPAL','NUM_FORMACAO']])



##Pontuação para o Modelo
# Base final para usar no modelo - com a criação de uma pontuação 
Base_Propostas_Resumo_Aceitas_3['SCORE'] = (Base_Propostas_Resumo_Aceitas_3['NUM_ESTILO_SECUNDARIO'] * 0.25 + Base_Propostas_Resumo_Aceitas_3['NUM_ESTILO_PRINCIPAL'] * 0.65 + Base_Propostas_Resumo_Aceitas_3['AVALIACAO_ARTISTA'] *0.10 )*10
Base_Propostas_Resumo_Aceitas_3 = Base_Propostas_Resumo_Aceitas_3.sort_values(['SCORE'], ascending=False)

#Tabela dimamica com os dos bares, artistas e pontuação
Base_Bares_bar_Resumo_Pivot2= Base_Propostas_Resumo_Aceitas_3.pivot_table(index =( 'NOME_BAR', 'NOME_ARTISTA'),
                 aggfunc = {'SCORE':'mean'})
                 
# Tranformar a tabela dinamica em um dataframe
Base_Bares_bar_Resumo_5=pd.DataFrame(Base_Bares_bar_Resumo_Pivot2.to_records())
                

##########################################################################################################################################       

# Função para criar um dicionário de bares com base no seu índice
# Criar um dataframe matricial 
MatrizInteracao = Base_Bares_bar_Resumo_5.groupby(['NOME_BAR', 'NOME_ARTISTA'])['SCORE'].sum().unstack().reset_index().fillna(0).set_index('NOME_BAR')

bar_id = list(MatrizInteracao.index)
bar_dict = {}
counter = 0 
for i in bar_id:
    bar_dict[i] = counter
    counter += 1

# Função para criar um dicionário de artistas com base no seu índice
Base_Bares_bar_Resumo_5 = Base_Bares_bar_Resumo_5.reset_index()
artista_dict ={}
for i in range(Base_Bares_bar_Resumo_5.shape[0]):
    artista_dict[(Base_Bares_bar_Resumo_5.loc[i,'NOME_ARTISTA'])] = Base_Bares_bar_Resumo_5.loc[i,'NOME_ARTISTA']
    
# Function to run matrix-factorization algorithm
#x = sparse.csr_matrix(MatrizInteracao.values)
#modelLFM = LightFM(no_components= 150, loss='warp')
#modelLFM.fit(x,epochs=3000,num_threads = 4)

# Evaluate the trained model
#k = 10
#st.write('Train precision at k={}:\t{:.4f}'.format(k, precision_at_k(modelLFM, x, k=k).mean()))



#Função para realizar as recomendações dos artistas
def recomendacao_artistas(model, matriz, bar_id, bar_dict, 
                               artista_dict,threshold = 0,nrec_artista = 10, show = True):

    n_bar, n_artista = matriz.shape
    bar_x = bar_dict[bar_id]
   
    scores = pd.Series(model.predict(bar_x,np.arange(n_artista)))
    scores.index = matriz.columns
    

    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    
    known_artista = list(pd.Series(matriz.loc[bar_id,:] \
                                 [matriz.loc[bar_id,:] > threshold].index) \
                                .sort_values(ascending=False))
    
    
    scores = [x for x in scores if x not in known_artista]
    return_score_list = scores[0:nrec_artista]
    
   
    known_items = list(pd.Series(known_artista).apply(lambda x: artista_dict[x]))
    scores = list(pd.Series(return_score_list).apply(lambda x: artista_dict[x]))
    if show == True:
        st.write("Artistas Conhecidos:")
        counter = 1
        for i in known_artista:
            #print(i)
            st.success(str(counter) + '- ' + i)
            counter+=1

        st.write("\n Artistas Recomendados:")
        counter = 1
        for i in scores:
            #print(i)
            st.success(str(counter) + '- ' + i)
            counter+=1
    return return_score_list


#st.subheader('Modelo - Recomendação de Artistas\n')  
lista_bares = list(Base_Bares_bar_Resumo_5['NOME_BAR'].unique())
Bares = st.sidebar.selectbox('Selecione o bar', options = lista_bares)
#dados23 = interactions.drop(['NOME_BAR'],axis=1)


if (os.path.exists('modelo_preditivo_artistas_para_bares.pk1')):
    modePersistidolLFM = load('modelo_preditivo_artistas_para_bares.pk1')
    botao = st.sidebar.button('Efetuar previsão')
    if(botao):
        df_bares = MatrizInteracao.query('NOME_BAR == @Bares')
        st.subheader('Artistas recomendados para o bar {0}'.format(Bares))
        recomendacao_artistas(model = modePersistidolLFM, 
                                      matriz = MatrizInteracao, 
                                      bar_id = 'amic-739', 
                                      bar_dict = bar_dict,
                                      artista_dict = artista_dict, 
                                      threshold = 4,
                                      nrec_artista = 10,
                                      show = True)


