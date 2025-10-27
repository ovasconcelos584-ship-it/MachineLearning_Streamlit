import pandas as pd
import joblib
import os
from sklearn import model_selection, preprocessing, pipeline, linear_model, metrics

#ETAPA 01 - CARREGAR DADOS
def carregar_dados(caminho_arquivo = "historicoAcademico.csv"):
    try:

        if os.path.exists(caminho_arquivo):

            df = pd.read_csv(caminho_arquivo, encoding= "latin1", sep=",")

            print("o arquivo foi carregado com sucesso!")

            return df 
        else:
            print("o arquivo não foi encontrado dentro da pasta!")

            return None
    except Exception as e:
        print("Erro inesperado ao carregar o arquivo ", e)        

        return None

    # --------- chamar a função para armazenar o resultado -----

dados = carregar_dados()

# ------- etapa 02 -- PREPARAÇAO E DIVISÃO DOS DADOS -----
# DEFINIÇÃO DE X (FEATURES) e Y (TARGET)

if dados is not None:
    print(f"\n total de registros carregados: {len(dados)}")
    print("iniciando o pepeline de treinamento")

    TARGET_COLUMN = "Status_Final"

#ETAPA 2.1 - DEFINIÇAO 
    try:
        X = dados.drop(TARGET_COLUMN, axis=1)
        y = dados[TARGET_COLUMN]

        print(f"Features (x) definidas: {list(X.columns)}")
        print(f"Features (y) definidas: {TARGET_COLUMN})")

    except KeyError:

        print(f"\n ----- Erro Critico -----")
        print(f"A coluna {TARGET_COLUMN} não foi encontrado no CSV")
        print(f"Colunas disponives: {list(dados.columns)}")
        print(f" Por Favor, ajuste  variável 'TARGET_COLUMN' e tente novamente! ")
        #se o target não for encontrado< irá encerrar o script!

    #ETAPA 2.2 - DIVISÃO ENTRE TREINO E TESTE
    print("\n --------- Dividindo dados em treino e teste.... -------")

    X_train, X_test, y_train, y_test = model_selection. train_test_split(
        X,y,
        test_size= 0.2,     #20 dos dados serão utilizados para teste
        random_state= 42,   #garantir a reprodutibilidade
        stratify=y         #Manter a proporção de aprovafos e reprovados
    )


    print(f"Dados de treino: {len(X_train)} | Dados de teste: {len(X_test)}")

    # ETAPA 3: CRIAÇAO DE PIPELIE DE ML
    # scaler -> a normalização de dados(colocano tudo na mesma escala)
    # model -> aplica o modelo de regressão logistica
    print("\ ---------- Criação de pipeline de ML.... ------")
    pipeline_model = pipeline.Pipeline([
         ('scarler', preprocessing.StandardScaler()),
         ('model', linear_model.LogisticRegression(random_state= 42))    
    ])

    #etapa 4 --> TREINAMENTO E AVALIAÇAO DE DADOS/MODELO

    print("\n -------- Treinamento de modelo ------")
    
    pipeline_model.fit(X_train, y_train)

    print("modelo treinado. Avaliando com os dados de teste...")
    y_pred = pipeline_model.predict(X_test)

    # AVALIAÇÃO DE DESEMPENHO
    accuracy = metrics.accuracy_score(y_test, y_pred)
    report = metrics.classification_report(y_test, y_pred)

    print("\n ------- relatorio de avalição geral -----")
    print(f"accuracy geral: {accuracy * 100:.2f}%")
    print("\nRelatorio de classificação detalhado:")
    print(report)

    #etapa 5: salvando o modelo

    model_filename = "modelo_previsao_desempenho.joblib"

    print(f"\nSalvandi o pipeline treinado em...{model_filename}")
    joblib.dump(pipeline_model, model_filename)

    print("Processo concluido com sucesso!")
    print(f"O arquivo '{model_filename}' está para ser utilizado!")

else:
    print("o pipeline não pode continuar pois os dados não foram carregados!")
