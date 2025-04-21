Lucas de Assis Lauri - 00290411

Neste repositório tem-se a implementação do trabalho final da disciplina ECD15 MLOps do Curso de Especialização em Engenharia de Software para Aplicações de Ciência de Dados. Neste trabalho foram aplicadas as técnicas de MLOps para desenvolver um pipeline de machine learning que, de forma automatizada, faça o pré-processamento dos dados e o treinamento, avaliação, versionamento, monitoramento do modelo de predição.

O dataset escolhido foi o Telco Custumer Churn (https://www.kaggle.com/datasets/blastchar/telco-customer-churn) que consiste de diversas informações sobre clientes anônimos como gênero, número de meses em que o cliente usou os serviços da empresa, entre outros. O dataset é carregado e pré-processado pelo script 'churn.py'. Neste script diversas versões do modelo 'RandomForestGridSearch' são treinados com parâmetros de aprendizados diferentes. Estes modelos são versionados e armazenados usando o MLflow no mesmo script. O script 'churn_promote.py' é então utilizado para, de forma automatizada, decidir qual a melhor versão do modelo conforme o f1-score da classificação onde os modelos são classificados entre
    - 'archived': modelo com baixo f1-score;
    - 'staging': modelo com f1-score razoável;
    - 'production': modelo com o maior f1-score.
Com a melhor versão do modelo como 'production' podemos usar o MLflow para fazer o deploy local do modelo usando a linha de comando
    mlflow models serve -m "models:/RandomForestGridSearch/Production" --env-manager virtualenv --no-conda --port 8000
é importante notar que, neste momento, o MLflow já deve estar rodando com
    mlflow ui --backend-store-uri sqlite:///mlflow.db
e a variável de ambiente "MLFLOW_TRACKING_URI" deve estar apontando para o servidor do MLflow, no Windows isso pode ser feito pelo powershell com o comando
    $env:MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
Neste momento temos o modelo exposto localmente em 'http://127.0.0.1:8000' e, com a linha de comando,
    curl.exe -X POST "http://localhost:8000/invocations" -H "Content-Type: application/json" -d "@test.json"
é possível enviar as requisições em 'test.json' para a predição pelo modelo.

Quanto ao monitoramento e re-treinamento automático do modelo isso pode ser feito pelo script 'monitor.py' que, usando a biblioteca Evidently AI, é responsável por verificar a resposta atual do modelo a um conjunto de dados e, por meio de testes estatísticos, determinar se houve o drift de algum dado e, caso houver e ser expressivo, retreinar o modelo re-executando o script 'churn.py'.

Por fim, conclui-se que foi possível criar um fluxo automatizado e reprodutível, desde o pré-processamento dos dados até a implantação e monitoramento contínuo do modelo em produção. A abordagem adotada garantiu rastreabilidade dos experimentos, seleção automática da melhor versão do modelo com base no f1-score e detecção proativa de drift dos dados, permitindo re-treinamento quando necessário. Como trabalhos futuros, sugere-se a expansão do pipeline para ambientes de nuvem, a inclusão de mais modelos e técnicas de otimização, além da implementação de alertas automatizados para drift significativo. Este projeto consolida os princípios de MLOps, destacando a importância da automação, monitoramento e governança de modelos em cenários reais de ciência de dados.