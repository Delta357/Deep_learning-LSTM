# Rede neural para processamento de linguagem natural para análise de sentimento utilizando - LSTM

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)
[![author](https://img.shields.io/badge/author-RafaelGallo-red.svg)](https://github.com/RafaelGallo?tab=repositories) 
[![](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-374/) 
[![](https://img.shields.io/badge/Tensorflow-orange.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Keras-red.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/CUDA-gree.svg)](https://powerbi.microsoft.com/pt-br/)
[![](https://img.shields.io/badge/Pandas-blue.svg)](https://pandas.pydata.org/) 
[![](https://img.shields.io/badge/Matplotlib-blue.svg)](https://matplotlib.org/)
[![](https://img.shields.io/badge/Seaborn-green.svg)](https://seaborn.pydata.org/)
[![](https://img.shields.io/badge/Matplotlib-orange.svg)](https://scikit-learn.org/stable/) 
[![](https://img.shields.io/badge/Numpy-white.svg)](https://numpy.org/)

![Logo](https://img.freepik.com/vetores-gratis/bandeira-de-techno-baixo-poli_1048-11799.jpg?w=1480&t=st=1693787169~exp=1693787769~hmac=3db9f6b149fd5a999eea89fb8ca893c1da4dae755ae189bc958505c1e017e7a7)

## Definição

Nos últimos anos, a pesquisa em processamento de linguagem natural (PLN) alcançou avanços notáveis, principalmente devido ao desenvolvimento de redes neurais profundas. Uma abordagem particularmente poderosa e eficaz é a utilização de Long Short-Term Memory (LSTM), um tipo de rede neural recorrente (RNN) que demonstrou excelentes resultados em uma variedade de tarefas de PLN. Uma aplicação notável dessa tecnologia é a análise de sentimentos, que envolve a identificação e classificação das emoções expressas em um texto. As redes neurais com LSTM provaram ser altamente adequadas para essa tarefa, devido à sua capacidade de capturar dependências de longo prazo nas sequências de texto. Isso significa que elas são capazes de reconhecer nuances e contextos complexos nas palavras e frases utilizadas, tornando-as ideais para determinar o sentimento geral de um texto. Essas redes neurais para análise de sentimentos têm aplicações amplas, desde a análise de opiniões de clientes em avaliações de produtos até a monitorização da satisfação do cliente em redes sociais. A capacidade de identificar automaticamente sentimentos e emoções em grande escala é inestimável para empresas que desejam compreender as opiniões e necessidades de seus clientes. Portanto, a utilização de uma rede neural para processamento de linguagem natural, especialmente aquelas que incorporam LSTM, representa um avanço significativo no campo da análise de sentimentos. Essas redes não apenas automatizam o processo de avaliação de textos, economizando tempo e recursos, como também oferecem insights valiosos que podem impulsionar decisões de negócios mais informadas e estratégias de marketing direcionadas.

## Introdução

O processamento de linguagem natural (PLN) é uma área em constante evolução que desempenha um papel vital em várias aplicações, desde chatbots até análise de sentimentos. Uma das abordagens mais eficazes para lidar com tarefas de PLN é a utilização de redes neurais, e neste estudo, exploraremos uma técnica específica: redes neurais com Long Short-Term Memory (LSTM). Nosso foco principal será a aplicação dessas redes para a análise de sentimentos em textos. A capacidade de compreender as emoções expressas em mensagens de texto é fundamental para empresas que desejam entender seus clientes, adaptar suas estratégias de marketing e melhorar a experiência do usuário.

## Metodologia

Para realizar a análise de sentimentos, implementamos uma rede neural LSTM. As LSTMs são redes neurais recorrentes que se destacam em tarefas de sequência devido à sua capacidade de capturar dependências temporais de longo prazo. Nossa metodologia incluiu as seguintes etapas Pré-processamento de dados: Coletamos uma grande quantidade de dados de texto de fontes diversas, como avaliações de produtos, posts de redes sociais e análises de clientes. Em seguida, realizamos tarefas de pré-processamento, como tokenização, remoção de stop words e normalização de texto. Construção do modelo LSTM arquitetura de rede neural com camadas LSTM, uma camada de embedding para representação de palavras e camadas de saída para a classificação de sentimentos. Treinamos o modelo em um conjunto de dados rotulados, usando técnicas de aprendizado supervisionado. Avaliação de desempenho o desempenho do modelo utilizando métricas como acurácia, precisão, recall e F1-score. Também realizamos validação cruzada para garantir a robustez do nosso modelo.
Aplicação prática e implementamos o modelo em cenários do mundo real, como a classificação de avaliações de produtos e análise de sentimentos em tempo real em plataformas de mídia social.

## Conclusão

Os resultados deste estudo demonstram a eficácia das redes neurais com LSTM na análise de sentimentos em texto. A capacidade do modelo em compreender as nuances e contextos nas palavras e frases utilizadas se mostrou fundamental para a precisão das classificações de sentimentos. A aplicação prática dessas redes neurais é vasta, abrangendo desde a identificação de opiniões de clientes em avaliações de produtos até a detecção de feedbacks negativos nas redes sociais. Empresas podem usar essas ferramentas para tomar decisões informadas, melhorar a satisfação do cliente e personalizar suas estratégias de marketing. Em resumo, a rede neural com LSTM para análise de sentimentos representa uma poderosa abordagem no campo do PLN, com amplas aplicações e a capacidade de fornecer insights valiosos para empresas e pesquisadores interessados em compreender o mundo das emoções expressas em textos.

## Stack utilizada

**Programação** Python

**Leitura CSV**: Pandas.

**Análise de dados**: Seaborn, Matplotlib, Ploty.

**Deep learning**: Rede neural LSTM 

# Dataset projetos

| Dataset              | Link                                                | 
| ----------------- | ---------------------------------------------------------------- |
|  |[Link]()|
|  |[Link]()|

# Resultados projetos


## Notebooks projetos
| Projeto              | Link projeto                                                |
| ----------------- | ---------------------------------------------------------------- |
|Projeto|[Notebook]()|

# Resultados do projeto


## Variáveis de Ambiente

Para criar e gerenciar ambientes virtuais (env) no Windows, você pode usar a ferramenta venv, que vem com as versões mais recentes do Python (3.3 e posteriores). Aqui está um guia passo a passo de como criar um ambiente virtual no Windows:

1 Abrir o Prompt de Comando: Pressione a tecla Win + R para abrir o "Executar", digite cmd e pressione Enter. Isso abrirá o Prompt de Comando do Windows.

2 Navegar até a pasta onde você deseja criar o ambiente virtual: Use o comando cd para navegar até a pasta onde você deseja criar o ambiente virtual. Por exemplo:

`cd caminho\para\sua\pasta`

3 Criar o ambiente virtual: Use o comando python -m venv nome_do_seu_env para criar o ambiente virtual. Substitua nome_do_seu_env pelo nome que você deseja dar ao ambiente. 

`python -m venv myenv`

4 Ativar o ambiente virtual: No mesmo Prompt de Comando, você pode ativar o ambiente virtual usando o script localizado na pasta Scripts dentro do ambiente virtual. Use o seguinte comando:

`nome_do_seu_env\Scripts\activate`

5 Desativar o ambiente virtual: Para desativar o ambiente virtual quando você não precisar mais dele, simplesmente digite deactivate e pressione Enter no Prompt de Comando.

`myenv\Scripts\activate`

Obs: Lembre-se de que, uma vez que o ambiente virtual está ativado, todas as instalações de pacotes usando pip serão isoladas dentro desse ambiente, o que é útil para evitar conflitos entre diferentes projetos. Para sair completamente do ambiente virtual, feche o Prompt de Comando ou digite deactivate. Lembre-se também de que essas instruções pressupõem que o Python esteja configurado corretamente em seu sistema e acessível pelo comando python. Certifique-se de ter o Python instalado e adicionado ao seu PATH do sistema. Caso prefira, você também pode usar ferramentas como o Anaconda, que oferece uma maneira mais abrangente de gerenciar ambientes virtuais e pacotes em ambientes de desenvolvimento Python.

## Pacote no anaconda
Para instalar um pacote no Anaconda, você pode usar o gerenciador de pacotes conda ou o gerenciador de pacotes Python padrão, pip, que também está disponível dentro dos ambientes do Anaconda. Aqui estão as etapas para instalar um pacote usando ambos os métodos:

## Usando o Conda
1 Abra o Anaconda Navigator ou o Anaconda Prompt, dependendo da sua preferência.

2 Para criar um novo ambiente (opcional, mas recomendado), você pode executar o seguinte comando no Anaconda Prompt (substitua nome_do_seu_ambiente pelo nome que você deseja dar ao ambiente):

`conda create --name nome_do_seu_ambiente`

3 Ative o ambiente recém-criado com o seguinte comando (substitua nome_do_seu_ambiente pelo nome do ambiente que você criou):

`conda create --name nome_do_seu_ambiente`

4 Para instalar um pacote específico, use o seguinte comando (substitua nome_do_pacote pelo nome do pacote que você deseja instalar):

`conda install nome_do_pacote`


## Instalação
Instalação das bibliotecas para esse projeto no python.

```bash
  conda install pandas 
  conda install scikitlearn
  conda install numpy
  conda install scipy
  conda install matplotlib

  python==3.6.4
  numpy==1.13.3
  scipy==1.0.0
  matplotlib==2.1.2
```
Instalação do Python É altamente recomendável usar o anaconda para instalar o python. Clique aqui para ir para a página de download do Anaconda https://www.anaconda.com/download. Certifique-se de baixar a versão Python 3.6. Se você estiver em uma máquina Windows: Abra o executável após a conclusão do download e siga as instruções. 

Assim que a instalação for concluída, abra o prompt do Anaconda no menu iniciar. Isso abrirá um terminal com o python ativado. Se você estiver em uma máquina Linux: Abra um terminal e navegue até o diretório onde o Anaconda foi baixado. 
Altere a permissão para o arquivo baixado para que ele possa ser executado. Portanto, se o nome do arquivo baixado for Anaconda3-5.1.0-Linux-x86_64.sh, use o seguinte comando: chmod a x Anaconda3-5.1.0-Linux-x86_64.sh.

Agora execute o script de instalação usando.


Depois de instalar o python, crie um novo ambiente python com todos os requisitos usando o seguinte comando

```bash
conda env create -f environment.yml
```
Após a configuração do novo ambiente, ative-o usando (windows)
```bash
activate "Nome do projeto"
```
ou se você estiver em uma máquina Linux
```bash
source "Nome do projeto" 
```
Agora que temos nosso ambiente Python todo configurado, podemos começar a trabalhar nas atribuições. Para fazer isso, navegue até o diretório onde as atribuições foram instaladas e inicie o notebook jupyter a partir do terminal usando o comando
```bash
jupyter notebook
```

## Suporte
Para suporte, mande um email para rafaelhenriquegallo@gmail.com


## Feedback
Se você tiver algum feedback, por favor nos deixe saber por meio de rafaelhenriquegallo@gmail.com
