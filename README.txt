DEMANDA+ - Sistema de Previsão de Demanda Semanal

Como rodar:
1. Extraia a pasta do ZIP.
2. Abra a pasta no VS Code.
3. No terminal, execute:
   python -m pip install -r requirements.txt
   python -m streamlit run app.py

O aplicativo integra:
- Cadastro de produtos e base de pequeno comércio.
- Entrada manual ou por CSV/Excel.
- Previsão por média móvel, média ponderada, suavização exponencial, regressão linear e método ingênuo.
- Comparação de modelos, erro histórico, precisão aproximada e intervalo de confiança.
- KPIs, estoque, reposição, cenários, tendência, sazonalidade, anomalias, regiões e produtos.
- Aba Projeto e Estudos com requisitos, variáveis, testes, limitações e papel da IA.
- Exportação em TXT, CSV e Excel.

Observação gerencial:
A previsão apoia a decisão, mas não deve ser tratada como certeza. O gestor deve considerar contexto, capacidade produtiva, estoque, promoções, eventos, sazonalidade e qualidade dos dados.
