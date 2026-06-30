# Plano de Testes - Demanda+

| Teste | Entrada | Resultado esperado |
|---|---|---|
| Demanda estável | 100,100,100,100 | Previsão próxima de 100. |
| Crescimento | 100,110,120,130 | Previsão crescente. |
| Queda | 200,180,160,140 | Previsão decrescente. |
| Média móvel | 155,160,165 | Previsão próxima de 160. |
| Poucos dados | 100,120 | Alerta de baixa confiabilidade. |
| Valor negativo | -50 | Bloqueio ou alerta. |
| Comparação de modelos | Série válida | Indicação do menor erro histórico. |
| Gráfico | Dados válidos | Histórico e previsão exibidos. |
| Exportação | Clique nos botões | TXT, CSV e Excel baixados corretamente. |
| Celular | Abrir no smartphone | Layout legível e utilizável. |

## Debug realizado
- Compilação Python: aprovada.
- Execução direta do app.py: concluída sem erro fatal.
- Dependências revisadas: streamlit, pandas, numpy, plotly, scikit-learn e openpyxl.
