# Relatório reproduzível

Os PDFs em `output/aerial/pdf/` e `output/pdf/` resumem, respectivamente, o experimento AerialYield e o primeiro treino curto AgRob. Os JSONs, o histórico CSV e as figuras publicados junto deles permitem conferir cada número. Nenhum resultado autoriza um braço robótico a se mover; dados brutos e pesos permanecem fora do Git.

## Reproduzir a comparação AerialYield

Instale a dependência opcional do relatório com `pip install -e '.[report]'`. Faça o download verificado e prepare a base com `fruit-harvest data fetch-aerial` e `fruit-harvest data prepare-aerial --ripe-stage red`. O [arquivo de auditoria](../output/aerial/metrics/source_split_audit.json) registra as 312 imagens repetidas entre pastas do ZIP YOLO e as duas etiquetas idênticas removidas.

As configurações sob `configs/aerial-cpu*.yaml` registram três ajustes em CPU: 320 px com `class_weight_power` 0,25; 320 px com 0,75; e 512 px com 0,50. Cada execução usa AdamW, weight decay 0,0005 e taxa de aprendizado cossenoidal. Treine cada perfil com um nome de execução diferente e avalie **somente a validação** antes de escolher um checkpoint. `scripts/compare_aerial_runs.py` registra a regra de escolha e as métricas dos três relatórios em `output/aerial/metrics/validation_comparison.json`. Avalie o teste uma vez com o checkpoint selecionado. Caso a validação não tenha uma política, use `--detection-only`; o comando não gera decisão de colheita.

Para regenerar os exemplos licenciados e o PDF depois das avaliações:

```bash
python scripts/make_examples.py \
  --data data/aerial-processed/dataset.yaml \
  --weights artifacts/runs/aerial-adamw-highres/weights/best.pt \
  --split test --image-size 512 --confidence 0.05 \
  --credit "Afeefa Azam, AerialYield-T2M" \
  --source-url https://zenodo.org/records/22071809 \
  --license-url https://creativecommons.org/licenses/by/4.0/ \
  --output output/aerial/examples

python scripts/build_report.py --experiment aerial --output output/aerial \
  --author "Marco Antônio Vasconcelos Freitas Filho" \
  --public-details "Engenharia da computação | Centro de Informática-UFPE | GitHub: mavff" \
  --examples output/aerial/examples --examples-credit "Afeefa Azam"
```

O gerador lê os registros locais, copia métricas públicas sem caminhos dos pesos e produz curvas, matriz de confusão, comparação e PDF. Os caminhos dos três treinos e suas avaliações aparecem no [guia de experimentos](experiments.md).

## Reproduzir o primeiro experimento AgRob

Instale a dependência opcional do relatório com `pip install -e '.[report]'`. Depois de baixar e preparar AgRobTomato conforme o README:

```bash
fruit-harvest train --data data/agrob-processed/dataset.yaml \
  --config configs/agrob-smoke.yaml --name smoke-adamw-cosine

fruit-harvest evaluate --split val \
  --weights artifacts/runs/smoke-adamw-cosine/weights/best.pt \
  --data data/agrob-processed/dataset.yaml --config configs/agrob-smoke.yaml \
  --output artifacts/evaluation/smoke-adamw-validation-report

fruit-harvest evaluate --split test --detection-only \
  --weights artifacts/runs/smoke-adamw-cosine/weights/best.pt \
  --data data/agrob-processed/dataset.yaml --config configs/agrob-smoke.yaml \
  --output artifacts/evaluation/smoke-adamw-test

python scripts/build_report.py --experiment agrob \
  --author "Marco Antônio Vasconcelos Freitas Filho" \
  --public-details "Engenharia da computação | Centro de Informática-UFPE | GitHub: mavff" \
  --examples output/examples \
  --examples-credit "Sandro Augusto Magalhães, Germano Moreira, Filipe Neves dos Santos e Mário Cunha"
```

Cada comando de treino/avaliação exige um diretório de saída novo. A configuração do smoke limita a execução a 0,04 hora; a quantidade de épocas pode variar entre computadores. Para um experimento de qualidade, use um novo nome e a configuração de treino mais longa.

## Como ler os números

- **Precisão, recall e mAP de detecção** vêm do avaliador do Ultralytics. mAP@0,5 mede localização/classificação sob IoU 0,5; mAP@0,5:0,95 resume limiares mais exigentes.
- **Acurácia, precisão, recall, macro F1 e matriz de confusão de classe** usam apenas frutas cujas caixas foram pareadas espacialmente com IoU ≥ 0,5. Acurácia alta neste grupo pode esconder erros nas classes raras e não inclui caixas perdidas nem previsões extras.
- **Previsões sem correspondência** são contadas pelo avaliador do projeto com confiança mínima 0,05. Esse corte difere da forma como o Ultralytics agrega suas métricas de detecção.
- **Métricas de colheita** não estão disponíveis nos experimentos publicados: nenhuma fruta madura foi classificada corretamente na validação. O teste com `--detection-only` não seleciona um limiar no conjunto de teste nem autoriza o braço robótico.

O repositório identifica as contribuições de maneira transparente: o proprietário definiu o objetivo e o escopo; a implementação inicial teve assistência do OpenAI Codex. A [API oficial do Zenodo](https://zenodo.org/api/records/5596799) registra licença CC BY 4.0 para AgRobTomato. Apenas dois recortes anotados são publicados, com [crédito, links e modificações](../output/examples/ATTRIBUTION.md).
