"""Build public PDF, charts, and metric snapshots from a recorded experiment."""

import argparse
import csv
import json
from datetime import date
from dataclasses import dataclass
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


CLASSES = ("unripe", "semi_ripe", "ripe")
LABELS = ("Não maduro", "Intermediário", "Maduro")
COLORS = ("#147D64", "#D69B31", "#C84A4A")
REPO_URL = "https://github.com/mavff/Ripe-or-unripe-tomatoes"
@dataclass(frozen=True)
class Experiment:
    name: str
    dataset_url: str
    manifest: str
    history: str
    validation: str
    test: str
    checkpoint: str
    source_note: str
    training_note: str


EXPERIMENTS = {
    "agrob": Experiment(
        "AgRobTomato", "https://zenodo.org/records/5596799",
        "data/agrob-processed/dataset_manifest.json",
        "artifacts/runs/smoke-adamw-cosine/results.csv",
        "artifacts/evaluation/smoke-adamw-validation-report/report.json",
        "artifacts/evaluation/smoke-adamw-test/report.json",
        "artifacts/runs/smoke-adamw-cosine/weights/best.pt",
        "449 imagens; unriped → não maduro, breaking/reddish → intermediário, riped → maduro. "
        "Quadros próximos do vídeo foram mantidos em blocos de 20 (semente 42).",
        "Entrada de 320 px, batch 4 e limite de 0,04 hora em CPU.",
    ),
    "aerial": Experiment(
        "AerialYield-T2M", "https://zenodo.org/records/22071809",
        "data/aerial-processed/dataset_manifest.json",
        "artifacts/runs/aerial-adamw-highres/results.csv",
        "artifacts/evaluation/aerial-highres-validation/report.json",
        "artifacts/evaluation/aerial-test/report.json",
        "artifacts/runs/aerial-adamw-highres/weights/best.pt",
        "677 imagens; Green → não maduro, Breakers/Turning/Pink/Light Red → intermediário "
        "e somente Red (>90% vermelho) → maduro. As listas COCO oficiais definem as partições; "
        "312 cópias repetidas entre pastas no ZIP YOLO foram filtradas para impedir vazamento.",
        "Entrada de 512 px, batch 4 e limite de 0,4 hora em CPU. "
        "O checkpoint foi escolhido entre três perfis pela validação.",
    ),
}


def read_experiment(root: Path, experiment: Experiment) -> tuple[dict, list[dict], dict, dict]:
    manifest = json.loads((root / experiment.manifest).read_text())
    with (root / experiment.history).open(newline="") as stream:
        history = list(csv.DictReader(stream))
    validation = json.loads((root / experiment.validation).read_text())
    test = json.loads((root / experiment.test).read_text())
    if not history or validation["split"] != "val" or test["split"] != "test":
        raise ValueError("Missing or mismatched experiment records")
    return manifest, history, validation, test


def save_snapshot(output: Path, manifest: dict, history: list[dict], validation: dict,
                  test: dict, checkpoint: str) -> None:
    """Keep the numbers behind the PDF with a portable checkpoint path."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    for split, report in (("validation", validation), ("test", test)):
        public_report = dict(report)
        public_report["weights"] = checkpoint
        (output / f"{split}_report.json").write_text(
            json.dumps(public_report, indent=2, ensure_ascii=False) + "\n"
        )
    with (output / "training_history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)


def save_figures(output: Path, manifest: dict, history: list[dict], test: dict) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    epochs = [int(row["epoch"]) for row in history]

    figure, axes = plt.subplots(2, 1, figsize=(9, 5.2), constrained_layout=True)
    for key, label, color in (
        ("train/box_loss", "Caixa", COLORS[0]),
        ("train/cls_loss", "Classe", COLORS[1]),
        ("train/dfl_loss", "DFL", COLORS[2]),
    ):
        axes[0].plot(epochs, [float(row[key]) for row in history], marker="o", label=label, color=color)
    axes[0].set(ylabel="Perda de treino", xticks=epochs)
    axes[0].legend(ncol=3, frameon=False)
    for key, label, color in (
        ("metrics/mAP50(B)", "mAP@0,5", COLORS[0]),
        ("metrics/mAP50-95(B)", "mAP@0,5:0,95", COLORS[2]),
    ):
        axes[1].plot(epochs, [float(row[key]) for row in history], marker="o", label=label, color=color)
    axes[1].set(xlabel="Época", ylabel="mAP na validação", xticks=epochs, ylim=(0, 1))
    axes[1].legend(ncol=2, frameon=False)
    training_path = output / "training_curves.png"
    figure.savefig(training_path, dpi=180)
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(9, 3.6), constrained_layout=True)
    for index, split in enumerate(("train", "val", "test")):
        counts = [manifest["counts"][split]["objects"][name] for name in CLASSES]
        for class_index, count in enumerate(counts):
            y = index * 4 + class_index
            ax.barh(y, count, color=COLORS[class_index], height=0.72)
            ax.text(count * 1.08, y, str(count), va="center", fontsize=9)
    ax.set_yticks([index * 4 + class_index for index in range(3) for class_index in range(3)])
    ax.set_yticklabels([f"{split} - {label}" for split in ("Treino", "Validação", "Teste") for label in LABELS])
    ax.set(xscale="log", xlim=(1, 10000), xlabel="Objetos anotados (escala logarítmica)")
    ax.invert_yaxis()
    balance_path = output / "class_balance.png"
    figure.savefig(balance_path, dpi=180)
    plt.close(figure)

    matrix = [[test["matched_class_confusion"][truth][prediction] for prediction in CLASSES] for truth in CLASSES]
    figure, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax.imshow(matrix, cmap="Blues", vmin=0)
    largest = max(max(row) for row in matrix)
    for row in range(3):
        for column in range(3):
            ax.text(column, row, str(matrix[row][column]), ha="center", va="center", fontsize=12,
                    color="white" if matrix[row][column] > largest / 2 else "#17304A")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS,
           xlabel="Classe prevista", ylabel="Classe real", title="Teste: frutas com caixa correspondente")
    confusion_path = output / "test_confusion.png"
    figure.savefig(confusion_path, dpi=180)
    plt.close(figure)
    return {"training": training_path, "balance": balance_path, "confusion": confusion_path}


def save_comparison_figure(output: Path, comparison: dict) -> Path:
    """Plot validation criteria used to select the final AerialYield checkpoint."""
    labels = list(comparison["runs"])
    measures = (("detection_map50", "mAP@0,5"),
                ("matched_macro_f1", "Macro F1"),
                ("matched_ripe_recall", "Recall Red"))
    figure, ax = plt.subplots(figsize=(9, 3.6), constrained_layout=True)
    positions = range(len(labels))
    for index, (key, title) in enumerate(measures):
        x = [position + (index - 1) * 0.24 for position in positions]
        ax.bar(x, [comparison["runs"][label][key] for label in labels],
               width=0.22, color=COLORS[index], label=title)
    ax.set(xticks=list(positions), xticklabels=labels, ylim=(0, 1), ylabel="Fração na validação")
    ax.legend(ncol=3, frameon=False)
    path = output / "validation_comparison.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def styles() -> dict[str, ParagraphStyle]:
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    pdfmetrics.registerFont(TTFont("DejaVu", str(font)))
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("ProjectTitle", parent=base["Title"], fontName="DejaVu", fontSize=21,
                                leading=27, textColor=colors.HexColor("#123D3A"), spaceAfter=13),
        "section": ParagraphStyle("Section", parent=base["Heading2"], fontName="DejaVu", fontSize=13,
                                  leading=18, textColor=colors.HexColor("#147D64"), spaceBefore=13, spaceAfter=7),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName="DejaVu", fontSize=9,
                               leading=14, spaceAfter=7),
        "small": ParagraphStyle("Small", parent=base["BodyText"], fontName="DejaVu", fontSize=8,
                                leading=12, spaceAfter=5),
        "caption": ParagraphStyle("Caption", parent=base["BodyText"], fontName="DejaVu", fontSize=8,
                                  leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#4B5E5B")),
    }


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def table(rows: list[list[str]], widths: list[float], style: ParagraphStyle) -> Table:
    cells = [[paragraph(escape(str(value)), style) for value in row] for row in rows]
    result = Table(cells, colWidths=widths, hAlign="LEFT", repeatRows=1)
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDECE8")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F9F8")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#9FBDB6")),
    ]))
    return result


def pct(value: float) -> str:
    return f"{100 * value:.1f}%".replace(".", ",")


def figure(path: Path, width: float) -> Image:
    from PIL import Image as PILImage

    with PILImage.open(path) as source:
        aspect = source.height / source.width
    return Image(str(path), width=width, height=width * aspect)


def draw_footer(canvas, document, dataset_name: str) -> None:
    canvas.setFont("DejaVu", 7.5)
    canvas.setFillColor(colors.HexColor("#5F6C69"))
    canvas.drawString(45, 33, f"Ripe-or-unripe-tomatoes | {dataset_name} | relatório experimental")
    canvas.drawRightString(A4[0] - 45, 33, str(document.page))


def build_pdf(path: Path, author: str, details: str, manifest: dict, history: list[dict],
              validation: dict, test: dict, charts: dict[str, Path], examples: Path | None,
              examples_credit: str | None, experiment: Experiment,
              comparison: dict | None = None) -> None:
    s = styles()
    usable = A4[0] - 90
    path.parent.mkdir(parents=True, exist_ok=True)
    story = []
    add = story.append

    add(paragraph("Visão computacional para colheita de tomates", s["title"]))
    add(paragraph(f"<b>Autor:</b> {escape(author)}<br/><b>Dados públicos:</b> {escape(details)}<br/>"
                  f"<b>Data:</b> {date.today():%d/%m/%Y}<br/>"
                  f"<b>Repositório:</b> <link href='{REPO_URL}'>{REPO_URL}</link>", s["body"]))
    add(paragraph("Estado do projeto", s["section"]))
    policy_ready = validation["policy_status"] == "selected_on_validation"
    status_note = ("Uma política de colheita foi selecionada na validação; consulte as métricas de teste "
                   "antes de considerar qualquer uso físico." if policy_ready else
                   "Nenhuma política de colheita foi liberada, pois o modelo não identificou "
                   "corretamente frutos maduros na validação.")
    add(paragraph("O pipeline prepara dados, ajusta um detector, avalia caixas e classes e pode exportar ONNX. "
                  + status_note + " O braço robótico ainda não está integrado.", s["body"]))
    add(paragraph("Dataset e divisão", s["section"]))
    rows = [["Partição", "Imagens", "Não maduro", "Intermediário", "Maduro"]]
    for split, label in (("train", "Treino"), ("val", "Validação"), ("test", "Teste")):
        counts = manifest["counts"][split]
        rows.append([label, str(counts["images"])] + [str(counts["objects"][name]) for name in CLASSES])
    add(table(rows, [100, 75, 110, 110, 110], s["small"]))
    add(Spacer(1, 8))
    add(paragraph(f"Fonte: <link href='{experiment.dataset_url}'>{experiment.name} (Zenodo)</link>. "
                  + escape(experiment.source_note) + " Licença CC BY 4.0; imagens originais e "
                  "versões preparadas são republicadas com crédito.", s["small"]))
    add(paragraph("Modelo e autoria", s["section"]))
    add(paragraph("YOLO11n de detecção, implementado em PyTorch pelo Ultralytics. O ajuste usa pesos "
                  "pré-treinados em COCO (transfer learning), AdamW, decaimento de taxa cossenoidal e "
                  "BCE de classificação com pesos por classe. " + escape(experiment.training_note), s["body"]))
    add(paragraph(f"{escape(author)} definiu o objetivo e as decisões de escopo. A implementação e "
                  "a execução inicial receberam assistência do OpenAI Codex. Este relatório não atribui "
                  "a autoria integral do código a uma única pessoa.", s["small"]))
    add(paragraph("Conclusão provisória", s["section"]))
    add(paragraph("O software executa de ponta a ponta; estes resultados não autorizam uma decisão "
                  "física de colheita. As métricas de detecção e de classe são separadas nas páginas "
                  "seguintes. O conjunto de teste foi usado apenas para avaliação, sem ajuste de limiar.", s["body"]))
    add(paragraph("Próximos passos", s["section"]))
    add(paragraph("Ampliar os exemplos maduros, repetir o treino com orçamento maior e avaliar a "
                  "decisão de colheita no teste após selecionar a política na validação. A integração "
                  "do braço exigirá calibração e controle.", s["body"]))
    add(paragraph(f"Dataset: <link href='{experiment.dataset_url}'>{experiment.dataset_url}</link><br/>"
                  f"Código, imagens, anotações e checkpoint: <link href='{REPO_URL}'>{REPO_URL}</link>. "
                  "Os ZIPs originais acima de 100 MiB estão disponíveis na fonte Zenodo.", s["small"]))

    add(PageBreak())
    add(paragraph("Treinamento e distribuição", s["title"]))
    add(paragraph(f"Foram registradas {len(history)} épocas. O limite de tempo pode encerrar uma execução "
                  "em épocas diferentes em outro computador; os valores abaixo são desta execução.", s["body"]))
    add(figure(charts["training"], usable))
    add(paragraph("Perdas de treino e mAP na validação ao longo das épocas.", s["caption"]))
    add(Spacer(1, 15))
    add(figure(charts["balance"], usable))
    add(paragraph("A classe madura é rara em todas as partições; a escala horizontal é logarítmica.", s["caption"]))

    add(PageBreak())
    add(paragraph("Métricas e matriz de confusão", s["title"]))
    rows = [["Métrica", f"Validação ({validation['images']} img)", f"Teste ({test['images']} img)"]]
    metric_labels = (("metrics/precision(B)", "Precisão de detecção"),
                     ("metrics/recall(B)", "Recall de detecção"),
                     ("metrics/mAP50(B)", "mAP@0,5"),
                     ("metrics/mAP50-95(B)", "mAP@0,5:0,95"))
    for key, label in metric_labels:
        rows.append([label, pct(validation["detector_metrics"][key]), pct(test["detector_metrics"][key])])
    rows.append(["Acurácia entre caixas pareadas", pct(validation["matched_classification"]["accuracy"]),
                 pct(test["matched_classification"]["accuracy"])])
    rows.append(["Macro F1 entre caixas pareadas", pct(validation["matched_classification"]["macro_f1"]),
                 pct(test["matched_classification"]["macro_f1"])])
    add(table(rows, [220, 142, 143], s["small"]))
    add(Spacer(1, 8))
    ripe = test["matched_classification"]["per_class"]["ripe"]
    add(paragraph("A acurácia é condicional a caixas correspondentes com IoU ≥ 0,5; ela não mede "
                  "falsos positivos nem frutas perdidas. O macro F1 mostra o efeito do desbalanceamento. "
                  f"No teste, o recall de maduro entre caixas pareadas foi {pct(ripe['recall'])} "
                  f"em {ripe['support']} frutos maduros pareados. A amostra desta classe é pequena.", s["small"]))
    class_rows = [["Classe no teste", "Precisão", "Recall", "F1", "Suporte"]]
    for class_name, label in zip(CLASSES, LABELS):
        scores = test["matched_classification"]["per_class"][class_name]
        class_rows.append([label, pct(scores["precision"]), pct(scores["recall"]),
                           pct(scores["f1"]), str(scores["support"])])
    add(table(class_rows, [145, 90, 90, 90, 90], s["small"]))
    if "ripe" in test.get("per_class_detection", {}):
        detected_ripe = test["per_class_detection"]["ripe"]
        add(paragraph("Detecção da classe Red (inclui localização): "
                      f"precisão {pct(detected_ripe['precision'])}, "
                      f"recall {pct(detected_ripe['recall'])}, "
                      f"mAP@0,5 {pct(detected_ripe['map50'])}.", s["small"]))
    add(Spacer(1, 7))
    add(figure(charts["confusion"], usable * 0.65))
    add(paragraph("Linhas: classe real; colunas: classe prevista. "
                  f"{test['matched_objects']} caixas pareadas e "
                  f"{sum(test['unmatched_predictions'].values())} "
                  "previsões sem correspondência no corte de confiança 0,05; consulte o JSON.", s["caption"]))

    add(PageBreak())
    add(paragraph("Exemplos e limites de uso", s["title"]))
    if examples and (examples / "correct.jpg").exists() and (examples / "incorrect.jpg").exists():
        add(paragraph("Exemplos com anotações e previsões", s["section"]))
        for name, caption in (("correct.jpg", "Exemplo correto"), ("incorrect.jpg", "Exemplo incorreto")):
            add(figure(examples / name, usable * 0.8))
            add(paragraph(caption, s["caption"]))
            add(Spacer(1, 8))
        add(paragraph(f"<b>Crédito das fotografias:</b> {escape(examples_credit or '')}. "
                      f"Fonte: <link href='{experiment.dataset_url}'>{experiment.name}</link>; licença "
                      "<link href='https://creativecommons.org/licenses/by/4.0/'>CC BY 4.0</link>. "
                      "Os recortes, as caixas e as legendas foram acrescentados neste projeto.", s["small"]))
    else:
        add(paragraph("Não foram passados exemplos visuais a esta execução do gerador. Os gráficos e a "
                      "matriz de confusão são visualizações novas dos resultados numéricos.", s["body"]))

    if comparison:
        add(PageBreak())
        add(paragraph("Comparação na validação", s["title"]))
        add(paragraph("O checkpoint foi escolhido somente com dados de validação. A regra e os números "
                      "integrais estão em validation_comparison.json; o teste foi avaliado uma única vez "
                      "após a escolha.", s["body"]))
        rows = [["Execução", "mAP@0,5", "Macro F1", "Recall Red", "Política"]]
        for label, result in comparison["runs"].items():
            rows.append([label, pct(result["detection_map50"]),
                         pct(result["matched_macro_f1"]),
                         pct(result["matched_ripe_recall"]),
                         "sim" if result["policy_status"] == "selected_on_validation" else "não"])
        add(table(rows, [165, 85, 85, 85, 85], s["small"]))
        add(Spacer(1, 15))
        add(figure(charts["comparison"], usable))
        add(paragraph(f"Selecionado: {escape(comparison['selected_run'])}. Recall Red refere-se "
                      "somente a frutos com caixa pareada; leia também a matriz de confusão e o "
                      "suporte por classe.", s["caption"]))
        add(paragraph("Decisão de colheita no teste", s["section"]))
        if test["harvest_decision"]:
            decision = test["harvest_decision"]
            add(paragraph(f"Limiar escolhido na validação: {test['harvest_threshold']:.2f}. "
                          f"TP {decision['tp']}, FP {decision['fp']}, FN {decision['fn']}; "
                          f"precisão {pct(decision['precision'])}, recall {pct(decision['recall'])}, "
                          f"F1 {pct(decision['f1'])}. A integração física ainda requer calibração "
                          "e avaliação de segurança.", s["body"]))
        else:
            add(paragraph("Não houve política aprovada na validação; o teste apresenta apenas "
                          "métricas de detecção e classificação. Nenhuma recomendação de colheita "
                          "foi emitida.", s["body"]))

    document = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=45, rightMargin=45,
                                 topMargin=42, bottomMargin=49, author=author,
                                 title="Visão computacional para colheita de tomates",
                                 subject="Relatório técnico de avaliação de detector de maturação")
    footer = lambda canvas, doc: draw_footer(canvas, doc, experiment.name)
    document.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author", required=True, help="Name authorized for public PDF")
    parser.add_argument("--experiment", choices=EXPERIMENTS, default="agrob")
    parser.add_argument("--public-details", default="GitHub: mavff", help="Only details approved for publication")
    parser.add_argument("--examples", type=Path, help="Licensed correct.jpg and incorrect.jpg examples")
    parser.add_argument("--examples-credit", help="Dataset, creator and license for published photos")
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    if args.examples and not args.examples_credit:
        parser.error("--examples requires --examples-credit")
    root = Path(__file__).resolve().parents[1]
    experiment = EXPERIMENTS[args.experiment]
    manifest, history, validation, test = read_experiment(root, experiment)
    output = (root / args.output).resolve()
    save_snapshot(output / "metrics", manifest, history, validation, test,
                  experiment.checkpoint)
    charts = save_figures(output / "figures", manifest, history, test)
    comparison_path = output / "metrics/validation_comparison.json"
    comparison = (json.loads(comparison_path.read_text(encoding="utf-8"))
                  if args.experiment == "aerial" and comparison_path.exists() else None)
    if comparison:
        charts["comparison"] = save_comparison_figure(output / "figures", comparison)
    pdf_path = output / "pdf/relatorio_tecnico_tomates.pdf"
    build_pdf(pdf_path, args.author, args.public_details, manifest, history, validation, test,
              charts, args.examples, args.examples_credit, experiment, comparison)
    print(pdf_path)


if __name__ == "__main__":
    main()
