"""
clasificador_beto.py
----------------------
Fine-tuning de BETO (BERT en español) para clasificación de sentimiento
de 3 clases (negativo / neutral / positivo).

Requiere GPU para tiempos de entrenamiento razonables.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


MODELO_BASE = "dccuchile/bert-base-spanish-wwm-uncased"
ETIQUETAS = ["negativo", "neutral", "positivo"]
ETIQUETA_A_ID = {e: i for i, e in enumerate(ETIQUETAS)}
ID_A_ETIQUETA = {i: e for i, e in enumerate(ETIQUETAS)}


class DatasetSentimiento(Dataset):
    """Envoltorio de Dataset de PyTorch para textos + etiquetas tokenizados."""

    def __init__(self, textos, etiquetas, tokenizer, max_length=128):
        # Blindaje: forzar todo a string y reemplazar nulos/vacíos por un
        # placeholder, para que el tokenizer nunca reciba NaN o tipos raros.
        textos_limpios = [str(t) if pd.notna(t) and str(t).strip() != "" else "vacio" for t in textos]

        self.encodings = tokenizer(
            textos_limpios,
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        self.labels = [ETIQUETA_A_ID[e] for e in etiquetas]

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def calcular_metricas(eval_pred):
    """Función que usa el Trainer de HuggingFace para calcular métricas en cada evaluación."""
    logits, labels = eval_pred
    predicciones = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predicciones)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicciones, average="weighted", zero_division=0
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


class ClasificadorBeto:
    """
    Envoltorio de alto nivel para fine-tunear BETO sobre un dataset de
    sentimiento de 3 clases.

    Uso típico:
        clf = ClasificadorBeto()
        clf.preparar_datos(df["texto_transformer"], df["sentimiento"])
        clf.entrenar(epochs=3)
        clf.evaluar()
        clf.matriz_confusion()
    """

    def __init__(self, modelo_base=MODELO_BASE, max_length=128, test_size=0.2, random_state=42):
        self.max_length = max_length
        self.test_size = test_size
        self.random_state = random_state

        print(f"Cargando tokenizer y modelo base: {modelo_base}")
        self.tokenizer = AutoTokenizer.from_pretrained(modelo_base)
        self.modelo = AutoModelForSequenceClassification.from_pretrained(
            modelo_base, num_labels=len(ETIQUETAS)
        )

        self.trainer = None
        self.dataset_test = None
        self.y_test = None
        self.y_pred = None

    def preparar_datos(self, textos, etiquetas):
        X_train, X_test, y_train, y_test = train_test_split(
            textos, etiquetas,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=etiquetas,
        )

        self.dataset_train = DatasetSentimiento(X_train, y_train, self.tokenizer, self.max_length)
        self.dataset_test = DatasetSentimiento(X_test, y_test, self.tokenizer, self.max_length)
        self.y_test = [ETIQUETA_A_ID[e] for e in y_test]

        print(f"Entrenamiento: {len(X_train)} ejemplos | Prueba: {len(X_test)} ejemplos")

    def entrenar(self, epochs=3, batch_size=16, learning_rate=2e-5, output_dir="./resultados_beto"):
        args_entrenamiento = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_steps=50,
            report_to="none",
        )

        self.trainer = Trainer(
            model=self.modelo,
            args=args_entrenamiento,
            train_dataset=self.dataset_train,
            eval_dataset=self.dataset_test,
            compute_metrics=calcular_metricas,
        )

        print("Iniciando entrenamiento...")
        self.trainer.train()
        print("Entrenamiento terminado.")

    def evaluar(self):
        predicciones_raw = self.trainer.predict(self.dataset_test)
        self.y_pred = np.argmax(predicciones_raw.predictions, axis=-1)

        acc = accuracy_score(self.y_test, self.y_pred)
        print(f"\nExactitud (accuracy): {acc:.4f}\n")
        print("Reporte de clasificación:")
        print(classification_report(
            self.y_test, self.y_pred,
            target_names=ETIQUETAS,
        ))
        return acc

    def matriz_confusion(self, guardar_como=None):
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=ETIQUETAS, yticklabels=ETIQUETAS)
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        plt.title("Matriz de confusión - BETO")
        plt.tight_layout()
        if guardar_como:
            plt.savefig(guardar_como)
        plt.show()

    def predecir(self, texto):
        """Predice el sentimiento de un solo texto (sin preprocesar, BETO maneja texto natural)."""
        inputs = self.tokenizer(texto, return_tensors="pt", truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.modelo.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.modelo(**inputs)
        pred_id = torch.argmax(outputs.logits, dim=-1).item()
        return ID_A_ETIQUETA[pred_id]

    def guardar_modelo(self, ruta="modelo_beto_sentimiento"):
        self.modelo.save_pretrained(ruta)
        self.tokenizer.save_pretrained(ruta)
        print(f"Modelo guardado en: {ruta}")
