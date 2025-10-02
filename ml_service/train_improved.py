#!/usr/bin/env python3
"""
УЛУЧШЕННЫЙ TRAINING SCRIPT - Повышение точности sentiment
"""

import json
import logging
import pickle
from pathlib import Path
from typing import List, Tuple
import warnings
from collections import Counter

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score, hamming_loss
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss для борьбы с дисбалансом классов"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedTrainer(Trainer):
    """Trainer с Focal Loss"""
    
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.focal_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss


class ReviewDataset(Dataset):
    """Датасет для обучения"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float if isinstance(self.labels[idx], (list, np.ndarray)) else torch.long)
        }


def load_data() -> Tuple[List[str], List[List[str]], List[List[str]]]:
    """Загрузка размеченных данных"""
    
    logger.info("Загрузка данных...")
    
    reviews_path = Path('../backend/GoBackend/data/reviews.json')
    with open(reviews_path, 'r', encoding='utf-8') as f:
        labeled_data = json.load(f)
    
    site_reviews_path = Path('../backend/GoBackend/data/siteReviews.json')
    with open(site_reviews_path, 'r', encoding='utf-8') as f:
        site_reviews_data = json.load(f)
    
    id_to_text = {review['id']: review['text'] for review in site_reviews_data['reviews']}
    
    texts = []
    topics_list = []
    sentiments_list = []
    
    for review in labeled_data:
        if 'topics' in review and 'sentiments' in review and review['id'] in id_to_text:
            texts.append(id_to_text[review['id']])
            topics_list.append(review['topics'])
            sentiments_list.append(review['sentiments'])
    
    logger.info(f"Загружено {len(texts)} размеченных отзывов")
    
    return texts, topics_list, sentiments_list


def compute_metrics_multilabel(pred):
    """Метрики для multilabel классификации"""
    labels = pred.label_ids
    preds = pred.predictions
    preds = (preds > 0.5).astype(int)
    
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    hamming = hamming_loss(labels, preds)
    
    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'hamming_loss': hamming
    }


def compute_metrics_classification(pred):
    """Метрики для обычной классификации с детальным отчетом"""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    
    accuracy = (preds == labels).mean()
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
    
    # Per-class F1
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_negative': f1_per_class[0] if len(f1_per_class) > 0 else 0.0,
        'f1_neutral': f1_per_class[1] if len(f1_per_class) > 1 else 0.0,
        'f1_positive': f1_per_class[2] if len(f1_per_class) > 2 else 0.0,
    }


def train_topic_classifier(
    texts: List[str],
    topics_list: List[List[str]],
    output_dir: str = './trained_models/topic_classifier',
    model_name: str = 'DeepPavlov/rubert-base-cased',
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5
):
    """Обучение классификатора тем (без изменений)"""
    
    logger.info("=" * 60)
    logger.info("ОБУЧЕНИЕ КЛАССИФИКАТОРА ТЕМ")
    logger.info("=" * 60)
    
    mlb = MultiLabelBinarizer()
    topic_labels = mlb.fit_transform(topics_list)
    
    logger.info(f"Найдено уникальных тем: {len(mlb.classes_)}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        texts, topic_labels, test_size=0.15, random_state=42
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(mlb.classes_),
        problem_type="multi_label_classification"
    )
    
    train_dataset = ReviewDataset(X_train, y_train, tokenizer)
    val_dataset = ReviewDataset(X_val, y_val, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=200,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available()
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_multilabel,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    with open(Path(output_dir) / "mlb.pkl", 'wb') as f:
        pickle.dump(mlb, f)
    
    eval_results = trainer.evaluate()
    logger.info(f"F1-micro: {eval_results['eval_f1_micro']:.4f}")
    
    return model, tokenizer, mlb


def create_balanced_sentiment_dataset(
    texts: List[str], 
    topics_list: List[List[str]], 
    sentiments_list: List[List[str]]
):
    """Создание УМЕРЕННО сбалансированного датасета"""
    
    expanded_texts = []
    expanded_labels = []
    
    sentiment_to_id = {
        'отрицательно': 0,
        'нейтрально': 1,
        'положительно': 2
    }
    
    # Собираем базовый датасет
    for text, topics, sentiments in zip(texts, topics_list, sentiments_list):
        for topic, sentiment in zip(topics, sentiments):
            augmented_text = f"[{topic}] {text}"
            expanded_texts.append(augmented_text)
            expanded_labels.append(sentiment_to_id[sentiment])
    
    # Статистика
    label_counts = Counter(expanded_labels)
    logger.info(f"Исходное: neg={label_counts[0]}, neu={label_counts[1]}, pos={label_counts[2]}")
    
    # УМЕРЕННАЯ аугментация - до 40% от максимума (было 70%)
    max_count = max(label_counts.values())
    target_ratio = 0.4  # Меньше дубликатов!
    
    augmented_texts = []
    augmented_labels = []
    
    for label_id in range(3):
        indices = [i for i, l in enumerate(expanded_labels) if l == label_id]
        current_count = len(indices)
        
        # Добавляем оригиналы
        augmented_texts.extend([expanded_texts[i] for i in indices])
        augmented_labels.extend([label_id] * current_count)
        
        # Дублируем МЕНЬШЕ
        if current_count < max_count * target_ratio:
            times_to_duplicate = int((max_count * target_ratio) / current_count)
            for _ in range(times_to_duplicate - 1):
                augmented_texts.extend([expanded_texts[i] for i in indices])
                augmented_labels.extend([label_id] * current_count)
    
    final_counts = Counter(augmented_labels)
    logger.info(f"После балансировки: neg={final_counts[0]}, neu={final_counts[1]}, pos={final_counts[2]}")
    
    return augmented_texts, augmented_labels


def train_sentiment_classifier_improved(
    texts: List[str],
    topics_list: List[List[str]],
    sentiments_list: List[List[str]],
    output_dir: str = './trained_models/sentiment_classifier',
    model_name: str = 'DeepPavlov/rubert-base-cased',
    epochs: int = 4,  # Уменьшил с 6 - меньше переобучения
    batch_size: int = 16,
    learning_rate: float = 1.5e-5  # Чуть меньше learning rate
):
    """УЛУЧШЕННОЕ обучение sentiment с focal loss и балансировкой"""
    
    logger.info("=" * 60)
    logger.info("УЛУЧШЕННОЕ ОБУЧЕНИЕ SENTIMENT (Focal Loss + Balancing)")
    logger.info("=" * 60)
    
    # Создаём сбалансированный датасет
    expanded_texts, expanded_labels = create_balanced_sentiment_dataset(
        texts, topics_list, sentiments_list
    )
    
    # Вычисляем class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(expanded_labels),
        y=expanded_labels
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    
    # Train/Val split со стратификацией
    X_train, X_val, y_train, y_val = train_test_split(
        expanded_texts, expanded_labels, 
        test_size=0.15, 
        random_state=42, 
        stratify=expanded_labels
    )
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Модель с dropout для регуляризации
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        hidden_dropout_prob=0.2,
        attention_probs_dropout_prob=0.2
    )
    
    train_dataset = ReviewDataset(X_train, y_train, tokenizer)
    val_dataset = ReviewDataset(X_val, y_val, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=500,  # Еще больше warmup для стабильности
        weight_decay=0.05,  # Увеличил с 0.01 - сильнее регуляризация!
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",  # Macro F1 важнее!
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available()
    )
    
    # Используем WeightedTrainer с Focal Loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_classification,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # Раньше останов
        class_weights=class_weights_tensor.to(device)
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Детальная оценка
    eval_results = trainer.evaluate()
    logger.info("Финальные метрики:")
    logger.info(f"  Accuracy: {eval_results['eval_accuracy']:.4f}")
    logger.info(f"  F1-macro: {eval_results['eval_f1_macro']:.4f}")
    logger.info(f"  F1-weighted: {eval_results['eval_f1_weighted']:.4f}")
    logger.info(f"  F1-negative: {eval_results['eval_f1_negative']:.4f}")
    logger.info(f"  F1-neutral: {eval_results['eval_f1_neutral']:.4f}")
    logger.info(f"  F1-positive: {eval_results['eval_f1_positive']:.4f}")
    
    return model, tokenizer


def main():
    """Главная функция"""
    
    logger.info("УЛУЧШЕННОЕ ОБУЧЕНИЕ - Повышение точности sentiment")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Устройство: {device}")
    
    texts, topics_list, sentiments_list = load_data()
    
    # 1. Topic classifier - пропускаем если уже есть
    topic_dir = Path('./trained_models/topic_classifier')
    if (topic_dir / 'config.json').exists():
        logger.info("=" * 60)
        logger.info("Topic classifier уже обучен, ПРОПУСКАЕМ")
        logger.info("=" * 60)
    else:
        topic_model, topic_tokenizer, mlb = train_topic_classifier(
            texts, topics_list,
            output_dir='./trained_models/topic_classifier',
            epochs=5
        )
    
    # 2. УЛУЧШЕННЫЙ Sentiment classifier (правильная балансировка)
    sentiment_model, sentiment_tokenizer = train_sentiment_classifier_improved(
        texts, topics_list, sentiments_list,
        output_dir='./trained_models/sentiment_classifier',
        epochs=4  # Оптимально для избежания переобучения
    )
    
    logger.info("=" * 60)
    logger.info("ГОТОВО! Модели переобучены с улучшениями")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
