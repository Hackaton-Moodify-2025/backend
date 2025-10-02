"""
OPTIMIZED MODEL INFERENCE
Быстрый инференс с оптимизированными моделями и батчингом
"""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import time

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Иерархия тем для подавления родителей
PARENTS = {
    "Карты": {"Комиссии и тарифы", "Переводы и платежи", "Безопасность", "Кэшбек и бонусы"}
}

# Лексиконы для извлечения релевантных спанов
TOPIC_LEX = {
    "Отделения": ["очеред", "ждал", "талон", "оператор", "окно"],
    "Карты": ["комисс", "снят", "не проход", "карт", "пластик"],
    "Безопасность": ["блокир", "фрод", "подтвержд", "безопасн"],
    "Мобильное приложение": ["завис", "лаг", "обновлен", "краш", "приложен", "апп"],
    "Депозиты и вклады": ["вклад", "депозит", "процент", "досрочн"],
    "Переводы и платежи": ["перевод", "платеж", "коммунал", "штраф"],
    "Служба поддержки": ["поддерж", "горячая линия", "колл", "оператор"],
    "Кредиты и ипотека": ["кредит", "ипотек", "заём", "долг"],
    "Кэшбек и бонусы": ["кэшбек", "кешбэк", "бонус", "милл"],
    "Банкоматы": ["банкомат", "наличн", "снять"]
}

# Негативные триггеры для логит-байаса
NEG_HINTS = {
    "Карты": ["комисс", "снят", "налич", "не работа"],
    "Отделения": ["очеред", "ждал", "долго"],
    "Мобильное приложение": ["завис", "ошибк", "лаг", "глюч"],
    "Безопасность": ["блокир", "подтвержд", "заблок"],
    "Служба поддержки": ["не отвечает", "недозвон", "игнор"]
}


class OptimizedReviewClassifier:
    """Оптимизированный классификатор отзывов с батчингом и кэшированием"""
    
    def __init__(self, models_dir: str = "./trained_models"):
        """
        Args:
            models_dir: Путь к директории с обученными моделями
        """
        self.models_dir = Path(models_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info("Инициализация OptimizedReviewClassifier...")
        logger.info(f"Устройство: {self.device}")
        
        self._load_models()
        
        # Статистика для мониторинга
        self.stats = {
            'total_predictions': 0,
            'total_time': 0.0,
        }
    
    def _load_models(self):
        """Загрузка обученных моделей с оптимизациями"""
        
        topic_model_path = self.models_dir / "topic_classifier"
        sentiment_model_path = self.models_dir / "sentiment_classifier"
        
        if not topic_model_path.exists() or not sentiment_model_path.exists():
            raise FileNotFoundError(
                f"Модели не найдены в {self.models_dir}. "
                "Запустите train_optimized.py для обучения."
            )
        
        # Topic Classifier
        logger.info(f"Загрузка topic модели из {topic_model_path}")
        self.topic_tokenizer = AutoTokenizer.from_pretrained(str(topic_model_path))
        self.topic_model = AutoModelForSequenceClassification.from_pretrained(str(topic_model_path))
        self.topic_model.to(self.device)
        self.topic_model.eval()
        
        # CPU оптимизации
        if self.device.type == "cpu":
            torch.set_num_threads(max(1, torch.get_num_threads()))
            if hasattr(torch.backends, 'mkldnn'):
                torch.backends.mkldnn.enabled = True
            # Динамическая квантизация для CPU (ускоряет в 1.5-2x)
            try:
                from torch.quantization import quantize_dynamic
                self.topic_model = quantize_dynamic(
                    self.topic_model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("✅ Topic model: применена динамическая квантизация")
            except Exception as e:
                logger.warning(f"Квантизация topic model недоступна: {e}")
        
        # Загрузка MultiLabelBinarizer
        mlb_path = topic_model_path / "mlb.pkl"
        with open(mlb_path, 'rb') as f:
            self.mlb = pickle.load(f)
        
        self.topics = list(self.mlb.classes_)
        logger.info(f"Загружено {len(self.topics)} тем")
        
        # Загрузка калиброванных порогов per-class
        thr_path = topic_model_path / "thresholds.json"
        self.class_thresholds = None
        if thr_path.exists():
            import json
            obj = json.load(open(thr_path, "r", encoding="utf-8"))
            # Убедимся, что порядок classes совпадает с self.topics
            if obj.get("classes") == self.topics:
                self.class_thresholds = obj["thresholds"]
                logger.info(f"✅ Загружены калиброванные пороги для {len(self.class_thresholds)} классов")
            else:
                logger.warning("⚠️  Порядок классов в thresholds.json не совпадает с моделью")
        else:
            logger.warning("⚠️  Калиброванные пороги не найдены, используется threshold=0.4")
        
        # Sentiment Classifier
        logger.info(f"Загрузка sentiment модели из {sentiment_model_path}")
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(str(sentiment_model_path))
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(str(sentiment_model_path))
        self.sentiment_model.to(self.device)
        self.sentiment_model.eval()
        
        # CPU оптимизации для sentiment
        if self.device.type == "cpu":
            try:
                from torch.quantization import quantize_dynamic
                self.sentiment_model = quantize_dynamic(
                    self.sentiment_model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info("✅ Sentiment model: применена динамическая квантизация")
            except Exception as e:
                logger.warning(f"Квантизация sentiment model недоступна: {e}")
        
        self.sentiment_labels = {
            0: "отрицательно",
            1: "нейтрально",
            2: "положительно"
        }
        
        logger.info("✅ Модели успешно загружены и оптимизированы")
    
    def get_available_topics(self) -> List[str]:
        """Возвращает список всех доступных тем"""
        return self.topics
    
    def _suppress_parents(self, topics: List[str]) -> List[str]:
        """Подавление родительских тем при наличии дочерних"""
        ts = set(topics)
        for parent, childs in PARENTS.items():
            if parent in ts and any(ch in ts for ch in childs):
                ts.discard(parent)
        return list(ts)
    
    def _extract_relevant_span(self, text: str, topic: str) -> str:
        """Извлечение релевантного предложения для темы"""
        # Разбиваем на предложения
        sents = [s.strip() for s in text.replace("—", ".").replace("–", ".").split(".") if s.strip()]
        keys = TOPIC_LEX.get(topic, [])
        
        # Ищем предложение с ключевыми словами
        for s in sents:
            low = s.lower()
            if any(k in low for k in keys):
                return s
        
        # Fallback: первые 220 символов
        return text[:220]
    
    def predict_topics_batch(self, texts: List[str], threshold: float = 0.4) -> List[List[str]]:
        """
        BATCH предсказание тем для списка текстов (быстрее!)
        
        Args:
            texts: Список текстов отзывов
            threshold: Порог вероятности для multilabel (0-1)
        
        Returns:
            Список списков предсказанных тем для каждого текста
        """
        if not texts:
            return []
        
        # Батч токенизация с динамическим padding (быстрее)
        inputs = self.topic_tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="longest"  # Паддинг до самого длинного в батче
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Батч предсказание с inference_mode (быстрее)
        with torch.inference_mode():
            outputs = self.topic_model(**inputs)
            probabilities = torch.sigmoid(outputs.logits).cpu().numpy()
        
        # Обработка результатов с per-class thresholds
        results = []
        for i in range(len(texts)):
            if self.class_thresholds:
                # Используем калиброванные пороги для каждого класса
                import numpy as np
                mask = probabilities[i] > np.array(self.class_thresholds)
                predicted_topics = [self.topics[j] for j, ok in enumerate(mask) if ok]
            else:
                # Fallback на единый threshold
                mask = probabilities[i] > threshold
                predicted_topics = [self.topics[j] for j, val in enumerate(mask) if val]
            
            # Если ничего не найдено, берём топ-1 по вероятности
            if not predicted_topics:
                top_idx = int(probabilities[i].argmax())
                predicted_topics = [self.topics[top_idx]]
            
            # Подавляем родительские темы
            predicted_topics = self._suppress_parents(predicted_topics)
            
            results.append(predicted_topics)
        
        return results
    
    def predict_topics(self, text: str, threshold: float = 0.4) -> List[str]:
        """Предсказание тем для одного текста"""
        return self.predict_topics_batch([text], threshold)[0]
    
    def predict_sentiment_batch(self, texts: List[str], topics: List[str]) -> List[str]:
        """
        BATCH предсказание тональности для пар (текст, тема)
        
        Args:
            texts: Список текстов (может повторяться)
            topics: Список тем (соответствует текстам по индексу)
        
        Returns:
            Список тональностей для каждой пары
        """
        if not texts or not topics or len(texts) != len(topics):
            return []
        
        # Извлекаем релевантные спаны для каждой темы
        spans = [self._extract_relevant_span(text, topic) for text, topic in zip(texts, topics)]
        
        # Формируем батч с темами
        augmented_texts = [f"[{topic}] {span}" for topic, span in zip(topics, spans)]
        
        # Батч токенизация с динамическим padding
        inputs = self.sentiment_tokenizer(
            augmented_texts,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="longest"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Батч предсказание с улучшениями
        with torch.inference_mode():
            outputs = self.sentiment_model(**inputs)
            logits = outputs.logits
            
            # Применяем логит-байас по негативным триггерам
            biases = []
            for span, topic in zip(spans, topics):
                low = span.lower()
                if any(k in low for k in NEG_HINTS.get(topic, [])):
                    biases.append([+0.25, 0.0, -0.25])  # [neg, neu, pos]
                else:
                    biases.append([0.0, 0.0, 0.0])
            
            biases_tensor = torch.tensor(biases, dtype=torch.float).to(logits.device)
            logits = logits + biases_tensor
            
            # Температурная калибровка (снижаем уверенность)
            logits = logits / 1.4
            
            # Softmax для вероятностей
            probs = torch.softmax(logits, dim=-1)
            conf, sentiment_ids = probs.max(dim=-1)
            
            # Нейтральный дефолт при низкой уверенности
            sentiment_ids = torch.where(
                conf < 0.55,
                torch.tensor(1, device=sentiment_ids.device),  # 1 = нейтрально
                sentiment_ids
            )
            
            sentiment_ids = sentiment_ids.cpu().numpy()
        
        # Конвертация в строки
        return [self.sentiment_labels[int(sid)] for sid in sentiment_ids]
    
    def predict_sentiment(self, text: str, topic: str) -> str:
        """Предсказание тональности для одной пары (текст, тема)"""
        return self.predict_sentiment_batch([text], [topic])[0]
    
    def predict_batch(self, texts: List[str], threshold: float = 0.4) -> List[Dict]:
        """
        ОПТИМИЗИРОВАННОЕ батч предсказание для списка отзывов
        
        Использует батчинг для максимальной производительности
        
        Args:
            texts: Список текстов отзывов
            threshold: Порог для классификации тем
        
        Returns:
            Список словарей с topics и sentiments для каждого отзыва
        """
        if not texts:
            return []
        
        start_time = time.time()
        
        # 1. Батч предсказание тем для всех текстов сразу
        all_topics = self.predict_topics_batch(texts, threshold)
        
        # 2. Подготовка батча для sentiment анализа
        # Собираем все пары (текст, тема) в один большой батч
        sentiment_texts = []
        sentiment_topics = []
        review_indices = []  # Для отслеживания к какому отзыву относится результат
        
        for idx, (text, topics) in enumerate(zip(texts, all_topics)):
            for topic in topics:
                sentiment_texts.append(text)
                sentiment_topics.append(topic)
                review_indices.append(idx)
        
        # 3. Батч предсказание тональности для всех пар сразу
        all_sentiments = self.predict_sentiment_batch(sentiment_texts, sentiment_topics)
        
        # 4. Группировка результатов по отзывам
        results = [{"topics": [], "sentiments": []} for _ in range(len(texts))]
        
        for review_idx, topic, sentiment in zip(review_indices, sentiment_topics, all_sentiments):
            results[review_idx]["topics"].append(topic)
            results[review_idx]["sentiments"].append(sentiment)
        
        # Статистика
        elapsed = time.time() - start_time
        self.stats['total_predictions'] += len(texts)
        self.stats['total_time'] += elapsed
        
        avg_time = elapsed / len(texts) if texts else 0
        logger.info(f"Обработано {len(texts)} отзывов за {elapsed:.2f}с ({avg_time:.3f}с/отзыв)")
        
        return results
    
    def predict(self, texts: List[str]) -> List[Dict]:
        """Алиас для predict_batch (совместимость с API)"""
        return self.predict_batch(texts)
    
    def predict_single(self, text: str) -> Dict:
        """
        Предсказание для одного отзыва
        
        Args:
            text: Текст отзыва
        
        Returns:
            Словарь с topics и sentiments
        """
        return self.predict_batch([text])[0]
    
    def get_stats(self) -> Dict:
        """Получить статистику работы модели"""
        if self.stats['total_predictions'] > 0:
            avg_time = self.stats['total_time'] / self.stats['total_predictions']
        else:
            avg_time = 0
        
        return {
            'total_predictions': self.stats['total_predictions'],
            'total_time': self.stats['total_time'],
            'avg_time_per_review': avg_time,
        }


# Глобальный экземпляр для быстрого доступа
_classifier_instance = None


def get_classifier() -> OptimizedReviewClassifier:
    """Singleton для получения классификатора"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = OptimizedReviewClassifier()
    return _classifier_instance


if __name__ == "__main__":
    # Тестирование производительности
    logger.info("=" * 60)
    logger.info("ТЕСТИРОВАНИЕ ОПТИМИЗИРОВАННОЙ МОДЕЛИ")
    logger.info("=" * 60)
    
    classifier = OptimizedReviewClassifier()
    
    test_texts = [
        "Карты говно не рекомендую, отделения хорошие, персонал так себе",
        "Очень понравилось обслуживание в отделении, но мобильное приложение часто зависает",
        "Кэшбек отличный, всем советую!",
        "Ужасная поддержка, никто не отвечает",
        "Хороший банк, но комиссии высокие и страховки навязывают",
        "Премиум обслуживание на высоте, интернет-банк удобный",
        "Кредит взял - процент нормальный, но ипотека дорогая",
        "Безопасность на уровне, но дистанционное обслуживание тормозит",
    ]
    
    logger.info(f"\n🧪 Тест 1: Обработка {len(test_texts)} отзывов по одному")
    start = time.time()
    for text in test_texts:
        result = classifier.predict_single(text)
        logger.info(f"\nТекст: {text[:60]}...")
        logger.info(f"Темы: {result['topics']}")
        logger.info(f"Тональности: {result['sentiments']}")
    elapsed_single = time.time() - start
    logger.info(f"\n⏱️  Время (по одному): {elapsed_single:.2f}с ({elapsed_single/len(test_texts):.3f}с/отзыв)")
    
    logger.info(f"\n🚀 Тест 2: Батч обработка {len(test_texts)} отзывов")
    start = time.time()
    results = classifier.predict_batch(test_texts)
    elapsed_batch = time.time() - start
    logger.info(f"⏱️  Время (батч): {elapsed_batch:.2f}с ({elapsed_batch/len(test_texts):.3f}с/отзыв)")
    logger.info(f"⚡ Ускорение: {elapsed_single/elapsed_batch:.1f}x")
    
    for text, result in zip(test_texts, results):
        logger.info(f"\nТекст: {text[:60]}...")
        logger.info(f"Темы: {result['topics']}")
        logger.info(f"Тональности: {result['sentiments']}")
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 ИТОГОВАЯ СТАТИСТИКА:")
    stats = classifier.get_stats()
    logger.info(f"Всего обработано: {stats['total_predictions']} отзывов")
    logger.info(f"Среднее время: {stats['avg_time_per_review']:.3f}с/отзыв")
    logger.info("=" * 60)
