from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import logging
import time

# Используем оптимизированную модель для лучшей производительности
try:
    from model_optimized import OptimizedReviewClassifier as ReviewClassifier
    logger_init = logging.getLogger(__name__)
    logger_init.info("Используется оптимизированная модель (model_optimized.py)")
except ImportError:
    from model_production import ProductionReviewClassifier as ReviewClassifier
    logger_init = logging.getLogger(__name__)
    logger_init.info("Используется production модель (model_production.py)")


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Review Analysis ML Service",
    description="Сервис для классификации тем и определения тональности отзывов",
    version="1.0.0"
)

# CORS настройки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация модели при старте
classifier = None


class ReviewInput(BaseModel):
    id: int
    text: str


class PredictRequest(BaseModel):
    data: List[ReviewInput]


class ReviewPrediction(BaseModel):
    id: int
    topics: List[str]
    sentiments: List[str]


class PredictResponse(BaseModel):
    predictions: List[ReviewPrediction]


class ErrorResponse(BaseModel):
    error: str
    detail: str


@app.on_event("startup")
async def startup_event():
    """Загрузка модели при старте приложения"""
    global classifier
    try:
        logger.info("Загрузка ML модели...")
        classifier = ReviewClassifier()
        logger.info("Модель успешно загружена")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {str(e)}")
        raise


@app.get("/")
async def root():
    """Базовый эндпоинт для проверки работоспособности"""
    return {
        "service": "Review Analysis ML Service",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "healthy",
        "model_loaded": classifier is not None
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Основной эндпоинт для предсказания тем и тональностей
    
    Принимает список отзывов и возвращает предсказания для каждого.
    Максимум 250 отзывов за запрос, время обработки <= 3 минуты.
    
    ОПТИМИЗИРОВАНО: использует батч-обработку для ускорения
    """
    try:
        # Валидация
        if not request.data:
            raise HTTPException(
                status_code=400,
                detail="Пустой массив данных"
            )
        
        if len(request.data) > 250:
            raise HTTPException(
                status_code=400,
                detail="Превышен лимит: максимум 250 отзывов за запрос"
            )
        
        # Проверка, что все тексты не пустые
        for review in request.data:
            if not review.text or not review.text.strip():
                raise HTTPException(
                    status_code=400,
                    detail=f"Пустой текст для отзыва с id={review.id}"
                )
        
        logger.info(f"Получен запрос на обработку {len(request.data)} отзывов")
        start_time = time.time()
        
        # ОПТИМИЗАЦИЯ: Батч обработка всех отзывов сразу
        texts = [review.text for review in request.data]
        ids = [review.id for review in request.data]
        
        # Используем batch predict если доступен
        if hasattr(classifier, 'predict_batch'):
            results = classifier.predict_batch(texts)
        else:
            # Fallback на старый метод
            results = [classifier.predict_single(text) for text in texts]
        
        # Формирование ответа
        predictions = []
        for review_id, result in zip(ids, results):
            predictions.append(ReviewPrediction(
                id=review_id,
                topics=result["topics"],
                sentiments=result["sentiments"]
            ))
        
        elapsed = time.time() - start_time
        logger.info(f"Успешно обработано {len(predictions)} отзывов за {elapsed:.2f}с ({elapsed/len(predictions):.3f}с/отзыв)")
        
        return PredictResponse(predictions=predictions)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Внутренняя ошибка сервера: {str(e)}"
        )


@app.get("/topics")
async def get_topics():
    """Получить список всех доступных тем"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    return {
        "topics": classifier.get_available_topics(),
        "total": len(classifier.get_available_topics())
    }


@app.get("/stats")
async def get_stats():
    """Получить статистику работы модели"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    if hasattr(classifier, 'get_stats'):
        stats = classifier.get_stats()
        return {
            "status": "ok",
            "statistics": stats
        }
    else:
        return {
            "status": "ok",
            "statistics": {
                "message": "Статистика недоступна для текущей версии модели"
            }
        }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
