package service

import (
	"context"
	"fmt"
	"math"

	"reviews-backend/internal/models"
	"reviews-backend/internal/repository"
	"reviews-backend/pkg/logger"
)

// ReviewService handles business logic for reviews
type ReviewService interface {
	GetPaginatedReviews(ctx context.Context, req models.ReviewsRequest) (*models.PaginatedReviews, error)
	GetAnalyticsData(ctx context.Context) (*models.AnalyticsData, error)
	GetFilteredAnalyticsData(ctx context.Context, req models.AnalyticsRequest) (*models.AnalyticsData, error)
	GetReviewByID(ctx context.Context, id int) (*models.Review, error)
}

// ReviewServiceImpl implements ReviewService
type ReviewServiceImpl struct {
	repo   repository.ReviewRepository
	logger *logger.Logger
}

// NewReviewService creates a new review service
func NewReviewService(repo repository.ReviewRepository, log *logger.Logger) ReviewService {
	return &ReviewServiceImpl{
		repo:   repo,
		logger: log,
	}
}

// GetPaginatedReviews returns paginated reviews with filtering
func (s *ReviewServiceImpl) GetPaginatedReviews(ctx context.Context, req models.ReviewsRequest) (*models.PaginatedReviews, error) {
	// Validate and set defaults
	if req.Page < 1 {
		req.Page = 1
	}
	if req.Limit < 1 || req.Limit > 100 {
		req.Limit = 20
	}

	offset := (req.Page - 1) * req.Limit

	s.logger.WithFields(map[string]interface{}{
		"page":      req.Page,
		"limit":     req.Limit,
		"topic":     req.Topic,
		"sentiment": req.Sentiment,
		"offset":    offset,
	}).Info("Getting paginated reviews")

	// Get total count
	total, err := s.repo.GetTotalReviews(req.Topic, req.Sentiment, req.DateFrom, req.DateTo)
	if err != nil {
		s.logger.WithError(err).Error("Failed to get total reviews count")
		return nil, fmt.Errorf("failed to get total reviews count: %w", err)
	}

	// Get reviews
	reviews, err := s.repo.GetReviews(offset, req.Limit, req.Topic, req.Sentiment, req.DateFrom, req.DateTo)
	if err != nil {
		s.logger.WithError(err).Error("Failed to get reviews")
		return nil, fmt.Errorf("failed to get reviews: %w", err)
	}

	totalPages := int(math.Ceil(float64(total) / float64(req.Limit)))

	result := &models.PaginatedReviews{
		Reviews:    reviews,
		Total:      total,
		Page:       req.Page,
		Limit:      req.Limit,
		TotalPages: totalPages,
	}

	s.logger.WithFields(map[string]interface{}{
		"total_reviews": total,
		"returned":      len(reviews),
		"total_pages":   totalPages,
	}).Info("Successfully retrieved paginated reviews")

	return result, nil
}

// GetAnalyticsData returns all data needed for analytics
func (s *ReviewServiceImpl) GetAnalyticsData(ctx context.Context) (*models.AnalyticsData, error) {
	s.logger.Info("Getting analytics data (excluding otzovik.com - only sravni.ru and banki.ru)")

	reviews, predictions, err := s.repo.GetAllReviewsForAnalytics()
	if err != nil {
		s.logger.WithError(err).Error("Failed to get analytics data")
		return nil, fmt.Errorf("failed to get analytics data: %w", err)
	}

	result := &models.AnalyticsData{
		Reviews:     reviews,
		Predictions: predictions,
	}

	s.logger.WithFields(map[string]interface{}{
		"reviews_count":     len(reviews),
		"predictions_count": len(predictions),
		"source_filter":     "sravni.ru,banki.ru only",
	}).Info("Successfully retrieved analytics data")

	return result, nil
}

// GetFilteredAnalyticsData returns filtered data for analytics
func (s *ReviewServiceImpl) GetFilteredAnalyticsData(ctx context.Context, req models.AnalyticsRequest) (*models.AnalyticsData, error) {
	s.logger.WithFields(map[string]interface{}{
		"topic":     req.Topic,
		"sentiment": req.Sentiment,
		"date_from": req.DateFrom,
		"date_to":   req.DateTo,
	}).Info("Getting filtered analytics data (excluding otzovik.com - only sravni.ru and banki.ru)")

	reviews, predictions, err := s.repo.GetFilteredReviewsForAnalytics(req.Topic, req.Sentiment, req.DateFrom, req.DateTo)
	if err != nil {
		s.logger.WithError(err).Error("Failed to get filtered analytics data")
		return nil, fmt.Errorf("failed to get filtered analytics data: %w", err)
	}

	result := &models.AnalyticsData{
		Reviews:     reviews,
		Predictions: predictions,
	}

	s.logger.WithFields(map[string]interface{}{
		"reviews_count":     len(reviews),
		"predictions_count": len(predictions),
		"source_filter":     "sravni.ru,banki.ru only",
	}).Info("Successfully retrieved filtered analytics data")

	return result, nil
}

// GetReviewByID returns a single review by ID
func (s *ReviewServiceImpl) GetReviewByID(ctx context.Context, id int) (*models.Review, error) {
	s.logger.WithFields(map[string]interface{}{
		"review_id": id,
	}).Info("Getting review by ID")

	review, err := s.repo.GetReviewByID(id)
	if err != nil {
		s.logger.WithError(err).WithFields(map[string]interface{}{
			"review_id": id,
		}).Error("Failed to get review by ID")
		return nil, fmt.Errorf("failed to get review by ID: %w", err)
	}

	if review == nil {
		s.logger.WithFields(map[string]interface{}{
			"review_id": id,
		}).Warn("Review not found")
		return nil, fmt.Errorf("review with ID %d not found", id)
	}

	s.logger.WithFields(map[string]interface{}{
		"review_id": id,
	}).Info("Successfully retrieved review by ID")

	return review, nil
}
