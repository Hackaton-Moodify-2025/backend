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
	total, err := s.repo.GetTotalReviews(req.Topic, req.Sentiment)
	if err != nil {
		s.logger.WithError(err).Error("Failed to get total reviews count")
		return nil, fmt.Errorf("failed to get total reviews count: %w", err)
	}

	// Get reviews
	reviews, err := s.repo.GetReviews(offset, req.Limit, req.Topic, req.Sentiment)
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
	s.logger.Info("Getting analytics data")

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
	}).Info("Successfully retrieved analytics data")

	return result, nil
}