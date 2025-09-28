package handlers

import (
	"context"
	"strconv"
	"time"

	"reviews-backend/internal/models"
	"reviews-backend/internal/service"
	"reviews-backend/pkg/logger"

	"github.com/gofiber/fiber/v3"
)

// ReviewHandler handles review-related HTTP requests
type ReviewHandler struct {
	service service.ReviewService
	logger  *logger.Logger
}

// NewReviewHandler creates a new review handler
func NewReviewHandler(service service.ReviewService, logger *logger.Logger) *ReviewHandler {
	return &ReviewHandler{
		service: service,
		logger:  logger,
	}
}

// GetReviews handles GET /api/reviews
func (h *ReviewHandler) GetReviews(c fiber.Ctx) error {
	// Parse query parameters
	page, _ := strconv.Atoi(c.Query("page", "1"))
	limit, _ := strconv.Atoi(c.Query("limit", "20"))
	topic := c.Query("topic", "")
	sentiment := c.Query("sentiment", "")
	dateFrom := c.Query("date_from", "")
	dateTo := c.Query("date_to", "")

	req := models.ReviewsRequest{
		Page:      page,
		Limit:     limit,
		Topic:     topic,
		Sentiment: sentiment,
		DateFrom:  dateFrom,
		DateTo:    dateTo,
	}

	// Get reviews
	result, err := h.service.GetPaginatedReviews(context.Background(), req)
	if err != nil {
		h.logger.WithError(err).Errorf("Failed to get reviews")
		return c.Status(fiber.StatusInternalServerError).JSON(models.ErrorResponse{
			Error:   "internal_error",
			Message: "Failed to get reviews",
		})
	}

	return c.JSON(result)
}

// GetAnalytics handles GET /api/analytics
func (h *ReviewHandler) GetAnalytics(c fiber.Ctx) error {
	// Parse query parameters for filtering
	topic := c.Query("topic", "")
	sentiment := c.Query("sentiment", "")
	dateFrom := c.Query("date_from", "")
	dateTo := c.Query("date_to", "")

	// If no date filters are provided, default to last 180 days
	if dateFrom == "" && dateTo == "" {
		now := time.Now()
		dateFrom = now.AddDate(0, 0, -180).Format("2006-01-02")
		dateTo = now.Format("2006-01-02")

		h.logger.WithFields(map[string]interface{}{
			"default_date_from": dateFrom,
			"default_date_to":   dateTo,
		}).Info("No date filters provided, using default 180 days period")
	}

	h.logger.WithFields(map[string]interface{}{
		"topic":     topic,
		"sentiment": sentiment,
		"date_from": dateFrom,
		"date_to":   dateTo,
	}).Info("Processing analytics request with filters")

	// Always use filtered analytics (including default date range)
	req := models.AnalyticsRequest{
		Topic:     topic,
		Sentiment: sentiment,
		DateFrom:  dateFrom,
		DateTo:    dateTo,
	}

	result, err := h.service.GetFilteredAnalyticsData(context.Background(), req)
	if err != nil {
		h.logger.WithError(err).Errorf("Failed to get filtered analytics data")
		return c.Status(fiber.StatusInternalServerError).JSON(models.ErrorResponse{
			Error:   "internal_error",
			Message: "Failed to get filtered analytics data",
		})
	}

	return c.JSON(result)
}

// GetFilteredAnalytics handles GET /api/analytics with filters
func (h *ReviewHandler) GetFilteredAnalytics(c fiber.Ctx) error {
	topic := c.Query("topic", "")
	sentiment := c.Query("sentiment", "")
	dateFrom := c.Query("date_from", "")
	dateTo := c.Query("date_to", "")

	req := models.AnalyticsRequest{
		Topic:     topic,
		Sentiment: sentiment,
		DateFrom:  dateFrom,
		DateTo:    dateTo,
	}

	result, err := h.service.GetFilteredAnalyticsData(context.Background(), req)
	if err != nil {
		h.logger.WithError(err).Errorf("Failed to get filtered analytics data")
		return c.Status(fiber.StatusInternalServerError).JSON(models.ErrorResponse{
			Error:   "internal_error",
			Message: "Failed to get filtered analytics data",
		})
	}

	return c.JSON(result)
}

// HealthCheck handles GET /health
func (h *ReviewHandler) HealthCheck(c fiber.Ctx) error {
	return c.JSON(models.HealthResponse{
		Status:    "healthy",
		Timestamp: time.Now(),
		Version:   "1.0.0",
	})
}
