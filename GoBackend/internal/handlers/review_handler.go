package handlers


import (
	"context"
	"strconv"
	"time"

	"github.com/gofiber/fiber/v3"
	"reviews-backend/internal/models"
	"reviews-backend/internal/service"
	"reviews-backend/pkg/logger"
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

	req := models.ReviewsRequest{
		Page:      page,
		Limit:     limit,
		Topic:     topic,
		Sentiment: sentiment,
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
	result, err := h.service.GetAnalyticsData(context.Background())
	if err != nil {
		h.logger.WithError(err).Errorf("Failed to get analytics data")
		return c.Status(fiber.StatusInternalServerError).JSON(models.ErrorResponse{
			Error:   "internal_error",
			Message: "Failed to get analytics data",
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