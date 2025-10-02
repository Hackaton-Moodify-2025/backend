package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"reviews-backend/internal/models"
	"reviews-backend/pkg/logger"

	"github.com/gofiber/fiber/v3"
)

// MLHandler handles ML prediction requests
type MLHandler struct {
	logger       *logger.Logger
	mlServiceURL string
	httpClient   *http.Client
}

// NewMLHandler creates a new ML handler
func NewMLHandler(logger *logger.Logger) *MLHandler {
	mlServiceURL := os.Getenv("ML_SERVICE_URL")
	if mlServiceURL == "" {
		mlServiceURL = "http://ml-service:8000"
	}

	return &MLHandler{
		logger:       logger,
		mlServiceURL: mlServiceURL,
		httpClient: &http.Client{
			Timeout: 5 * time.Minute, // 5 minutes timeout for ML predictions
		},
	}
}

// PredictReviews handles POST /api/v1/ml/predict
func (h *MLHandler) PredictReviews(c fiber.Ctx) error {
	// Parse request body
	var req models.PredictRequest
	if err := c.Bind().JSON(&req); err != nil {
		h.logger.WithError(err).Error("Failed to parse predict request")
		return c.Status(fiber.StatusBadRequest).JSON(models.ErrorResponse{
			Error:   "invalid_request",
			Message: "Invalid request format",
		})
	}

	// Validate request
	if len(req.Data) == 0 {
		return c.Status(fiber.StatusBadRequest).JSON(models.ErrorResponse{
			Error:   "empty_data",
			Message: "Request data cannot be empty",
		})
	}

	if len(req.Data) > 400 {
		return c.Status(fiber.StatusBadRequest).JSON(models.ErrorResponse{
			Error:   "too_many_records",
			Message: "Maximum 400 reviews per request",
		})
	}

	startTime := time.Now()

	h.logger.WithFields(map[string]interface{}{
		"review_count": len(req.Data),
	}).Info("Processing ML prediction request")

	// Forward request to ML service
	jsonData, err := json.Marshal(req)
	if err != nil {
		h.logger.WithError(err).Error("Failed to marshal request")
		return c.Status(fiber.StatusInternalServerError).JSON(models.ErrorResponse{
			Error:   "internal_error",
			Message: "Failed to process request",
		})
	}

	mlURL := fmt.Sprintf("%s/predict", h.mlServiceURL)
	httpReq, err := http.NewRequestWithContext(context.Background(), "POST", mlURL, bytes.NewBuffer(jsonData))
	if err != nil {
		h.logger.WithError(err).Error("Failed to create ML service request")
		return c.Status(fiber.StatusInternalServerError).JSON(models.ErrorResponse{
			Error:   "internal_error",
			Message: "Failed to create ML service request",
		})
	}

	httpReq.Header.Set("Content-Type", "application/json")

	// Send request to ML service
	resp, err := h.httpClient.Do(httpReq)
	if err != nil {
		h.logger.WithError(err).Error("Failed to send request to ML service")
		return c.Status(fiber.StatusServiceUnavailable).JSON(models.ErrorResponse{
			Error:   "ml_service_unavailable",
			Message: "ML service is temporarily unavailable",
		})
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		h.logger.WithError(err).Error("Failed to read ML service response")
		return c.Status(fiber.StatusInternalServerError).JSON(models.ErrorResponse{
			Error:   "internal_error",
			Message: "Failed to read ML service response",
		})
	}

	// Check ML service response status
	if resp.StatusCode != http.StatusOK {
		h.logger.WithFields(map[string]interface{}{
			"status_code": resp.StatusCode,
			"response":    string(body),
		}).Error("ML service returned error")

		return c.Status(resp.StatusCode).JSON(models.ErrorResponse{
			Error:   "ml_service_error",
			Message: fmt.Sprintf("ML service error: %s", string(body)),
		})
	}

	// Parse ML service response
	var mlResponse models.PredictResponse
	if err := json.Unmarshal(body, &mlResponse); err != nil {
		h.logger.WithError(err).Error("Failed to parse ML service response")
		return c.Status(fiber.StatusInternalServerError).JSON(models.ErrorResponse{
			Error:   "internal_error",
			Message: "Failed to parse ML service response",
		})
	}

	duration := time.Since(startTime)

	h.logger.WithFields(map[string]interface{}{
		"predictions_count": len(mlResponse.Predictions),
		"duration_ms":       duration.Milliseconds(),
		"duration_sec":      duration.Seconds(),
		"avg_per_review_ms": duration.Milliseconds() / int64(len(req.Data)),
	}).Info("Successfully processed ML predictions")

	// Return predictions
	return c.JSON(mlResponse)
}

// GetMLServiceHealth handles GET /api/v1/ml/health
func (h *MLHandler) GetMLServiceHealth(c fiber.Ctx) error {
	mlURL := fmt.Sprintf("%s/health", h.mlServiceURL)

	resp, err := h.httpClient.Get(mlURL)
	if err != nil {
		h.logger.WithError(err).Error("Failed to check ML service health")
		return c.Status(fiber.StatusServiceUnavailable).JSON(models.ErrorResponse{
			Error:   "ml_service_unavailable",
			Message: "Cannot connect to ML service",
		})
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		h.logger.WithError(err).Error("Failed to read ML service health response")
		return c.Status(fiber.StatusInternalServerError).JSON(models.ErrorResponse{
			Error:   "internal_error",
			Message: "Failed to read ML service health response",
		})
	}

	var healthResponse map[string]interface{}
	if err := json.Unmarshal(body, &healthResponse); err != nil {
		h.logger.WithError(err).Error("Failed to parse ML service health response")
		return c.Status(fiber.StatusInternalServerError).JSON(models.ErrorResponse{
			Error:   "internal_error",
			Message: "Failed to parse ML service health response",
		})
	}

	return c.JSON(healthResponse)
}
