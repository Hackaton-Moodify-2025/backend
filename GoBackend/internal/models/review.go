package models

import "time"

// Review represents a single review from the site
type Review struct {
	ID         int      `json:"id"`
	Link       string   `json:"link"`
	Date       string   `json:"date"`
	Title      string   `json:"title"`
	Text       string   `json:"text"`
	Rating     string   `json:"rating"`
	Status     *string  `json:"status"`
	Product    *string  `json:"product"`
	City       string   `json:"city"`
	Topics     []string `json:"topics,omitempty"`
	Sentiments []string `json:"sentiments,omitempty"`
}

// ReviewPrediction represents sentiment analysis data
type ReviewPrediction struct {
	ID         int      `json:"id"`
	Topics     []string `json:"topics"`
	Sentiments []string `json:"sentiments"`
}

// PaginatedReviews represents paginated response for reviews
type PaginatedReviews struct {
	Reviews    []Review `json:"reviews"`
	Total      int      `json:"total"`
	Page       int      `json:"page"`
	Limit      int      `json:"limit"`
	TotalPages int      `json:"total_pages"`
}

// AnalyticsData represents all data needed for analytics
type AnalyticsData struct {
	Reviews     []Review           `json:"reviews"`
	Predictions []ReviewPrediction `json:"predictions"`
}

// TopicFilter represents filter parameters
type TopicFilter struct {
	Topic     string `json:"topic"`
	Sentiment string `json:"sentiment"`
}

// ReviewsRequest represents request parameters for getting reviews
type ReviewsRequest struct {
	Page      int    `form:"page" json:"page"`
	Limit     int    `form:"limit" json:"limit"`
	Topic     string `form:"topic" json:"topic"`
	Sentiment string `form:"sentiment" json:"sentiment"`
	DateFrom  string `form:"date_from" json:"date_from"` // Формат: YYYY-MM-DD
	DateTo    string `form:"date_to" json:"date_to"`     // Формат: YYYY-MM-DD
}

// AnalyticsRequest represents request parameters for getting analytics data
type AnalyticsRequest struct {
	Topic     string `form:"topic" json:"topic"`
	Sentiment string `form:"sentiment" json:"sentiment"`
	DateFrom  string `form:"date_from" json:"date_from"` // Формат: YYYY-MM-DD
	DateTo    string `form:"date_to" json:"date_to"`     // Формат: YYYY-MM-DD
}

// ErrorResponse represents API error response
type ErrorResponse struct {
	Error   string `json:"error"`
	Message string `json:"message"`
}

// HealthResponse represents health check response
type HealthResponse struct {
	Status    string    `json:"status"`
	Timestamp time.Time `json:"timestamp"`
	Version   string    `json:"version"`
}

// ML Prediction Models

// ReviewInput represents input data for ML prediction
type ReviewInput struct {
	ID   int    `json:"id"`
	Text string `json:"text"`
}

// PredictRequest represents request to ML service
type PredictRequest struct {
	Data []ReviewInput `json:"data"`
}

// ReviewPredictionOutput represents ML prediction output for a single review
type ReviewPredictionOutput struct {
	ID         int      `json:"id"`
	Topics     []string `json:"topics"`
	Sentiments []string `json:"sentiments"`
}

// PredictResponse represents response from ML service
type PredictResponse struct {
	Predictions []ReviewPredictionOutput `json:"predictions"`
}
