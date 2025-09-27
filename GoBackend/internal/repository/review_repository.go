package repository


import (
	"encoding/json"
	"fmt"
	"os"
	"sync"

	"reviews-backend/internal/models"
)

// ReviewRepository handles review data operations
type ReviewRepository interface {
	GetReviews(offset, limit int, topic, sentiment string) ([]models.Review, error)
	GetTotalReviews(topic, sentiment string) (int, error)
	GetAllReviewsForAnalytics() ([]models.Review, []models.ReviewPrediction, error)
}

// JSONReviewRepository implements ReviewRepository using JSON files
type JSONReviewRepository struct {
	reviewsPath     string
	predictionsPath string
	reviews         []models.Review
	predictions     []models.ReviewPrediction
	mu              sync.RWMutex
	loaded          bool
}

// NewJSONReviewRepository creates a new JSON review repository
func NewJSONReviewRepository(reviewsPath, predictionsPath string) *JSONReviewRepository {
	return &JSONReviewRepository{
		reviewsPath:     reviewsPath,
		predictionsPath: predictionsPath,
	}
}

// loadData loads data from JSON files (thread-safe)
func (r *JSONReviewRepository) loadData() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.loaded {
		return nil
	}

	// Load reviews
	if err := r.loadReviews(); err != nil {
		return fmt.Errorf("failed to load reviews: %w", err)
	}

	// Load predictions
	if err := r.loadPredictions(); err != nil {
		return fmt.Errorf("failed to load predictions: %w", err)
	}

	// Merge reviews with predictions
	r.mergeData()

	r.loaded = true
	return nil
}

// loadReviews loads reviews from JSON file
func (r *JSONReviewRepository) loadReviews() error {
	file, err := os.Open(r.reviewsPath)
	if err != nil {
		return err
	}
	defer file.Close()

	var data struct {
		Reviews []models.Review `json:"reviews"`
	}

	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&data); err != nil {
		return err
	}

	r.reviews = data.Reviews
	return nil
}

// loadPredictions loads predictions from JSON file
func (r *JSONReviewRepository) loadPredictions() error {
	file, err := os.Open(r.predictionsPath)
	if err != nil {
		return err
	}
	defer file.Close()

	var data []models.ReviewPrediction
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&data); err != nil {
		return err
	}

	r.predictions = data
	return nil
}

// mergeData merges reviews with their predictions
func (r *JSONReviewRepository) mergeData() {
	// Create a map for faster lookup
	predMap := make(map[int]models.ReviewPrediction)
	for _, pred := range r.predictions {
		predMap[pred.ID] = pred
	}

	// Merge data
	for i, review := range r.reviews {
		if pred, exists := predMap[review.ID]; exists {
			r.reviews[i].Topics = pred.Topics
			r.reviews[i].Sentiments = pred.Sentiments
		}
	}
}

// GetReviews returns paginated reviews with optional filtering
func (r *JSONReviewRepository) GetReviews(offset, limit int, topic, sentiment string) ([]models.Review, error) {
	if err := r.loadData(); err != nil {
		return nil, err
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	// Filter reviews
	var filtered []models.Review
	for _, review := range r.reviews {
		if r.matchesFilter(review, topic, sentiment) {
			filtered = append(filtered, review)
		}
	}

	// Apply pagination
	if offset >= len(filtered) {
		return []models.Review{}, nil
	}

	end := offset + limit
	if end > len(filtered) {
		end = len(filtered)
	}

	return filtered[offset:end], nil
}

// GetTotalReviews returns total count of reviews with optional filtering
func (r *JSONReviewRepository) GetTotalReviews(topic, sentiment string) (int, error) {
	if err := r.loadData(); err != nil {
		return 0, err
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	if topic == "" && sentiment == "" {
		return len(r.reviews), nil
	}

	count := 0
	for _, review := range r.reviews {
		if r.matchesFilter(review, topic, sentiment) {
			count++
		}
	}

	return count, nil
}

// GetAllReviewsForAnalytics returns all reviews and predictions for analytics
func (r *JSONReviewRepository) GetAllReviewsForAnalytics() ([]models.Review, []models.ReviewPrediction, error) {
	if err := r.loadData(); err != nil {
		return nil, nil, err
	}

	r.mu.RLock()
	defer r.mu.RUnlock()

	// Return copies to avoid data races
	reviews := make([]models.Review, len(r.reviews))
	copy(reviews, r.reviews)

	predictions := make([]models.ReviewPrediction, len(r.predictions))
	copy(predictions, r.predictions)

	return reviews, predictions, nil
}

// matchesFilter checks if a review matches the given filters
func (r *JSONReviewRepository) matchesFilter(review models.Review, topic, sentiment string) bool {
	if topic != "" {
		found := false
		for _, t := range review.Topics {
			if t == topic {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	if sentiment != "" {
		found := false
		for _, s := range review.Sentiments {
			if s == sentiment {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	return true
}