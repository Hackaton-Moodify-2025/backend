package main


import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gofiber/fiber/v3"
	"github.com/gofiber/fiber/v3/middleware/compress"
	"github.com/gofiber/fiber/v3/middleware/cors"
	"github.com/gofiber/fiber/v3/middleware/helmet"
	"github.com/gofiber/fiber/v3/middleware/limiter"
	"github.com/gofiber/fiber/v3/middleware/recover"

	"reviews-backend/internal/config"
	"reviews-backend/internal/handlers"
	"reviews-backend/internal/middleware"
	"reviews-backend/internal/repository"
	"reviews-backend/internal/service"
	"reviews-backend/pkg/logger"
)

func main() {
	// Load configuration
	cfg, err := config.LoadConfig("config.yaml")
	if err != nil {
		fmt.Printf("Failed to load config: %v\n", err)
		os.Exit(1)
	}

	// Initialize logger
	log, err := logger.NewLogger(cfg.Logger)
	if err != nil {
		fmt.Printf("Failed to initialize logger: %v\n", err)
		os.Exit(1)
	}
	defer log.Close()

	log.Infof("Starting reviews backend server...")
	log.Infof("Server will listen on %s:%d", cfg.Server.Host, cfg.Server.Port)

	// Initialize repository
	repo := repository.NewJSONReviewRepository(cfg.Data.ReviewsPath, cfg.Data.PredictionsPath)

	// Initialize service
	reviewService := service.NewReviewService(repo, log)

	// Initialize handlers
	reviewHandler := handlers.NewReviewHandler(reviewService, log)

	// Create Fiber app with configuration
	app := fiber.New(fiber.Config{
		ReadTimeout:  time.Duration(cfg.Server.ReadTimeout) * time.Second,
		WriteTimeout: time.Duration(cfg.Server.WriteTimeout) * time.Second,
		IdleTimeout:  time.Duration(cfg.Server.IdleTimeout) * time.Second,
		ErrorHandler: middleware.ErrorHandler(log),
		AppName:      "Reviews Backend API v1.0.0",
	})

	// Setup middleware
	setupMiddleware(app, log)

	// Setup routes
	setupRoutes(app, reviewHandler)

	// Start server in goroutine
	go func() {
		addr := fmt.Sprintf("%s:%d", cfg.Server.Host, cfg.Server.Port)
		if err := app.Listen(addr); err != nil {
			log.WithError(err).Fatalf("Failed to start server")
		}
	}()

	log.Infof("Server started successfully")

	// Wait for interrupt signal for graceful shutdown
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)

	<-c
	log.Infof("Received shutdown signal, shutting down gracefully...")

	// Graceful shutdown with timeout
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(cfg.Server.ShutdownTimeout)*time.Second)
	defer cancel()

	if err := app.ShutdownWithContext(ctx); err != nil {
		log.WithError(err).Errorf("Server forced to shutdown")
	} else {
		log.Infof("Server shutdown completed")
	}
}

// setupMiddleware configures all middleware
func setupMiddleware(app *fiber.App, log *logger.Logger) {
	// Recovery middleware (must be first)
	app.Use(recover.New())

	// Custom recovery middleware with logging
	app.Use(middleware.Recovery(log))

	// Security middleware
	app.Use(helmet.New())

	// CORS middleware
	app.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"http://localhost:3000", "http://localhost:5173", "https://your-frontend-domain.com"},
		AllowMethods:     []string{"GET", "POST", "HEAD", "PUT", "DELETE", "PATCH", "OPTIONS"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept", "Authorization", "X-Requested-With"},
		AllowCredentials: true,
		MaxAge:           86400, // 24 hours
	}))

	// Compression middleware
	app.Use(compress.New(compress.Config{
		Level: compress.LevelBestSpeed,
	}))

	// Rate limiting middleware
	app.Use(limiter.New(limiter.Config{
		Max:        100,                // Max requests per window
		Expiration: time.Minute,        // Window duration
		KeyGenerator: func(c fiber.Ctx) string {
			return c.IP()
		},
		LimitReached: func(c fiber.Ctx) error {
			log.WithField("ip", c.IP()).Warnf("Rate limit exceeded")
			return c.Status(fiber.StatusTooManyRequests).JSON(fiber.Map{
				"error":   "rate_limit_exceeded",
				"message": "Too many requests, please try again later",
			})
		},
	}))

	// Request logging middleware (should be last)
	app.Use(middleware.RequestLogger(log))
}

// setupRoutes configures all routes
func setupRoutes(app *fiber.App, reviewHandler *handlers.ReviewHandler) {
	// Health check
	app.Get("/health", reviewHandler.HealthCheck)

	// API routes
	api := app.Group("/api/v1")
	
	// Reviews routes
	api.Get("/reviews", reviewHandler.GetReviews)
	api.Get("/analytics", reviewHandler.GetAnalytics)

	// Catch-all for 404
	app.Use("*", func(c fiber.Ctx) error {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{
			"error":   "not_found",
			"message": "Route not found",
		})
	})
}