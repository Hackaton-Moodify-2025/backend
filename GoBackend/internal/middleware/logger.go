package middleware


import (
	"time"

	"github.com/gofiber/fiber/v3"
	"reviews-backend/pkg/logger"
)

// RequestLogger creates a logging middleware
func RequestLogger(log *logger.Logger) fiber.Handler {
	return fiber.Handler(func(c fiber.Ctx) error {
		start := time.Now()

		// Process request
		err := c.Next()

		// Calculate request duration
		duration := time.Since(start)

		// Log request details
		log.WithFields(map[string]interface{}{
			"method":     c.Method(),
			"path":       c.Path(),
			"status":     c.Response().StatusCode(),
			"duration":   duration.Milliseconds(),
			"ip":         c.IP(),
			"user_agent": c.Get("User-Agent"),
			"size":       len(c.Response().Body()),
		}).Infof("%s %s - %d (%dms)", 
			c.Method(), 
			c.Path(), 
			c.Response().StatusCode(), 
			duration.Milliseconds())

		return err
	})
}

// ErrorHandler creates a global error handling middleware
func ErrorHandler(log *logger.Logger) fiber.ErrorHandler {
	return func(c fiber.Ctx, err error) error {
		code := fiber.StatusInternalServerError

		if e, ok := err.(*fiber.Error); ok {
			code = e.Code
		}

		log.WithError(err).WithFields(map[string]interface{}{
			"method": c.Method(),
			"path":   c.Path(),
			"status": code,
			"ip":     c.IP(),
		}).Errorf("Request failed")

		return c.Status(code).JSON(fiber.Map{
			"error":   true,
			"message": err.Error(),
		})
	}
}

// Recovery creates a recovery middleware for panic handling
func Recovery(log *logger.Logger) fiber.Handler {
	return fiber.Handler(func(c fiber.Ctx) (err error) {
		defer func() {
			if r := recover(); r != nil {
				log.WithFields(map[string]interface{}{
					"method": c.Method(),
					"path":   c.Path(),
					"ip":     c.IP(),
					"panic":  r,
				}).Errorf("Panic recovered")

				err = c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
					"error":   true,
					"message": "Internal server error",
				})
			}
		}()

		return c.Next()
	})
}