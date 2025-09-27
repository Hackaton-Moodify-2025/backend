package config

import (
	"fmt"
	"os"
	"strconv"

	"gopkg.in/yaml.v3"
	"reviews-backend/pkg/logger"
)

// Config holds the application configuration
type Config struct {
	Server   ServerConfig        `yaml:"server"`
	Data     DataConfig          `yaml:"data"`
	Logger   logger.Config       `yaml:"logger"`
	Database DatabaseConfig      `yaml:"database"`
}

// ServerConfig holds server-specific configuration
type ServerConfig struct {
	Host            string `yaml:"host"`
	Port            int    `yaml:"port"`
	ReadTimeout     int    `yaml:"read_timeout"`     // seconds
	WriteTimeout    int    `yaml:"write_timeout"`    // seconds
	IdleTimeout     int    `yaml:"idle_timeout"`     // seconds
	ShutdownTimeout int    `yaml:"shutdown_timeout"` // seconds
	Prefork         bool   `yaml:"prefork"`          // enable prefork for production
}

// DataConfig holds data-specific configuration
type DataConfig struct {
	ReviewsPath     string `yaml:"reviews_path"`
	PredictionsPath string `yaml:"predictions_path"`
}

// DatabaseConfig holds database-specific configuration (for future use)
type DatabaseConfig struct {
	Driver          string `yaml:"driver"`
	Host            string `yaml:"host"`
	Port            int    `yaml:"port"`
	Name            string `yaml:"name"`
	User            string `yaml:"user"`
	Password        string `yaml:"password"`
	MaxOpenConns    int    `yaml:"max_open_conns"`
	MaxIdleConns    int    `yaml:"max_idle_conns"`
	ConnMaxLifetime int    `yaml:"conn_max_lifetime"` // minutes
}

// LoadConfig loads configuration from YAML file with environment variable overrides
func LoadConfig(configPath string) (*Config, error) {
	// Default configuration
	config := &Config{
		Server: ServerConfig{
			Host:            "0.0.0.0",
			Port:            8080,
			ReadTimeout:     10,
			WriteTimeout:    10,
			IdleTimeout:     60,
			ShutdownTimeout: 30,
			Prefork:         false,
		},
		Data: DataConfig{
			ReviewsPath:     "data/siteReviews.json",
			PredictionsPath: "data/reviews.json",
		},
		Logger: logger.Config{
			Level:      "info",
			Format:     "console",
			OutputPath: "",
		},
		Database: DatabaseConfig{
			Driver:          "postgres",
			Host:            "localhost",
			Port:            5432,
			Name:            "reviews_db",
			User:            "postgres",
			Password:        "",
			MaxOpenConns:    25,
			MaxIdleConns:    5,
			ConnMaxLifetime: 5,
		},
	}

	// Load from YAML file if exists
	if configPath != "" {
		if err := loadFromYAML(config, configPath); err != nil {
			return nil, fmt.Errorf("failed to load config from YAML: %w", err)
		}
	}

	// Override with environment variables
	overrideWithEnv(config)

	return config, nil
}

// loadFromYAML loads configuration from YAML file
func loadFromYAML(config *Config, configPath string) error {
	file, err := os.Open(configPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // File doesn't exist, use defaults
		}
		return err
	}
	defer file.Close()

	decoder := yaml.NewDecoder(file)
	return decoder.Decode(config)
}

// overrideWithEnv overrides configuration with environment variables
func overrideWithEnv(config *Config) {
	// Server config
	if host := getEnv("SERVER_HOST"); host != "" {
		config.Server.Host = host
	}
	if port := getEnvAsInt("SERVER_PORT"); port != 0 {
		config.Server.Port = port
	}
	if timeout := getEnvAsInt("SERVER_READ_TIMEOUT"); timeout != 0 {
		config.Server.ReadTimeout = timeout
	}
	if timeout := getEnvAsInt("SERVER_WRITE_TIMEOUT"); timeout != 0 {
		config.Server.WriteTimeout = timeout
	}
	if timeout := getEnvAsInt("SERVER_IDLE_TIMEOUT"); timeout != 0 {
		config.Server.IdleTimeout = timeout
	}
	if timeout := getEnvAsInt("SERVER_SHUTDOWN_TIMEOUT"); timeout != 0 {
		config.Server.ShutdownTimeout = timeout
	}
	if prefork := getEnv("SERVER_PREFORK"); prefork == "true" {
		config.Server.Prefork = true
	}

	// Data config
	if path := getEnv("REVIEWS_PATH"); path != "" {
		config.Data.ReviewsPath = path
	}
	if path := getEnv("PREDICTIONS_PATH"); path != "" {
		config.Data.PredictionsPath = path
	}

	// Logger config
	if level := getEnv("LOG_LEVEL"); level != "" {
		config.Logger.Level = level
	}
	if format := getEnv("LOG_FORMAT"); format != "" {
		config.Logger.Format = format
	}
	if path := getEnv("LOG_OUTPUT_PATH"); path != "" {
		config.Logger.OutputPath = path
	}

	// Database config
	if driver := getEnv("DB_DRIVER"); driver != "" {
		config.Database.Driver = driver
	}
	if host := getEnv("DB_HOST"); host != "" {
		config.Database.Host = host
	}
	if port := getEnvAsInt("DB_PORT"); port != 0 {
		config.Database.Port = port
	}
	if name := getEnv("DB_NAME"); name != "" {
		config.Database.Name = name
	}
	if user := getEnv("DB_USER"); user != "" {
		config.Database.User = user
	}
	if password := getEnv("DB_PASSWORD"); password != "" {
		config.Database.Password = password
	}
}

// getEnv gets environment variable
func getEnv(key string) string {
	return os.Getenv(key)
}

// getEnvAsInt gets environment variable as int
func getEnvAsInt(key string) int {
	valueStr := getEnv(key)
	if valueStr == "" {
		return 0
	}
	value, err := strconv.Atoi(valueStr)
	if err != nil {
		return 0
	}
	return value
}