package logger

import (
	"os"
	"path/filepath"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// Logger wraps zap.SugaredLogger with additional functionality
type Logger struct {
	*zap.SugaredLogger
	zapLogger *zap.Logger
}

// Config represents logger configuration
type Config struct {
	Level      string `yaml:"level"`
	Format     string `yaml:"format"` // json or console
	OutputPath string `yaml:"output_path"`
}

// NewLogger creates a new zap logger instance
func NewLogger(cfg Config) (*Logger, error) {
	// Parse log level
	level := zapcore.InfoLevel
	if err := level.UnmarshalText([]byte(cfg.Level)); err != nil {
		level = zapcore.InfoLevel
	}

	// Create encoder config
	var encoderConfig zapcore.EncoderConfig
	if cfg.Format == "json" {
		encoderConfig = zap.NewProductionEncoderConfig()
	} else {
		encoderConfig = zap.NewDevelopmentEncoderConfig()
		encoderConfig.EncodeLevel = zapcore.CapitalColorLevelEncoder
	}

	encoderConfig.TimeKey = "timestamp"
	encoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder
	encoderConfig.CallerKey = "caller"
	encoderConfig.EncodeCaller = zapcore.ShortCallerEncoder

	// Create encoder
	var encoder zapcore.Encoder
	if cfg.Format == "json" {
		encoder = zapcore.NewJSONEncoder(encoderConfig)
	} else {
		encoder = zapcore.NewConsoleEncoder(encoderConfig)
	}

	// Create writer syncer
	var writeSyncer zapcore.WriteSyncer
	if cfg.OutputPath != "" {
		// Ensure log directory exists
		if err := os.MkdirAll(filepath.Dir(cfg.OutputPath), 0755); err != nil {
			return nil, err
		}

		file, err := os.OpenFile(cfg.OutputPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
		if err != nil {
			return nil, err
		}

		writeSyncer = zapcore.NewMultiWriteSyncer(
			zapcore.AddSync(file),
			zapcore.AddSync(os.Stdout),
		)
	} else {
		writeSyncer = zapcore.AddSync(os.Stdout)
	}

	// Create core
	core := zapcore.NewCore(encoder, writeSyncer, level)

	// Create logger with caller info
	zapLogger := zap.New(core, zap.AddCaller(), zap.AddStacktrace(zapcore.ErrorLevel))

	return &Logger{
		SugaredLogger: zapLogger.Sugar(),
		zapLogger:     zapLogger,
	}, nil
}

// NewDevelopmentLogger creates a logger for development
func NewDevelopmentLogger() (*Logger, error) {
	config := Config{
		Level:      "debug",
		Format:     "console",
		OutputPath: "",
	}
	return NewLogger(config)
}

// NewProductionLogger creates a logger for production
func NewProductionLogger(outputPath string) (*Logger, error) {
	config := Config{
		Level:      "info",
		Format:     "json",
		OutputPath: outputPath,
	}
	return NewLogger(config)
}

// Close gracefully closes the logger
func (l *Logger) Close() error {
	return l.zapLogger.Sync()
}

// WithField creates a new logger with the given field
func (l *Logger) WithField(key string, value interface{}) *Logger {
	return &Logger{
		SugaredLogger: l.SugaredLogger.With(key, value),
		zapLogger:     l.zapLogger,
	}
}

// WithFields creates a new logger with the given fields
func (l *Logger) WithFields(fields map[string]interface{}) *Logger {
	var args []interface{}
	for k, v := range fields {
		args = append(args, k, v)
	}
	return &Logger{
		SugaredLogger: l.SugaredLogger.With(args...),
		zapLogger:     l.zapLogger,
	}
}

// WithError creates a new logger with error field
func (l *Logger) WithError(err error) *Logger {
	return l.WithField("error", err.Error())
}