CREATE DATABASE IF NOT EXISTS futurequant
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE futurequant;

CREATE TABLE IF NOT EXISTS daily_price (
  id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
  symbol VARCHAR(32) NOT NULL,
  `date` DATE NOT NULL,
  `open` DECIMAL(18,6) NOT NULL,
  high DECIMAL(18,6) NOT NULL,
  low DECIMAL(18,6) NOT NULL,
  `close` DECIMAL(18,6) NOT NULL,
  volume BIGINT NULL,
  open_interest BIGINT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uk_symbol_date (symbol, `date`),
  KEY idx_symbol (symbol),
  KEY idx_date (`date`)
);

GRANT ALL PRIVILEGES ON futurequant.* TO 'novel_user'@'%';
FLUSH PRIVILEGES;
