from futureQuant.data.storage.mysql_price_loader import MySQLPriceLoaderConfig


class TestMySQLPriceLoaderConfig:
    def test_from_mapping_builds_config(self):
        config = MySQLPriceLoaderConfig.from_mapping(
            {
                "enabled": True,
                "host": "127.0.0.1",
                "port": 3306,
                "user": "novel_user",
                "password": "novel_password",
                "database": "futurequant",
                "table": "daily_price",
            }
        )

        assert config is not None
        assert config.host == "127.0.0.1"
        assert config.port == 3306
        assert config.user == "novel_user"
        assert config.database == "futurequant"
        assert config.table == "daily_price"
