def test_install_mysql_driver_returns_a_working_driver_name():
    from zeeguu.mysql_driver import install_mysql_driver

    driver_name = install_mysql_driver()

    assert driver_name in {"mysqlclient", "pymysql"}

    import MySQLdb  # noqa: F401
