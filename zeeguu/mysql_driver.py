import sys


def install_mysql_driver():
    """
    Ensure a working MySQL DB-API module is available as ``MySQLdb``.

    On some local Python 3.12 macOS setups, ``mysqlclient`` installs but fails
    at import time because of a client-library symbol mismatch. In that case we
    transparently fall back to PyMySQL so the app can still boot.
    """
    try:
        import MySQLdb  # noqa: F401

        return "mysqlclient"
    except Exception:
        # Clear partially imported native modules before installing the fallback.
        sys.modules.pop("MySQLdb", None)
        sys.modules.pop("_mysql", None)

        import pymysql

        pymysql.install_as_MySQLdb()
        import MySQLdb  # noqa: F401

        return "pymysql"
