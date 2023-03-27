import psycopg2
import psycopg2.extras as extras
import os
import glob
import warnings
import pandas as pd
from dotenv import load_dotenv


class DB:
    """
    Class to define an interface to a Postgress database running on a
    remote server. Connection details of the database are loaded from a
    ".env" file with an entry like:

    # Database access credentials
    ML4FLOODS_DB_HOST="127.0.0.1"
    ML4FLOODS_DB_NAME="database_name"
    ML4FLOODS_DB_USER="db_user"
    ML4FLOODS_DB_PWD="<db_access_pass>"

    """

    def __init__(self, dotenv_path=".env"):
        """
        Load the DB connection details and initialise a DB cursor.
        """

        success_load = load_dotenv(dotenv_path, override=True)
        if not success_load:
            e = "[ERR] Failed to load the '{}' file.".format(dotenv_path)
            raise Exception(e)
        db_details_available = self._check_credentials_env()
        if not db_details_available:
            e = "[ERR] DB details missing from '{}' file.".format(dotenv_path)
            raise Exception(e)
        print("[INFO] Connecting to DB '{}'.".format(
            os.environ["ML4FLOODS_DB_NAME"]))
        self.conn = psycopg2.connect(
            host=os.environ["ML4FLOODS_DB_HOST"],
            database=os.environ["ML4FLOODS_DB_NAME"],
            user=os.environ["ML4FLOODS_DB_USER"],
            password=os.environ["ML4FLOODS_DB_PWD"])
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        print("[INFO] Connection successfully established.")

    def run_query(self, query, data=None, fetch=False):
        """
        Runs a SQL query on the DB and returns a DataFrame with results.
        """
        cur = self.conn.cursor()
        try:
            cur.execute(query, data)
            if fetch:
                df = pd.DataFrame(cur.fetchall(),
                            columns=[desc[0] for desc in cur.description])
                cur.close()
                return df
            else:
                cur.close()
                return
        except Exception as e:
            print("[ERR] SQL query failed: \n")
            print(e)
            return False

    def run_batch_insert(self, query, data=None, page_size=100):
        """
        Insert multiple rows into the database in a single query.
        """
        cur = self.conn.cursor()
        try:
            extras.execute_batch(cur, query, data)
            cur.close()
            return
        except Exception as e:
            print("[ERR] SQL query failed: \n")
            print(e)
            return False

    def close_connection(self):
        """
        Close the DB connection cleanly.
        """
        self.conn.close()

    def _check_credentials_env(self):
        """
        Check the DB credential environment variables exist.
        """
        required_keys = ["ML4FLOODS_DB_HOST",
                         "ML4FLOODS_DB_NAME",
                         "ML4FLOODS_DB_USER",
                         "ML4FLOODS_DB_PWD"]
        env_keys = list(os.environ.keys())
        return all(k in env_keys for k in required_keys)
