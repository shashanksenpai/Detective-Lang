"""pytest setup shared by every test module.

Points the app at a throwaway SQLite database *before* any test module imports
db.py, so the suite can never read or write the real detective.db (BACKLOG F-15).
It overrides rather than defaults on purpose: a DETECTIVE_DATABASE_URL left over
in someone's shell must not send the tests at real data.
"""
import os
import tempfile

_TEST_DB = os.path.join(tempfile.mkdtemp(prefix="detective-tests-"), "test.db")
os.environ["DETECTIVE_DATABASE_URL"] = "sqlite:///" + _TEST_DB.replace("\\", "/")
