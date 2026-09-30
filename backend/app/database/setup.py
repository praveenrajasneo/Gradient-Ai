"""Create gradient_nova; --local makes an isolated Windows dev cluster on 5433."""

import argparse
import os
import secrets
import subprocess
from pathlib import Path

import psycopg
from dotenv import set_key
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from app.config import PROJECT_ROOT


def local_cluster() -> str:
    root = PROJECT_ROOT / ".local" / "postgres"
    data = root / "data"
    binaries = Path(os.getenv("POSTGRES_BIN", r"C:\Program Files\PostgreSQL\18\bin"))
    root.mkdir(parents=True, exist_ok=True)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    if not (data / "PG_VERSION").exists():
        if os.getenv("DATABASE_ADMIN_URL"):
            raise RuntimeError("An admin connection is already configured; omit --local to use it")
        password = secrets.token_urlsafe(32)
        secret_file = root / "init-password.tmp"
        try:
            secret_file.write_text(password, encoding="utf-8")
            subprocess.run([str(binaries / "initdb.exe"), "-D", str(data),
                            "-U", "gradient_nova", "--pwfile", str(secret_file),
                            "--auth-host=scram-sha-256", "--auth-local=scram-sha-256",
                            "--encoding=UTF8", "--locale=C"], check=True, creationflags=flags)
        finally:
            secret_file.unlink(missing_ok=True)
        admin_url = f"postgresql://gradient_nova:{password}@127.0.0.1:5433/postgres"
        set_key(str(PROJECT_ROOT / ".env"), "DATABASE_ADMIN_URL", admin_url)
        set_key(str(PROJECT_ROOT / ".env"), "DATABASE_URL", admin_url.rsplit("/", 1)[0] + "/gradient_nova")
    else:
        admin_url = os.getenv("DATABASE_ADMIN_URL")
        if not admin_url:
            raise RuntimeError("Existing local cluster needs its saved DATABASE_ADMIN_URL")
    status = subprocess.run([str(binaries / "pg_ctl.exe"), "-D", str(data), "status"],
                            capture_output=True, creationflags=flags)
    if status.returncode:
        subprocess.run([str(binaries / "pg_ctl.exe"), "-D", str(data), "-l", str(root / "server.log"),
                        "-o", "-p 5433 -h 127.0.0.1 -c shared_buffers=32MB -c max_connections=20",
                        "-w", "start"], check=True, creationflags=flags)
    return admin_url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()
    admin_url = local_cluster() if args.local else os.getenv("DATABASE_ADMIN_URL")
    if not admin_url:
        raise RuntimeError("Set DATABASE_ADMIN_URL or use --local")
    with psycopg.connect(admin_url, autocommit=True, connect_timeout=5) as connection:
        if not connection.execute("SELECT 1 FROM pg_database WHERE datname = %s", ("gradient_nova",)).fetchone():
            connection.execute("CREATE DATABASE gradient_nova")
    config = conninfo_to_dict(admin_url)
    config["dbname"] = "gradient_nova"
    app_url = make_conninfo(**config)
    if not os.getenv("DATABASE_URL") and not args.local:
        set_key(str(PROJECT_ROOT / ".env"), "DATABASE_URL", app_url)
    with psycopg.connect(app_url, connect_timeout=5) as connection:
        connection.execute(Path(__file__).with_name("schema.sql").read_text(encoding="utf-8"))
    print("gradient_nova database and schema ready. Credentials are saved only in .env.")


if __name__ == "__main__":
    main()
