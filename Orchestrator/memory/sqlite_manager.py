import sqlite3
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime

from orchestartor_agent_ollamaV2 import (
    attackVector,
    portInfo,
    hostMemory,
    vulnerability,
)

DB_PATH = Path("Orchestartor/memory/orchestartor_memory.db")


# ------------------------------------------------------------------------------------------- #
#                                     DB initialization                                       #
# ------------------------------------------------------------------------------------------- #
def initializeDB():
    connect = createDBconnection()
    cursor = connect.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS hosts (
            ip TEXT PRIMARY KEY,
            status TEXT,
            os_guess TEXT,
            first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            host_ip TEXT,
            port INTEGER,
            service TEXT,
            version TEXT,
            state TEXT,
            UNIQUE(host_ip, port),
            FOREIGN KEY(host_ip) REFERENCES hosts(ip)
        );
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS attack_vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            method TEXT,
            parameters TEXT,
            origins TEXT
        );
        """
    )

    cursor.execute(
        """
       CREATE TABLE IF NOT EXISTS vulnerabilities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            host TEXT,
            url TEXT,
            parameters TEXT,
            vulner_type TEXT,
            severity TEXT,
            evidence TEXT
        );
       """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT,
            success INTEGER,
            fail_reason TEXT,
            summary TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ); 
        """
    )

    connect.commit()
    connect.close()


# ------------------------------------------------------------------------------------------- #
#                                       DB - store                                            #
# ------------------------------------------------------------------------------------------- #


def storeHosts(host_memory: hostMemory):
    connect = createDBconnection()
    cursor = connect.cursor()
    timeStamp = datetime.now().strftime("%d. %B %Y %H:%M:%S")

    cursor.execute(
        "INSERT OR IGNORE INTO hosts (ip, status, os_guess, first_seen) VALUES (?, ?, ?, ?)",
        (host_memory.ip, host_memory.status, host_memory.os_guess, timeStamp),
    )

    connect.commit()
    connect.close()


def storePorts(ports: portInfo):
    connect = createDBconnection()
    cursor = connect.cursor()

    cursor.execute(
        "INSERT OR IGNORE INTO ports (host_ip, port, service, version, state) VALUES (?,?, ?, ?, ?)",
        (ports.host_ip, ports.port, ports.service, ports.version, ports.state),
    )
    connect.commit()
    connect.execute()


def storeAttackVector(attack_vector: attackVector):
    connect = createDBconnection()
    cursor = connect.cursor()

    cursor.execute(
        "INSERT OR IGNORE INTO attack_vectors (url, method, parameters, origins) VALUES (?, ?, ?, ?)",
        attack_vector.url,
        attack_vector.method,
        attack_vector.parameters,
        attack_vector.origins,
    )

    connect.commit()
    connect.close()


def storeVulnerability(vulner: vulnerability):
    connect = createDBconnection()
    cursor = connect.cursor()

    cursor.execute(
        "INSERT OR IGNORE INTO vulnerabilities (host, url, parameters, vulner_type, severity, evidence) VALUES (?, ?, ?, ?, ?, ?)",
        (
            vulner.host,
            vulner.url,
            vulner.parameters,
            vulner.vulner_type,
            vulner.severity,
            vulner.evidence,
        ),
    )

    connect.commit()
    connect.close()


# ------------------------------------------------------------------------------------------- #
#                                      DB - retrieve                                          #
# ------------------------------------------------------------------------------------------- #
def getHosts() -> List[Dict]:
    connect = createDBconnection()
    cursor = connect.cursor()

    cursor.execute("SELECT * FROM hosts")

    return [dict(row) for row in cursor.fetchall()]


def getPorts(host_ip: str = None) -> List[Dict]:
    connect = createDBconnection()
    cursor = connect.cursor()

    if host_ip:
        cursor.execute("SELECT * FROM ports WHERE host_ip = ?", (host_ip,))
    else:
        cursor.execute("SELECT * FROM ports")

    return [dict(row) for row in cursor.fetchall()]


def getAttackVector() -> List[Dict]:
    connect = createDBconnection()
    cursor = connect.cursor()

    cursor.execute("SELECT * FROM attack_vectors")
    return [dict(row) for row in cursor.fetchall()]


def getVulnerability() -> List[Dict]:
    connect = createDBconnection()
    cursor = connect.cursor()

    cursor.execute("SELECT * FROM vulnerabilities")
    return [dict(row) for row in cursor.fetchall()]


# ------------------------------------------------------------------------------------------- #
#                                     Helper function                                         #
# ------------------------------------------------------------------------------------------- #
def createDBconnection():
    connect = sqlite3.connect(DB_PATH, timeout=60, check_same_thread=False)
    connect.row_factory = sqlite3.Row
    connect.execute("PRAGMA foreign_keys = ON")
    return connect
