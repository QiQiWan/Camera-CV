# -*- coding: utf-8 -*-
from __future__ import annotations

import subprocess
from typing import Any


def run_nmcli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["nmcli", *args], text=True, capture_output=True, check=False)


def ensure_ap(config: dict[str, Any]) -> dict[str, Any]:
    ap = dict(config or {})
    if not bool(ap.get("enabled", True)):
        return {"ok": True, "enabled": False, "message": "AP disabled by config"}
    name = str(ap.get("connection_name") or "CameraCV-Pi-AP")
    ssid = str(ap.get("ssid") or "CameraCV-Pi")
    password = str(ap.get("password") or "88888888")
    interface = str(ap.get("interface") or "wlan0")
    address = str(ap.get("address") or "10.42.0.1")
    prefix = int(ap.get("prefix", 24) or 24)
    channel = int(ap.get("channel", 6) or 6)

    existing = run_nmcli(["-t", "-f", "NAME", "connection", "show"])
    if existing.returncode != 0:
        return {"ok": False, "message": existing.stderr.strip() or existing.stdout.strip()}
    names = {line.strip() for line in existing.stdout.splitlines() if line.strip()}
    if name not in names:
        add = run_nmcli([
            "connection", "add",
            "type", "wifi",
            "ifname", interface,
            "con-name", name,
            "autoconnect", "yes",
            "ssid", ssid,
        ])
        if add.returncode != 0:
            return {"ok": False, "message": add.stderr.strip() or add.stdout.strip()}

    commands = [
        ["connection", "modify", name, "802-11-wireless.mode", "ap"],
        ["connection", "modify", name, "802-11-wireless.band", "bg"],
        ["connection", "modify", name, "802-11-wireless.channel", str(channel)],
        ["connection", "modify", name, "ipv4.method", "shared"],
        ["connection", "modify", name, "ipv4.addresses", f"{address}/{prefix}"],
        ["connection", "modify", name, "wifi-sec.key-mgmt", "wpa-psk"],
        ["connection", "modify", name, "wifi-sec.psk", password],
        ["connection", "up", name],
    ]
    for command in commands:
        result = run_nmcli(command)
        if result.returncode != 0:
            return {"ok": False, "message": result.stderr.strip() or result.stdout.strip(), "command": command}
    return {"ok": True, "enabled": True, "ssid": ssid, "address": address, "connection_name": name}
