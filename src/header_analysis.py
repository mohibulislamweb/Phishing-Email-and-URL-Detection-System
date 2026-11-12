"""
Email Header & IP threat analysis module.
"""

import re
import ipaddress

PRIVATE_IP_NOTE = "Private / non-routable address (internal network or header spoof artifact)"

def extract_header(pattern: str, text: str) -> str:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else "N/A"

def extract_domain(email_addr: str) -> str:
    m = re.search(r"@([\w.-]+)", email_addr)
    return m.group(1).lower() if m else ""

def is_private_ip(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip).is_private
    except ValueError:
        return False