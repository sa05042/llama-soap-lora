"""
Utility helper functions.
"""

import re


def format_soap(text):
    """
    Ensure SOAP sections appear on separate lines.
    """

    text = re.sub(r"\s*S:", "\nS:", text)
    text = re.sub(r"\s*O:", "\nO:", text)
    text = re.sub(r"\s*A:", "\nA:", text)
    text = re.sub(r"\s*P:", "\nP:", text)

    return re.sub(r"\n+", "\n", text).strip()