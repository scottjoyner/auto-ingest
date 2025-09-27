#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backwards compatible entry point for the transcript quality UI/API."""

from __future__ import annotations

import os

from quality_api import create_app

app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=True)
