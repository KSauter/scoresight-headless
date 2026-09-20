from __future__ import annotations

import argparse
import logging
import threading
import webbrowser
from dataclasses import replace
from pathlib import Path

import uvicorn

from scoresight.core.config import ConfigStore
from scoresight.core.deployment import DeploymentSettings
from scoresight.core.logging import configure_logging
from scoresight.web.app import create_app


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run the ScoreSight OCR service")
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=18099)
    result.add_argument("--config", type=Path)
    result.add_argument("--data-dir", type=Path)
    result.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )
    result.add_argument(
        "--open-browser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="open the signed-in dashboard on startup (default: enabled)",
    )
    return result


def _announce(
    store: ConfigStore, deployment: DeploymentSettings, host: str, port: int
) -> str | None:
    """Report where the service lives and how to sign in.

    Returns the self-signing URL, or None when local tokens are not in use.

    Without this the first run is a dead end: the configuration is created
    under a platform-specific directory the operator has no reason to guess,
    and the generated administrator token is only readable inside that file.

    The token is written to the console rather than rendered in the web UI on
    purpose. Anyone who can read this output already has access to the
    configuration file; showing it in the UI would hand it to every client that
    can reach the port, which defeats the token once --host is opened up.
    """
    logger = logging.getLogger("scoresight")
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host

    logger.info("Configuration: %s", store.path)
    logger.info("Web interface: http://%s:%d", display_host, port)
    if host in {"0.0.0.0", "::"}:
        logger.info("Listening on %s: reachable from other machines", host)

    if deployment.auth_mode != "token":
        return None

    token = store.load().security.admin_token
    logger.info("Administrator token: %s", token)
    # Selecting text out of a double-clicked console window requires QuickEdit
    # to be enabled first, so the token is also offered as a link that signs in
    # on its own. Log redaction strips the query string from access logs.
    sign_in_url = f"http://{display_host}:{port}/login?token={token}"
    logger.info("Sign in directly: %s", sign_in_url)
    return sign_in_url


def main() -> None:
    args = parser().parse_args()
    deployment = DeploymentSettings.from_env()
    if args.data_dir is not None:
        deployment = replace(deployment, data_dir=args.data_dir)
    configure_logging(args.log_level, json_logs=deployment.json_logs)
    app = create_app(args.config, deployment=deployment)
    sign_in_url = _announce(app.state.config_store, deployment, args.host, args.port)
    if sign_in_url and args.open_browser:
        # uvicorn.run blocks, so the browser is opened from a timer. The delay
        # covers startup; an early request would just fail to connect.
        threading.Timer(1.5, webbrowser.open, args=(sign_in_url,)).start()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        workers=1,
        proxy_headers=True,
        forwarded_allow_ips=deployment.trusted_proxies,
        access_log=deployment.access_log,
        log_config=None,
    )


if __name__ == "__main__":
    main()
