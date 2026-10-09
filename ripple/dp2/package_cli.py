"""CLI for the M2 Rubin DP2 masked-image cutout package."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from .client import Dp2Client
from .errors import Dp2AuthenticationError, Dp2ConfigurationError, Dp2Error
from .models import (
    DP2_EFFECTIVE_WAVELENGTH_M_BY_BAND,
    Dp2ClientConfig,
    Dp2CutoutRequest,
    SecurityEvidence,
)
from .package_models import Dp2PackageFailureEvidence
from .package_service import Dp2PackageService, write_package_model_json_atomic
from .service import create_private_run_directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve one typed g/r/i Rubin DP2 deep-coadd masked image and construct "
            "the immutable RIPPLe M2 cutout package. Authentication is read only "
            "from RSP_TOKEN."
        )
    )
    parser.add_argument(
        "--band",
        choices=tuple(DP2_EFFECTIVE_WAVELENGTH_M_BY_BAND),
        default="r",
        help="Rubin LSSTCam band selected by its pinned effective wavelength.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/dp2_m2"),
        help="Local root beneath which a new private M2 run directory is created.",
    )
    return parser


def _exit_code(error: Dp2Error) -> int:
    if isinstance(error, Dp2ConfigurationError):
        return 2
    if isinstance(error, Dp2AuthenticationError):
        return 3
    if error.stage in {"sia_query", "dataset_selection"}:
        return 4
    if error.stage.startswith(("fits", "masked", "mask", "variance", "celestial")):
        return 6
    if error.stage.startswith(("package", "archive", "calibration", "requested")):
        return 7
    return 5


def _security_evidence(token_present: bool) -> SecurityEvidence:
    return SecurityEvidence(
        credential_source="environment",
        credential_name="RSP_TOKEN",
        credential_was_present=token_present,
        credential_value_recorded=False,
        authorization_headers_recorded=False,
        access_urls_recorded=False,
    )


def _write_failure_safely(
    *,
    run_directory: Path,
    request: Dp2CutoutRequest,
    token_present: bool,
    stage: str,
    code: str,
    safe_message: str,
    http_status: int | None = None,
) -> Path | None:
    try:
        failure = Dp2PackageFailureEvidence(
            completed_at_utc=datetime.now(timezone.utc),
            stage=stage,
            code=code,
            safe_message=safe_message,
            http_status=http_status,
            request=request,
            security=_security_evidence(token_present),
        )
        path = run_directory / "failure.json"
        write_package_model_json_atomic(path, failure)
        return path
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    request = Dp2CutoutRequest(
        band_name=args.band,
        soda_service_type="cutout-sync-maskedimage",
    )
    token_present = bool(os.environ.get("RSP_TOKEN", "").strip())

    try:
        run_directory = create_private_run_directory(args.output_root)
    except Dp2Error as exc:
        print(f"DP2 M2 package: FAILED at {exc.stage} ({exc.code})", file=sys.stderr)
        print(
            "No evidence file was created because a private run directory was unavailable.",
            file=sys.stderr,
        )
        return _exit_code(exc)
    except Exception:
        print(
            "DP2 M2 package: FAILED before a private run directory was available",
            file=sys.stderr,
        )
        print("No evidence file was created.", file=sys.stderr)
        return 2

    try:
        client = Dp2Client.from_environment(Dp2ClientConfig())
        package = Dp2PackageService(client).run(request, run_directory)
    except (Dp2Error, ValidationError) as exc:
        if isinstance(exc, Dp2Error):
            stage = exc.stage
            code = exc.code
            safe_message = exc.safe_message
            http_status = exc.http_status
            exit_code = _exit_code(exc)
        else:
            stage = "package_contract"
            code = "invalid_pydantic_contract"
            safe_message = "A local M2 Pydantic package contract was invalid."
            http_status = None
            exit_code = 2
        failure_path = _write_failure_safely(
            run_directory=run_directory,
            request=request,
            token_present=token_present,
            stage=stage,
            code=code,
            safe_message=safe_message,
            http_status=http_status,
        )
        print(f"DP2 M2 package: FAILED at {stage} ({code})", file=sys.stderr)
        if failure_path is None:
            print("Sanitized failure evidence could not be written.", file=sys.stderr)
        else:
            print(f"sanitized evidence: {failure_path}", file=sys.stderr)
        return exit_code
    except Exception:
        failure_path = _write_failure_safely(
            run_directory=run_directory,
            request=request,
            token_present=token_present,
            stage="unexpected",
            code="unexpected_failure",
            safe_message="An unexpected M2 failure occurred; details were not serialized.",
        )
        print("DP2 M2 package: FAILED unexpectedly", file=sys.stderr)
        if failure_path is None:
            print("Sanitized failure evidence could not be written.", file=sys.stderr)
        else:
            print(f"sanitized evidence: {failure_path}", file=sys.stderr)
        return 10

    print("DP2 M2 package: SUCCESS")
    print(f"obs_id: {package.dataset.obs_id}")
    print(f"band: {package.dataset.band_name}")
    print(f"shape_yx: {package.image.shape_yx}")
    print(f"image_unit: {package.image.canonical_unit}")
    print(f"variance_unit: {package.variance.canonical_unit}")
    print(f"mask_bits: {len(package.mask.bits)}")
    print(f"psf: {package.psf.state}")
    print(f"FITS: {run_directory / package.artifact.filename}")
    print(f"package: {run_directory / 'package.json'}")
    print(f"fits_sha256: {package.artifact.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
