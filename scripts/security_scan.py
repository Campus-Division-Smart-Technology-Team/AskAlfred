#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Security dependency scanning script for Alfred V3.

This script performs comprehensive security checks on project dependencies:
1. Checks for known vulnerabilities using safety scan (new command)
2. Validates dependency pinning
3. Runs pip-audit checks
4. Generates detailed reports
5. Can be integrated into CI/CD pipelines

Usage:
    python security_scan.py [--json] [--strict] [--html]

Options:
    --json      Output results in JSON format
    --strict    Exit with non-zero code if any vulnerabilities found
    --html      Generate HTML report file

Note: Uses .safety-policy.json for vulnerability suppressions.
"""

from __future__ import annotations

import subprocess
import sys
import json
import argparse
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Any

# Force UTF-8 encoding for Windows compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
os.environ['PYTHONIOENCODING'] = 'utf-8'


class ProgressTracker:
    """Track progress of security scan with timing and visual indicators."""

    def __init__(self, total_checks: int = 3):
        """Initialize progress tracker."""
        self.total_checks = total_checks
        self.completed_checks = 0
        self.start_time = time.time()
        self.check_times: dict[str, float] = {}
        self.check_start: dict[str, float] = {}

    def start_check(self, check_name: str) -> None:
        """Mark the start of a check."""
        self.check_start[check_name] = time.time()
        elapsed = time.time() - self.start_time
        progress = (self.completed_checks / self.total_checks *
                    50) if self.total_checks > 0 else 0
        bar = "=" * int(progress) + "-" * (50 - int(progress))
        print(
            f"\n[{bar}] [{self.completed_checks}/{self.total_checks}] {check_name}...")
        sys.stdout.flush()

    def end_check(self, check_name: str, success: bool = True) -> None:
        """Mark the end of a check and update progress."""
        if check_name in self.check_start:
            elapsed = time.time() - self.check_start[check_name]
            self.check_times[check_name] = elapsed
        self.completed_checks += 1
        status = "[OK]" if success else "[!] "
        time_str = f"({self.check_times.get(check_name, 0):.1f}s)" if check_name in self.check_times else ""
        print(f"  {status} {check_name} {time_str}")
        sys.stdout.flush()

    def summary(self) -> str:
        """Get timing summary."""
        total_time = time.time() - self.start_time
        times_str = ", ".join(
            f"{name}: {duration:.1f}s"
            for name, duration in self.check_times.items()
        )
        return f"Total: {total_time:.1f}s ({times_str})"


def run_command(cmd: list[str]) -> tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            check=False
        )
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        return 1, "", f"Command not found: {cmd[0]}"


def check_safety_installed() -> bool:
    """Check if safety package is installed."""
    returncode, _, _ = run_command(
        [sys.executable, "-m", "safety", "--version"])
    return returncode == 0


def is_safety_policy_schema_error(stderr: str) -> bool:
    """Detect Safety policy schema/version mismatch errors."""
    lowered = stderr.lower()
    return (
        "unable to load the safety policy file" in lowered
        and "only supports version 3.0" in lowered
    )


def run_safety_scan(json_format: bool = False) -> dict[str, Any]:
    """Run safety scan for known vulnerabilities (new command)."""
    cmd = [sys.executable, "-m", "safety", "scan", "--target", "."]

    # Add policy file if it exists (use absolute path from script location)
    policy_file = Path(__file__).parent.parent / ".safety-policy.json"
    used_policy_file = False
    if policy_file.exists():
        cmd.extend(["--policy-file", str(policy_file)])
        used_policy_file = True

    if json_format:
        cmd.extend(["--output", "json"])
    else:
        cmd.append("--full-report")

    returncode, stdout, stderr = run_command(cmd)

    # Graceful fallback for policy schema incompatibility across Safety versions
    if used_policy_file and is_safety_policy_schema_error(stderr):
        retry_cmd = [sys.executable, "-m", "safety", "scan", "--target", "."]
        if json_format:
            retry_cmd.extend(["--output", "json"])
        else:
            retry_cmd.append("--detailed-output")

        retry_code, retry_stdout, retry_stderr = run_command(retry_cmd)

        warning = (
            "Safety policy file is incompatible with this Safety version; "
            "scan was retried without policy file. "
            "Update .safety-policy.json to Safety v3 schema to re-enable suppressions."
        )

        merged_error = warning
        if retry_stderr.strip():
            merged_error = f"{warning}\n{retry_stderr}"

        return {
            "tool": "safety-scan",
            "success": retry_code == 0,
            "returncode": retry_code,
            "output": retry_stdout,
            "error": merged_error,
            "used_policy_file": False,
            "policy_fallback": True,
        }

    return {
        "tool": "safety-scan",
        "success": returncode == 0,
        "returncode": returncode,
        "output": stdout,
        "error": stderr,
        "used_policy_file": used_policy_file,
        "policy_fallback": False,
    }


def run_pip_audit(json_format: bool = False) -> dict[str, Any]:
    """Run pip-audit for dependency security."""
    # Try executable first, then module invocation for environment consistency
    commands: list[list[str]] = [["pip-audit"],
                                 [sys.executable, "-m", "pip_audit"]]

    if json_format:
        commands = [cmd + ["--format", "json"] for cmd in commands]

    attempted: list[str] = []
    returncode = 1
    stdout = ""
    stderr = ""

    for cmd in commands:
        attempted.append(" ".join(cmd))
        returncode, stdout, stderr = run_command(cmd)

        # Continue trying only if the command itself is missing
        if "command not found" in stderr.lower():
            continue
        break

    pip_audit_missing = (
        ("command not found" in stderr.lower())
        or ("no module named pip_audit" in stderr.lower())
    )

    if pip_audit_missing and not stdout.strip():
        stderr = (
            "pip-audit is not installed in the active environment. "
            "Install with: python -m pip install pip-audit. "
            f"Attempted: {', '.join(attempted)}"
        )

    return {
        "tool": "pip-audit",
        "success": returncode == 0,
        "returncode": returncode,
        "output": stdout,
        "error": stderr,
        "attempted_commands": attempted,
    }


def check_requirements_pinning() -> dict[str, Any]:
    """Check that all dependencies in requirements.txt are pinned."""
    req_file = Path(__file__).parent.parent / "requirements.txt"

    if not req_file.exists():
        return {
            "tool": "requirements-pinning",
            "success": False,
            "error": f"requirements.txt not found at {req_file}"
        }

    unpinned = []
    pinned = []

    with open(req_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Skip platform-specific lines
            if ';' in line:
                line = line.split(';')[0].strip()

            # Check if pinned
            if '==' in line:
                pinned.append((line_num, line))
            elif any(op in line for op in ['>=', '<=', '>', '<', '~=']):
                unpinned.append((line_num, line))
            elif line and not line.startswith('-'):
                unpinned.append((line_num, line))

    return {
        "tool": "requirements-pinning",
        "success": len(unpinned) == 0,
        "pinned_count": len(pinned),
        "unpinned_count": len(unpinned),
        "unpinned_packages": unpinned,
    }


def generate_html_report(results: dict[str, Any], output_file: str = "security_report.html"):
    """Generate an HTML report of security scan results."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Alfred V3 Security Scan Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .summary {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            .success {{ color: green; }}
            .error {{ color: red; }}
            .warning {{ color: orange; }}
            pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
        </style>
    </head>
    <body>
        <h1>Alfred V3 Security Dependency Scan Report</h1>
        <p>Generated: {datetime.now().isoformat()}</p>
        <div class="summary">
            <h2>Summary</h2>
            {results.get('summary', 'No summary available')}
        </div>
        <h2>Detailed Results</h2>
        <pre>{json.dumps(results, indent=2)}</pre>
    </body>
    </html>
    """

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML report generated: {output_file}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Security dependency scanning for Alfred V3"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero code if any vulnerabilities found"
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML report"
    )
    args = parser.parse_args()

    print("[OK]  Alfred V3 - Security Dependency Scanner")
    print("=" * 60)
    print("[*] Starting security checks...\n")

    results = {
        "timestamp": datetime.now().isoformat(),
        "checks": []
    }

    # Initialize progress tracker
    tracker = ProgressTracker(total_checks=3)

    # Check 1: Requirements pinning
    tracker.start_check("Requirements pinning validation")
    pinning_result = check_requirements_pinning()
    tracker.end_check("Requirements pinning validation",
                      pinning_result["success"])
    results["checks"].append(pinning_result)

    if pinning_result["success"]:
        print(
            f"    {pinning_result['pinned_count']} packages verified")
    else:
        print(
            f"    Found {pinning_result['unpinned_count']} unpinned packages")

    # Check 2: Safety scan (new command)
    if check_safety_installed():
        tracker.start_check("Vulnerability database scan (safety)")
        safety_result = run_safety_scan(json_format=args.json)
        tracker.end_check("Vulnerability database scan (safety)",
                          safety_result["success"])
        results["checks"].append(safety_result)

        if safety_result["success"]:
            print("    No known vulnerabilities detected")
        else:
            print("    Vulnerabilities detected - review details below")
            if safety_result.get("policy_fallback"):
                print("    Note: policy file was skipped due to schema mismatch")
    else:
        print("\n[!] Safety not installed, skipping vulnerability check")
        print("    Install with: pip install safety")

    # Check 3: Pip audit (if available)
    tracker.start_check("Dependency audit scan (pip-audit)")
    pip_audit_result = run_pip_audit(json_format=args.json)
    tracker.end_check("Dependency audit scan (pip-audit)",
                      pip_audit_result["success"])
    results["checks"].append(pip_audit_result)
    if not pip_audit_result["success"] and "not installed" in pip_audit_result.get("error", "").lower():
        print("    pip-audit not installed in active environment")
        print("    Install with: python -m pip install pip-audit")

    # Summary
    total_checks = len(results["checks"])
    successful_checks = sum(
        1 for check in results["checks"] if check.get("success", False))

    summary = f"{successful_checks}/{total_checks} security checks passed"
    results["summary"] = summary

    print("\n" + "=" * 60)
    print(f"[*] Summary: {summary}")
    print(f"[*] {tracker.summary()}")
    print("=" * 60)

    # Output results
    if args.json:
        print("\n" + json.dumps(results, indent=2))
    elif args.html:
        generate_html_report(results)

    # Exit code
    if args.strict and not all(check.get("success", False) for check in results["checks"]):
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
