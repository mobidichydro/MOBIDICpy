# Security Policy

## Supported versions

Security fixes are released as new versions of MOBIDICpy; please upgrade to the latest release.

| Version  | Supported |
| -------- | --------- |
| >= 1.1.0 | Yes       |
| < 1.1.0  | No        |

## Reporting a vulnerability

Please **do not** report security vulnerabilities through public GitHub issues.

Report them privately through GitHub's private vulnerability reporting:
<https://github.com/mobidichydro/MOBIDICpy/security/advisories/new>

The report is visible only to the maintainers and is transmitted over HTTPS.
If you cannot use GitHub, contact the maintainer by email at matteo.masi@unifi.it
with the subject line `[MOBIDICpy security]`.

Please include:

- the affected version or commit SHA;
- a description of the issue and its potential impact;
- steps to reproduce, ideally a minimal configuration or input file.

## Response process

- We acknowledge reports within 14 days (typically within a few working days).
- We confirm the issue, assess its severity using CVSS (Common Vulnerability Scoring System), and agree on a disclosure date with the reporter.
- Fixes for confirmed vulnerabilities of medium or higher severity are released within 60 days of the report.
- Fixed vulnerabilities are listed in the release notes and published as GitHub Security Advisories,
  with a CVE (Common Vulnerabilities and Exposures) identifier where applicable. Reporters are credited unless they prefer otherwise.

## Scope

MOBIDICpy is a scientific simulation package that reads user-supplied configuration (YAML) and
geospatial/meteorological input files. Reports concerning unsafe handling of such inputs
(e.g. code execution through crafted files, path traversal) are in scope. Vulnerabilities in
third-party dependencies should be reported to their upstream projects.
