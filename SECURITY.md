# Security Policy

## Credentials

Never publish credentials, personal access tokens, private keys, passwords, or sensitive user data in source code, notebooks, issues, or pull requests.

If a credential is exposed:

1. revoke or rotate it immediately;
2. remove it from the working file;
3. inspect repository history and outputs for copies;
4. replace it with a secret-management mechanism.

## Notebook hygiene

Colab notebooks are sensitive because cell output can preserve authenticated shell commands.

Before committing a notebook:

- inspect cell source;
- inspect cell outputs;
- remove credentials;
- remove private paths or data where needed;
- verify the resulting notebook still runs as a research record.

## September 2026 training audit

The training notebook supplied for this repository audit contained a GitHub personal access credential in a cell. The repository publication copy is sanitized.

The exposed credential should be revoked even after sanitization because removing it from the current file does not invalidate a credential that was already visible.

## Reporting

For repository security problems, report privately to the maintainer rather than publishing the secret or exploit details in an issue.
