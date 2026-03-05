# Steve Jobserver in Docker Builds

This project uses Gentoo `dev-build/steve` during Docker builds to avoid
overscheduling and reduce total rebuild time.

## What is enabled

- Global Portage/build flags in `config/make.conf`:
  - `MAKEFLAGS="-l32 --jobserver-auth=fifo:/dev/steve"`
  - `NINJAOPTS="-l32"`
  - `FEATURES="${FEATURES} jobserver-token"`
- Docker build uses BuildKit with steve device access on heavy build steps.
- Built system includes:
  - `acct-group/cuse`
  - `acct-group/jobserver`
  - `acct-user/steve`
  - `dev-build/steve`
  - udev rules for `/dev/cuse` and `/dev/steve`
  - `steve.service` enabled

## How build access works

`setup01_build_image.sh` creates/uses a dedicated buildx builder with:

- `--allow-insecure-entitlement security.insecure`
- `--allow-insecure-entitlement device`

Build command uses:

- `--allow security.insecure`
- `--build-context hostdev=/dev`

Dockerfile `RUN` steps that must use steve are marked with:

- `--security=insecure`
- `--mount=from=hostdev,source=steve,target=/dev/steve`

Without these, `/dev/steve` is not usable inside build steps.

## Usage

Use the normal entrypoint:

```bash
./setup01_build_image.sh
```

or full flow:

```bash
./build_and_run.sh
```

## Troubleshooting

- `failed to solve: granting entitlement security.insecure is not allowed`:
  build is not using the configured buildx builder.
- `/dev/steve` permission errors during build:
  host steve/cuse permissions are wrong.
- Build falls back to normal parallelism:
  check `config/make.conf` still has steve `MAKEFLAGS`/`NINJAOPTS`.
