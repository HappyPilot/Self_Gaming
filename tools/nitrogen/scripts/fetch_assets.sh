#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"
VENDOR_DIR="${ROOT_DIR}/_vendor"
NITROGEN_DIR="${VENDOR_DIR}/NitroGen"

REPO_URL="${NITROGEN_REPO_URL:-https://github.com/MineDojo/NitroGen.git}"
REPO_REF="${NITROGEN_REPO_REF:-main}"
USE_SUBMODULE="${NITROGEN_USE_SUBMODULE:-0}"

HF_REPO="${NITROGEN_HF_REPO:-nvidia/NitroGen}"
HF_FILENAME="${NITROGEN_HF_FILENAME:-ng.pt}"
HF_URL="${NITROGEN_HF_URL:-https://huggingface.co/${HF_REPO}/resolve/main/${HF_FILENAME}}"
OUT_PATH="${VENDOR_DIR}/${HF_FILENAME}"
TMP_PATH="${OUT_PATH}.download"

EXPECTED_SIZE="${NITROGEN_NG_PT_SIZE:-}"
EXPECTED_SHA256="${NITROGEN_NG_PT_SHA256:-}"
FORCE_DOWNLOAD="${NITROGEN_FORCE_DOWNLOAD:-0}"

HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_TOKEN:-}}"

die() {
  echo "Error: $*" >&2
  exit 1
}

file_size() {
  if stat -c %s "$1" >/dev/null 2>&1; then
    stat -c %s "$1"
  else
    stat -f %z "$1"
  fi
}

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    die "sha256sum/shasum not found"
  fi
}

fetch_head_size() {
  if [ -n "${HF_TOKEN}" ]; then
    curl -sIL -H "Authorization: Bearer ${HF_TOKEN}" "${HF_URL}"
  else
    curl -sIL "${HF_URL}"
  fi | awk 'tolower($1)=="content-length:" {print $2}' | tail -n 1 | tr -d '\r'
}

download_ngpt() {
  if [ -n "${HF_TOKEN}" ]; then
    curl -L --fail --retry 3 -H "Authorization: Bearer ${HF_TOKEN}" -o "${TMP_PATH}" "${HF_URL}"
  else
    curl -L --fail --retry 3 -o "${TMP_PATH}" "${HF_URL}"
  fi
  mv "${TMP_PATH}" "${OUT_PATH}"
}

mkdir -p "${VENDOR_DIR}"

if [ "${USE_SUBMODULE}" = "1" ]; then
  git -C "${REPO_ROOT}" submodule update --init --recursive -- "${NITROGEN_DIR}" \
    || die "submodule update failed"
else
  if [ -d "${NITROGEN_DIR}" ] && [ ! -d "${NITROGEN_DIR}/.git" ]; then
    die "${NITROGEN_DIR} exists without git metadata; remove it or set NITROGEN_USE_SUBMODULE=1"
  fi
  if [ -d "${NITROGEN_DIR}/.git" ]; then
    git -C "${NITROGEN_DIR}" fetch --all --tags
    git -C "${NITROGEN_DIR}" checkout "${REPO_REF}"
    git -C "${NITROGEN_DIR}" pull --ff-only || true
  else
    git clone "${REPO_URL}" "${NITROGEN_DIR}"
    git -C "${NITROGEN_DIR}" checkout "${REPO_REF}"
  fi
fi

if [ -f "${OUT_PATH}" ] && [ "${FORCE_DOWNLOAD}" != "1" ]; then
  echo "Found ${OUT_PATH}; verifying."
else
  echo "Downloading ${HF_URL} -> ${OUT_PATH}"
  download_ngpt
fi

LOCAL_SIZE="$(file_size "${OUT_PATH}")"
HEAD_SIZE="${EXPECTED_SIZE}"
if [ -z "${HEAD_SIZE}" ]; then
  HEAD_SIZE="$(fetch_head_size || true)"
fi

if [ -n "${HEAD_SIZE}" ]; then
  if [ "${LOCAL_SIZE}" != "${HEAD_SIZE}" ]; then
    die "ng.pt size mismatch (local=${LOCAL_SIZE}, expected=${HEAD_SIZE})"
  fi
else
  echo "Warning: expected size unavailable; local size=${LOCAL_SIZE} bytes"
fi

LOCAL_SHA="$(file_sha256 "${OUT_PATH}")"
if [ -n "${EXPECTED_SHA256}" ]; then
  if [ "${LOCAL_SHA}" != "${EXPECTED_SHA256}" ]; then
    die "ng.pt checksum mismatch"
  fi
else
  echo "ng.pt sha256=${LOCAL_SHA} (set NITROGEN_NG_PT_SHA256 to enforce)"
fi

echo "NitroGen assets ready in ${VENDOR_DIR}"
