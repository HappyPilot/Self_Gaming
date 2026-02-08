# Jetson NvMapMemAlloc Error 12 - Audit Playbook (R36.4.x)

Goal: gather evidence and map the system state to the NVIDIA workaround path.
This playbook is read-only. It does not modify the system.

## What this playbook covers
- Confirm JetPack / L4T release (R36.4.x).
- Collect disk, swap, kernel, and NVIDIA package details.
- Capture recent kernel log lines for nvmap/nvgpu.
- Decide whether the NVIDIA workaround/patch applies.

## Step 0 - Run the audit script

Run on Jetson:

```bash
bash /home/dima/self-gaming/scripts/jetson_nvmap_audit.sh | tee /tmp/nvmap_audit.txt
```

This is safe. It only prints system state.

## Step 1 - Check release and kernel

Look for:
- `/etc/nv_tegra_release` like: `R36 (release), REVISION: 4.x`
- kernel like: `5.15.148-tegra`

If R36.4.x is present, continue. The NVIDIA workaround/patch applies to R36.4.x.

## Step 2 - Compare with NVIDIA official workaround/patch

The NVIDIA forum provides the official steps and patch files for R36.4.x.
We do not apply them in this audit step; we only confirm applicability.

Reference (official thread):
- https://forums.developer.nvidia.com/t/unable-to-allocate-cuda0-buffer-after-updating-ubuntu-packages/347862/206

If we proceed to apply:
- It is a kernel + OOT modules patch.
- NVIDIA notes it may require reflash.
- It involves building kernel and OOT modules from L4T sources.

## Step 3 - Disk and swap readiness (for later)

NVIDIA notes:
- Disk free >= 30 GB
- Swap total >= 8 GB (16 GB recommended)

This audit only records whether the system is ready.

## Step 4 - NVIDIA package inventory

We capture installed versions of:
- nvidia-l4t-*
- nvidia-cuda-*
- nvidia-tensorrt-*
- nvidia-cudnn-*
- nvidia-vpi-*
- nvidia-container-*

This helps tie failures to specific component updates.

## Step 5 - Kernel log hints

Look for lines matching:
- nvmap
- nvgpu
- nvhost

If you see recent "NvMapMemAlloc" or "NvMapMemHandleAlloc" errors, record them.

## Outcome

If the audit shows:
- R36.4.7 (or R36.4.x) AND
- NvMapMemAlloc errors appear in dmesg,

then the NVIDIA workaround/patch is applicable and likely the correct next step.

## Optional next step (NOT done in audit)

If you want to proceed, we can build a safe "apply" playbook:
- Download public_sources for the exact R36.4.x release.
- Apply NVIDIA patches in order.
- Build and install kernel and OOT modules.
- Update initramfs and reboot.
- Hold kernel packages until official fix.

We will not run those steps without explicit approval.
