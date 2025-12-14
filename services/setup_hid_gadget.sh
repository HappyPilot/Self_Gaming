#!/usr/bin/env bash
# Configures the Jetson USB controller to expose keyboard + mouse HID gadgets.
# Run as root on the Jetson: sudo ./services/setup_hid_gadget.sh

set -euo pipefail

GADGET_DIR=/sys/kernel/config/usb_gadget/jetson_hid
UDC=$(ls /sys/class/udc | head -n1)

if ! mountpoint -q /sys/kernel/config; then
  mount -t configfs none /sys/kernel/config
fi

modprobe libcomposite

if [ -d "$GADGET_DIR" ]; then
  if [ -f "$GADGET_DIR/UDC" ]; then
    echo "" > "$GADGET_DIR/UDC" || true
  fi
  if [ -d "$GADGET_DIR/configs" ]; then
    find "$GADGET_DIR/configs" -type l -delete || true
  fi
  if [ -d "$GADGET_DIR/functions" ]; then
    find "$GADGET_DIR/functions" -mindepth 1 -maxdepth 1 -type d -exec rmdir {} \; || true
  fi
else
  mkdir -p "$GADGET_DIR"
fi

cd "$GADGET_DIR"

echo 0x1d6b > idVendor        # Linux Foundation
echo 0x0104 > idProduct       # Multifunction Composite Gadget
echo 0x0100 > bcdDevice
echo 0x0200 > bcdUSB

mkdir -p strings/0x409
echo "112233445566" > strings/0x409/serialnumber
echo "Jetson" > strings/0x409/manufacturer
echo "Jetson HID" > strings/0x409/product

mkdir -p configs/c.1/strings/0x409
echo "HID Config" > configs/c.1/strings/0x409/configuration
echo 120 > configs/c.1/MaxPower

# Keyboard HID function
mkdir -p functions
mkdir -p functions/hid.keyboard
echo 1 > functions/hid.keyboard/protocol
echo 1 > functions/hid.keyboard/subclass
echo 8 > functions/hid.keyboard/report_length
cat <<'EOF' > functions/hid.keyboard/report_desc
0x05,0x01,      # Usage Page (Generic Desktop)
0x09,0x06,      # Usage (Keyboard)
0xA1,0x01,      # Collection (Application)
0x05,0x07,      #   Usage Page (Key Codes)
0x19,0xE0,      #   Usage Minimum (224)
0x29,0xE7,      #   Usage Maximum (231)
0x15,0x00,      #   Logical Minimum (0)
0x25,0x01,      #   Logical Maximum (1)
0x75,0x01,      #   Report Size (1)
0x95,0x08,      #   Report Count (8)
0x81,0x02,      #   Input (Data, Variable, Absolute)
0x95,0x01,      #   Report Count (1)
0x75,0x08,      #   Report Size (8)
0x81,0x01,      #   Input (Constant) reserved byte
0x95,0x06,      #   Report Count (6)
0x75,0x08,      #   Report Size (8)
0x15,0x00,      #   Logical Minimum (0)
0x25,0x73,      #   Logical Maximum (115)
0x05,0x07,      #   Usage Page (Key codes)
0x19,0x00,      #   Usage Minimum (0)
0x29,0x73,      #   Usage Maximum (115)
0x81,0x00,      #   Input (Data, Array)
0xC0           # End Collection
EOF

# Mouse HID function
mkdir -p functions/hid.mouse
echo 0 > functions/hid.mouse/protocol
echo 1 > functions/hid.mouse/subclass
echo 4 > functions/hid.mouse/report_length
cat <<'EOF' > functions/hid.mouse/report_desc
0x05,0x01,      # Usage Page (Generic Desktop)
0x09,0x02,      # Usage (Mouse)
0xA1,0x01,      # Collection (Application)
0x09,0x01,      #   Usage (Pointer)
0xA1,0x00,      #   Collection (Physical)
0x05,0x09,      #     Usage Page (Buttons)
0x19,0x01,      #     Usage Minimum (1)
0x29,0x03,      #     Usage Maximum (3)
0x15,0x00,      #     Logical Minimum (0)
0x25,0x01,      #     Logical Maximum (1)
0x95,0x03,      #     Report Count (3)
0x75,0x01,      #     Report Size (1)
0x81,0x02,      #     Input (Data, Variable, Absolute)
0x95,0x01,      #     Report Count (1)
0x75,0x05,      #     Report Size (5)
0x81,0x01,      #     Input (Constant)
0x05,0x01,      #     Usage Page (Generic Desktop)
0x09,0x30,      #     Usage (X)
0x09,0x31,      #     Usage (Y)
0x15,0x81,      #     Logical Minimum (-127)
0x25,0x7F,      #     Logical Maximum (127)
0x75,0x08,      #     Report Size (8)
0x95,0x02,      #     Report Count (2)
0x81,0x06,      #     Input (Data, Variable, Relative)
0xC0,
0xC0
EOF

ln -s functions/hid.keyboard configs/c.1/
ln -s functions/hid.mouse configs/c.1/

echo "$UDC" > UDC

chmod 666 /dev/hidg0 /dev/hidg1
echo "HID gadget ready (keyboard=/dev/hidg0 mouse=/dev/hidg1)."
