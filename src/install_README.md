# udev rules

```
git clone https://github.com/IntelRealSense/librealsense

cd librealsense

sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/

sudo udevadm control --reload-rules
sudo udevadm trigger
```
