from pprint import pprint
from driver.usb_camera_driver import mvCamera_control

cam = mvCamera_control()
print('=== runtime diagnostics ===')
pprint(cam.get_runtime_diagnostics())
print('\n=== enumerate ===')
ret, names = cam.mvCamera_find(force_refresh=True)
print('find_ret:', ret)
print('names:')
for name in names if isinstance(names, list) else [names]:
    print('  ', name)
print('\n=== parsed devices ===')
pprint(cam.get_enumerated_devices())
