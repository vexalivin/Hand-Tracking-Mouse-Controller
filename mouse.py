import ctypes


MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
INPUT_MOUSE = 0


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", _MOUSEINPUT),
    ]


class _INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("data", _INPUT_UNION),
    ]


def _send_mouse(dwFlags):
    inp = _INPUT()
    inp.type = INPUT_MOUSE
    inp.data.mi.dwFlags = dwFlags
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def _virtual_screen_size():
    vw = ctypes.windll.user32.GetSystemMetrics(78)
    vh = ctypes.windll.user32.GetSystemMetrics(79)
    return vw, vh


def _send_mouse_abs(x, y, flags):
    vx = ctypes.windll.user32.GetSystemMetrics(76)
    vy = ctypes.windll.user32.GetSystemMetrics(77)
    vw, vh = _virtual_screen_size()
    inp = _INPUT()
    inp.type = INPUT_MOUSE
    inp.data.mi.dx = int((x - vx) * 65535 / max(1, vw - 1))
    inp.data.mi.dy = int((y - vy) * 65535 / max(1, vh - 1))
    inp.data.mi.dwFlags = flags | MOUSEEVENTF_ABSOLUTE
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def send_left_click():
    _send_mouse(MOUSEEVENTF_LEFTDOWN)
    _send_mouse(MOUSEEVENTF_LEFTUP)


def send_right_click():
    _send_mouse(MOUSEEVENTF_RIGHTDOWN)
    _send_mouse(MOUSEEVENTF_RIGHTUP)


def send_left_down():
    _send_mouse(MOUSEEVENTF_LEFTDOWN)


def send_left_up():
    _send_mouse(MOUSEEVENTF_LEFTUP)


def send_right_down():
    _send_mouse(MOUSEEVENTF_RIGHTDOWN)


def send_right_up():
    _send_mouse(MOUSEEVENTF_RIGHTUP)


def send_middle_click():
    _send_mouse(MOUSEEVENTF_MIDDLEDOWN)
    _send_mouse(MOUSEEVENTF_MIDDLEUP)


def move_mouse_to(x, y):
    _send_mouse_abs(x, y, MOUSEEVENTF_MOVE)
