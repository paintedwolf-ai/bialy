"""Draw the reviewer in a terminal and feed it key names."""
import base64
import curses
import os
import shutil
import subprocess

from .model import Reviewer

KEYS = {curses.KEY_UP: 'up', curses.KEY_DOWN: 'down', curses.KEY_LEFT: 'left', curses.KEY_RIGHT: 'right',
        curses.KEY_NPAGE: 'pgdn', curses.KEY_PPAGE: 'pgup', curses.KEY_HOME: 'home', curses.KEY_END: 'end',
        curses.KEY_ENTER: 'enter', 10: 'enter', 13: 'enter', 27: 'esc', 9: 'tab',
        curses.KEY_BACKSPACE: 'backspace', 127: 'backspace', 8: 'backspace', 3: 'ctrl+c'}
CLIPBOARDS = (['pbcopy'], ['wl-copy'], ['xclip', '-selection', 'clipboard'], ['xsel', '--clipboard', '--input'])
REFRESH_MS = 5000


def key_name(key):
    """A curses key, a typed character, or None after the refresh timeout, as a key name."""
    if key is None:
        return 'tick'
    if isinstance(key, str):
        return KEYS.get(ord(key), key) if len(key) == 1 else key
    return KEYS.get(key, '')


def read_key(screen):
    try:
        return key_name(screen.get_wch())
    except curses.error:
        return key_name(None)


def copy(text, command):
    """Send text to a command, a local clipboard tool, or the terminal's clipboard over SSH."""
    if command:
        result = subprocess.run([os.environ.get('SHELL', '/bin/sh'), '-c', command], input=text, text=True)
        return 'Sent to: ' + command if result.returncode == 0 else f'The command exited {result.returncode}.'
    if not os.environ.get('SSH_CONNECTION'):
        for tool in CLIPBOARDS:
            if shutil.which(tool[0]):
                result = subprocess.run(tool, input=text, text=True)
                return 'Copied to the clipboard.' if result.returncode == 0 else f'{tool[0]} exited {result.returncode}.'
    # Terminals that support OSC 52 copy this into the clipboard of the machine you typed on.
    with open('/dev/tty', 'w') as terminal:
        terminal.write('\033]52;c;' + base64.b64encode(text.encode()).decode() + '\a')
    return 'Copied through the terminal. If nothing was copied, type a command such as: cat > item.txt'


def styles():
    curses.use_default_colors()
    pairs = {'pass': curses.COLOR_GREEN, 'fail': curses.COLOR_RED, 'status': curses.COLOR_YELLOW}
    for index, color in enumerate(pairs.values(), 1):
        curses.init_pair(index, color, -1)
    color = {name: curses.color_pair(index) for index, name in enumerate(pairs, 1)}
    return {'title': curses.A_BOLD, 'bold': curses.A_BOLD, 'hint': curses.A_DIM, 'footer': curses.A_DIM,
            'body': curses.A_NORMAL, 'cursor': curses.A_REVERSE, 'pass': color['pass'], 'fail': color['fail'],
            'status': color['status'] | curses.A_BOLD}


def draw(screen, reviewer, style):
    height, width = screen.getmaxyx()
    screen.erase()
    for row, (text, name) in enumerate(reviewer.render(height, width)[:height]):
        screen.addnstr(row, 0, text, max(width - 1, 0), style[name])
    screen.refresh()


def loop(screen, reviewer):
    curses.curs_set(0)
    screen.keypad(True)
    screen.timeout(REFRESH_MS)
    style = styles()
    while not reviewer.done:
        draw(screen, reviewer, style)
        reviewer.handle(read_key(screen))


def run(review, campaign):
    """Open the reviewer; returns when the person quits."""
    os.environ.setdefault('ESCDELAY', '25')
    reviewer = Reviewer(review, campaign, copy)
    try:
        curses.wrapper(loop, reviewer)
    except KeyboardInterrupt:
        pass
