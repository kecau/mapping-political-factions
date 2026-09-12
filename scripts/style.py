"""Shared figure styling: fonts, palettes, date axes, political markers, saving.

Channel titles are Korean, so every figure that prints one needs a CJK-capable
font. Matplotlib has no Korean font of its own; `apply_style()` picks the first
one actually installed rather than hardcoding a single platform's font name.
"""

import os
import warnings
from datetime import datetime

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import findSystemFonts, FontProperties
from matplotlib.ticker import FuncFormatter

from . import config

# ── Colours ─────────────────────────────────────────────────────────────────
WING_COLOR = {'left': '#2A7AAE', 'right': '#D51B23',
              'mixed': '#7A5B9A', 'unknown': '#888888'}

# Deliberately lighter than WING_COLOR['unknown'], so a grace-period segment is
# never confused with an actual unknown-wing segment.
GRACE_COLOR = '#BBBBBB'

WING_ORDER = {w: i for i, w in enumerate(WING_COLOR)}   # left, right, mixed, unknown
WING_LETTER = {'left': 'L', 'right': 'R', 'mixed': 'M', 'unknown': 'U'}

# One distinct hue per dynamic community, in fixed slot order rather than cycled.
# With up to TOP_K_DCS lines on one axis, colouring by wing would make same-wing
# DCs indistinguishable; wing stays readable from the legend text instead.
# Validated 8-slot categorical palette: adjacent pairs pass colour-vision-
# deficiency and contrast checks for line charts.
DC_PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
              '#e87ba4', '#008300', '#4a3aa7', '#e34948']

# ── Korean political events overlaid on every time-axis figure ──────────────
# `(start, end, label)`; `end` is None for a single-day event, which is drawn as
# a dashed vertical line rather than a shaded span.
POLITICAL_EVENTS = [
    (datetime(2024, 4, 10), None,                 '22nd general election'),
    (datetime(2024, 12, 3), None,                 'Martial law crisis'),
    (datetime(2025, 1, 15), None,                 "Yoon's arrest"),
    (datetime(2025, 6, 3),  None,                 '21st presidential election'),
    (datetime(2026, 6, 3),  None,                 '9th local elections'),
    (datetime(2026, 7, 21), datetime(2026, 8, 17), 'DP leadership race'),
]
EVENT_COLOR = '#000000'

KOREAN_FONT_CANDIDATES = [
    'AppleGothic',        # macOS
    'Malgun Gothic',      # Windows
    'NanumGothic',        # common Linux install
    'NanumBarunGothic',
    'Noto Sans CJK KR',
    'Noto Sans KR',
    'Source Han Sans KR',
]

_resolved_font = None


def resolve_korean_font():
    """Return the name of the first installed Korean-capable font, or None."""
    global _resolved_font
    if _resolved_font is not None:
        return _resolved_font or None

    installed = set()
    for path in findSystemFonts():
        try:
            installed.add(FontProperties(fname=path).get_name())
        except Exception:
            continue

    for name in KOREAN_FONT_CANDIDATES:
        if name in installed:
            _resolved_font = name
            return name

    _resolved_font = ''
    return None


def apply_style():
    """Set the theme, the Korean font, and the minus-sign fix.

    Call once before plotting. seaborn's `set_theme` resets `font.family`, so
    the font is applied after the theme, not before -- getting that order wrong
    is why the original notebook had to re-assert the font in six places.
    """
    sns.set_theme(style='whitegrid', rc={'axes.unicode_minus': False})

    font = resolve_korean_font()
    if font:
        plt.rcParams['font.family'] = font
    else:
        warnings.warn(
            'No Korean font found. Channel titles and keyword labels will render '
            'as boxes. Install one of: ' + ', '.join(KOREAN_FONT_CANDIDATES),
            stacklevel=2,
        )
    plt.rcParams['axes.unicode_minus'] = False

    # Butt caps keep adjacent timeline segments from visually overlapping.
    mpl.rcParams['lines.solid_capstyle'] = 'butt'
    return font


# ── Date axis ───────────────────────────────────────────────────────────────

def month_year_formatter(x, pos=None):
    """Month ticks ('Jan'), with the year on a second line each January.

    Over a 2.5-year window there are ~30 monthly ticks; repeating the year on
    every one of them is noise.
    """
    d = mdates.num2date(x)
    label = d.strftime('%b')
    if d.month == 1:
        label += '\n' + d.strftime('%Y')
    return label


def format_month_axis(ax):
    """Apply the month/year tick convention to a date x-axis."""
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(month_year_formatter))
    for tick_label in ax.get_xticklabels():
        tick_label.set_rotation(0)
        tick_label.set_ha('center')


def add_political_events(ax, fontsize=10, events=None):
    """Overlay the Korean political-event markers on a date axis.

    Labels sit inside the plot, rotated flush against their line, so they read
    as attached to the event rather than floating above the figure.

    Events outside the axis's current data range are skipped, and both limits
    are restored afterwards -- otherwise a replication over a shorter window
    would silently stretch its x-axis out to the last event in the list.
    """
    events = POLITICAL_EVENTS if events is None else events

    x_left, x_right = ax.get_xlim()
    y_bottom, y_top = ax.get_ylim()
    label_y = y_top - 0.02 * (y_top - y_bottom)

    for start, end, label in events:
        x_start = mdates.date2num(start)
        if not (x_left <= x_start <= x_right):
            continue
        if end is None:
            ax.axvline(start, color=EVENT_COLOR, linestyle='--', linewidth=1,
                       alpha=0.5, zorder=0)
        else:
            ax.axvspan(start, min(end, mdates.num2date(x_right).replace(tzinfo=None)),
                       color=EVENT_COLOR, alpha=0.08, zorder=0)
        ax.text(start, label_y, label + '  ', rotation=90, va='top', ha='right',
                fontsize=fontsize, color=EVENT_COLOR, alpha=0.85)

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)


# ── Output ──────────────────────────────────────────────────────────────────

def save_figure(fig, name, directory=None, dpi=None, pdf=None, verbose=True):
    """Write `<FIGURE_DIR>/<name>.png` and, unless disabled, `.pdf`."""
    directory = directory or config.FIGURE_DIR
    dpi = dpi or config.DPI
    pdf = config.SAVE_PDF if pdf is None else pdf

    os.makedirs(directory, exist_ok=True)
    png_path = os.path.join(directory, f'{name}.png')
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
    paths = [png_path]
    if pdf:
        pdf_path = os.path.join(directory, f'{name}.pdf')
        fig.savefig(pdf_path, bbox_inches='tight')
        paths.append(pdf_path)

    if verbose:
        print(f'  saved {" / ".join(os.path.basename(p) for p in paths)} -> {directory}')
    return paths


def save_table(df, name, directory=None, verbose=True, **kwargs):
    """Write a results table to `<TABLE_DIR>/<name>.csv`."""
    directory = directory or config.TABLE_DIR
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f'{name}.csv')
    df.to_csv(path, index=False, **kwargs)
    if verbose:
        print(f'  saved {name}.csv -> {directory}')
    return path
