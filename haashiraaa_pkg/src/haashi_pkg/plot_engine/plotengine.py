from __future__ import annotations

import logging
from datetime import datetime
from typing import (
    Tuple,
    Literal,
    Dict,
    Union,
    Optional,
    Iterable
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.ticker import StrMethodFormatter
from pandas import Series, Timestamp

from haashi_pkg.utility.utils import Utility

# ----------------------------
# TYPE ALIASES
# ----------------------------

Numeric = Union[int, float]

XData = Union[
    Iterable[int],
    Iterable[float],
    Iterable[str],
    Iterable[datetime],
    Iterable[Timestamp],  # forward ref
    Series,
    np.ndarray,
]

YData = Union[
    Iterable[Numeric],
    np.ndarray,
    Series,
]

PlotType = Literal["line", "bar", "scatter", "pie"]

Kwargs = Dict[str, Union[str, int, float, tuple[int]]]


# ----------------------------
# ENGINE
# ----------------------------

class PlotEngine:
    def __init__(self) -> None:

        self.ut: Utility = Utility(level=logging.INFO)

        self.colors_01: list[str] = [
            "#4E79A7",  # blue
            "#F28E2B",  # orange
            "#59A14F",  # green
            "#E15759",  # red
            "#B07AA1",  # purple
        ]

        self.colors_02: list[str] = [
            "#1f77b4",  # deep blue
            "#ff7f0e",  # warm orange
            "#2ca02c",  # green
            "#d62728",  # muted red
            "#9467bd",  # violet
        ]

        self.colors_03: list[str] = [
            "#5DA5DA",  # soft blue
            "#F17CB0",  # pink
            "#60BD68",  # green
            "#B2912F",  # gold
            "#B276B2",  # purple
        ]

    # ----------------------------
    # SETUP
    # ----------------------------

    def setup(
        self,
        figsize: Tuple[int, int] = (10, 6),
    ) -> Tuple[Figure, Axes]:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax

    # ----------------------------
    # DRAW
    # ----------------------------

    def try_draw(self, parse_plot_type: str) -> str:
        parse_plot_type = parse_plot_type.lower()
        if parse_plot_type in ["line", "bar", "scatter", "pie"]:
            return parse_plot_type
        else:
            raise ValueError(f"Unsupported plot type: {parse_plot_type}")

    def draw(
        self,
        ax: Axes,
        x: XData,
        y: YData,
        plot_type: PlotType = "line",
        **kwargs: Kwargs,
    ) -> None:
        """
        Draws a plot on the provided Axes.

        **kwargs are forwarded to the underlying matplotlib call.
        Common examples:
        - color
        - marker
        - markersize
        - linewidth
        - alpha
        """

        plot_type = self.try_draw(plot_type)  # type: ignore[assignment]

        if plot_type == "line":
            ax.plot(x, y, **kwargs)

        elif plot_type == "bar":
            ax.bar(x, y, **kwargs)

        elif plot_type == "scatter":
            ax.scatter(x, y, **kwargs)

        elif plot_type == "pie":
            ax.pie(y, **kwargs)

        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

    # ----------------------------
    # DECORATE
    # ----------------------------
    def set_labels_and_title(
        self,
        ax: Axes,
        *,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title_fontsize: int = 18,
        label_fontsize: int = 14,
        title_weight: str = "semibold",
        label_weight: str = "semibold",
    ) -> None:
        if title is not None:
            ax.set_title(
                title,
                fontsize=title_fontsize,
                fontweight=title_weight,
            )

        if xlabel is not None:
            ax.set_xlabel(
                xlabel,
                fontsize=label_fontsize,
                fontweight=label_weight,
            )

        if ylabel is not None:
            ax.set_ylabel(
                ylabel,
                fontsize=label_fontsize,
                fontweight=label_weight,
            )

    def format_y_axis(
        self,
        ax: Axes,
        currency: str = "$",
        style: str = "comma",
        decimals: int = 0
    ):
        if style == "comma":
            ax.yaxis.set_major_formatter(
                StrMethodFormatter(f"{currency}{{x:,.{decimals}f}}")
            )

    def force_xticks(
        self,
        ax: Axes,
        labels: np.ndarray,
        positions: Optional[np.ndarray] = None,
        rotation: int = 45
    ) -> None:
        if positions is None:
            positions = range(len(labels))

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=rotation, ha="right")

    def set_legend(
        self,
        ax: Axes,
        *,
        loc: str = "upper right",
        fontsize: int = 14,
    ) -> None:
        ax.legend(loc=loc, fontsize=fontsize)

    # ----------------------------
    # FINALIZE
    # ----------------------------
    def save_or_show(
        self,
        fig: Figure,
        *,
        dpi: int = 300,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi)
            self.ut.info(f"Saved figure to {save_path}")

        if show:
            plt.show()

        plt.close(fig)

