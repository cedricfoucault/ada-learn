import data_analysis_utils as dana
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Sizes

# Width in inches available for content (figure, text)
# in an A4, Nature-style paper
A4_PAPER_CONTENT_WIDTH = 7.1

DEFAULT_HEIGHT = 2.16 # default figure height, in inches

TWIN_AXIS_PAD = 30 / 300

CM_PER_INCH = 2.54

# Colors

# Refactoring UI Color Palette #8
COLORS_PRIMARY_BLUE = [ # Different shades of Primary (blue)
    "#002159", "#01337D", "#03449E", "#0552B5", "#0967D2",
    "#2186EB", "#47A3F3", "#7CC4FA", "#BAE3FF", "#E6F6FF"]
COLORS_SUPPORTING_CYAN = [ # Different shades of Supporting color (cyan)
    "#05606E","#07818F", "#099AA4", "#0FB5BA", "#1CD4D4",
    "#3AE7E1", "#62F4EB", "#92FDF2", "#C1FEF6", "#E1FCF8"]
COLORS_SUPPORTING_ORANGE = [ # Different shades of Supporting color (orange)
    "#841003","#AD1D07", "#C52707", "#DE3A11", "#F35627",
    "#F9703E", "#FF9466", "#FFB088", "#FFD0B5", "#FFE8D9"]
COLORS_NEUTRAL_GRAY = [ # Different shades of Neutral color (grey)
    "#1F2933", "#323F4B", "#3E4C59", "#52606D", "#616E7C",
    "#7B8794", "#9AA5B1", "#CBD2D9", "#E4E7EB", "#F5F7FA"]

SUBJECT_COLOR = COLORS_PRIMARY_BLUE[4]
IDEAL_LEARNER_COLOR = "#1CD4D4" # Supporting (cyan)
GENERATIVE_COLOR = "#616E7C" # Neutral (grey)
ADA_POS_OUTCOME_COLOR = "#9AA5B1" # Neutral (light grey)
ADA_PROB_OUTCOME_1_COLOR = "#3355FF"
ADA_PROB_OUTCOME_0_COLOR = "#FFDD33"
GRAY_COLOR = "#666666"
BLACK_COLOR = "#000000"

ERROR_SHADING_ALPHA = 0.2

NOISY_DR_COLORS = {
    "constant": "#7C5E10",
    "prop-error": "#C65D21"
}

# Text

SUBJECT_LABEL = "Subject"
IDEAL_LEARNER_LABEL = "Normative learner"
GENERATIVE_LABEL = "Generative"
DIST_TO_CP_LABEL = "Trials after change point"
OUTCOMES_AFTER_CP_LABEL = "Observations after change point"
SEC_LABEL = "Time (s)"
SEC_AFTER_CP_LABEL = "Time after change point (s)"
LABEL_FOR_CONGRUENT = {True: "congruent outcome", False: "incongruent outcome"}
CPP_LABEL = "Change-point probability"
REG_COEF_LABEL = "Regression weight"
ONLY_TRIALS_WITH_UPDATE_LABEL = "only lr≠0" # Analysis excluding data where the subject made no overt update (learning rate = 0)

NOISY_DR_TYPE_LABEL = {
    "constant": "constant",
    "prop-error": "prop. to error magnitude"
}

def setup_mpl_style(fontsize=8):
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Arial'
    mpl.rcParams['mathtext.default'] = 'regular'
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['font.size'] = fontsize
    mpl.rcParams['figure.titlesize'] = 10
    mpl.rcParams['axes.titlesize'] = fontsize
    mpl.rcParams['axes.labelsize'] = fontsize
    mpl.rcParams['xtick.labelsize'] = fontsize
    mpl.rcParams['ytick.labelsize'] = fontsize
    mpl.rcParams['legend.fontsize'] = fontsize
    mpl.rcParams['axes.labelpad'] = 4.0
    mpl.rcParams['lines.linewidth'] = 1.0
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams['figure.constrained_layout.use'] = True

def setup_xy_axes(ax, trials):
    ax.set_xlim(trials[0] - 1, trials[-1] + 1)
    ax.set_ylim(-0.01, 1.01)

def plot_generatives(ax, trials, generatives, **kwargs):
    ax.plot(trials, generatives, '-', label='Generative', **kwargs)

def plot_estimates(ax, trials, estimates, offset=0.5, **kwargs):
    trials = trials + offset # offset to reflect that estimates are recorded
                          # at the end of the current trial, after the outcome
                          # has been observed
    ax.plot(trials, estimates, '-', **kwargs)

def save_figure(fig, figpath, verbose=True,
    do_fit_axis_labels=False):
    if do_fit_axis_labels:
        fit_axis_labels_fontsize(fig)
    fig.savefig(figpath)
    plt.close(fig)
    if verbose:
        print(f"Figure saved at {figpath}")

def save_stats(stat_data_pd, fpath, verbose=True):
    stat_data_pd.to_csv(fpath)
    if verbose:
        print(f"Stats saved at {fpath}")

def stat_label(p, criteria=(0.05, 0.01, 0.001)):
    if p <= criteria[2]:
        return "***"
    elif p <= criteria[1]:
        return "**"
    elif p <= criteria[0]:
        return "*"
    else:
        return "ns"

def get_uncertainty_label(with_unit=True, unit=dana.UNCERTAINTY_UNIT):
    name = "Prior uncertainty"
    label = f"{name} ({unit})" if with_unit else name
    return label

def get_reg_label(reg_key,
    is_zscored=True,
    is_small_width=False,
    with_unit=False):
    if reg_key == "cpp":
        if is_small_width:
            label = "Change-point\nprobability"
        else:
            label = "Change-point probability"
    elif reg_key == "uncertainty":
        if is_small_width:
            label = "Prior\nuncertainty"
        else:
            label = "Prior uncertainty"
        if with_unit:
            label += f" ({dana.UNCERTAINTY_UNIT})"
    elif reg_key == "relative-uncertainty-*-1-cpp":
        label = "Relative uncertainty\n*(1-cpp)"
    if is_zscored:
        label += "\n(z-scored)"
    return label

def get_label_for_conf_type(conf_type):
    if conf_type == 'prev-conf':
        return f"Model prev. confidence"
    elif conf_type == 'prev-conf-pred':
        return f"Model prev. confidence pred."
    elif conf_type == 'new-conf':
        return f"Model new confidence "
    elif conf_type == 'new-conf-pred':
        return f"Model new confidence pred."
    elif conf_type == 'prev-uncertainty-pred':
        return f"Model prev. uncertainty pred."
    elif conf_type == 'relative-prev-conf-pred':
        return f"Model prev. relative confidence pred."
    elif conf_type == 'relative-prev-uncertainty-pred':
        return f"Model prev. relative uncertainty pred."
    
def fit_axis_labels_fontsize(fig, min_fontsize=6,
    DEFAULT_TEXT_DPI=72, margin_extra=0.025):
    backend = mpl.backend_bases.RendererBase()
    ax = fig.gca()
    figsize = fig.get_size_inches()
    for xory in ["x", "y"]:
        i = (0 if xory == "x" else 1)
        margin = ax.margins()[i] + margin_extra
        available_size = figsize[i] * (1 - 2 * margin)
        axis = (ax.xaxis if xory == "x" else ax.yaxis)
        label = axis.get_label()
        text = label.get_text()

        fontprop = label.get_fontproperties()
        text_width, _, _ = backend.get_text_width_height_descent(
            text, fontprop, False)
        text_width /= DEFAULT_TEXT_DPI # convert to inches

        fontsize = fontprop.get_size()
        should_reset = False
        while ((text_width > available_size)
            and fontsize >= min_fontsize):
            should_reset = True
            fontsize -= 1
            fontprop.set_size(fontsize)
            text_width, _, _ = backend.get_text_width_height_descent(
            text, fontprop, False)
            text_width /= DEFAULT_TEXT_DPI # convert to inches
        if should_reset:
            if xory == "x":
                ax.set_xlabel(text, fontproperties=fontprop)
            else:
                ax.set_ylabel(text, fontproperties=fontprop)

