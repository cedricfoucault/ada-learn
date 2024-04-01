//
// Define shared constants (known prior to execution)
//

// Colors

const WHITE_COLOR = "#FFFFFF"
const BACKGROUND_COLOR = "#F3F1F3";
const LIGHT_GRAY_COLOR = "#D3D3D3";
const GRAY_COLOR = "#B3B3B3"
const DARK_GRAY_COLOR = "#808080";
const BLACK_COLOR = "#000000"
const BLACK_HIGHLIGHTED_COLOR = "#666666";
const MEDIUM_GREEN_COLOR = "#3EBD93";

const BLUE_DOT_COLOR = "#3355FF"
const YELLOW_DOT_COLOR = "#FFDD33";

const TEXT_COLOR = BLACK_COLOR;

const BUTTON_TEXT_COLOR = "#0066CC";
const BUTTON_TEXT_COLOR_HIGHLIGHTED = "#75B3F0";
const BUTTON_TEXT_COLOR_DISABLED =  "#8E8E93"; // gray2: "#aeaeb2";

const SLIDER_THUMB_FILL_COLOR = WHITE_COLOR;
const SLIDER_THUMB_STROKE_COLOR = BLACK_COLOR;
const SLIDER_TRACK_COLOR = GRAY_COLOR;
const SLIDER_TICKS_LABEL_COLOR = DARK_GRAY_COLOR;

const POS_OUTCOME_DOT_FILL_COLOR = WHITE_COLOR;
const POS_OUTCOME_DOT_STROKE_COLOR = BLACK_COLOR;

const PROMPT_COLOR = BLACK_COLOR;

const SESSION_END_TITLE_COLOR = DARK_GRAY_COLOR;

const SCORE_METER_STROKE_COLOR = DARK_GRAY_COLOR;
const SCORE_METER_FILL_COLOR = MEDIUM_GREEN_COLOR;
const SCORE_DISPLAY_LABEL_COLOR = DARK_GRAY_COLOR;

const INDICATOR_MARKER_COLOR = BLACK_COLOR;
const TRUE_HIDVAR_LABEL_COLOR = DARK_GRAY_COLOR;
const TRUE_HIDVAR_ESTIMATE_MARKER_COLOR = "rgba(0,0,0,0.2)";

const TRUE_HIDVAR_DOT_FILL_COLOR = LIGHT_GRAY_COLOR;
const TRUE_HIDVAR_DOT_STROKE_COLOR = GRAY_COLOR;

// Font sizes (in CSS pixels)

const FONT = "Arial";
const TEXT_FONT_SIZE = 14;
const SLIDER_TICKS_LABEL_FONT_SIZE = 12;
const BUTTON_FONT_SIZE = 14;
const REVIEW_INSTRUCTIONS_BUTTON_FONT_SIZE = 13;
const PROMPT_FONT_SIZE = 16;
const SESSION_END_TITLE_FONT_SIZE = 20;
const SESSION_END_TITLE_BOLD = true;
const SCORE_LABEL_FONT_SIZE = 14;
const TRUE_HIDVARS_TITLE_FONT_SIZE = 14;
const TRUE_HIDVAR_RANK_FONT_SIZE = 12;
const TEXT_LINE_HEIGHT_MULTIPLIER = 1.4;
const MULTILINE_TEXT_LINE_HEIGHT = Math.ceil(TEXT_FONT_SIZE * TEXT_LINE_HEIGHT_MULTIPLIER);

// Sizes of UI display elements (in CSS pixels)

const PROB_OUTCOME_DOT_RADIUS = 18;
const POS_OUTCOME_DOT_RADIUS = 7;
const POS_OUTCOME_DOT_STROKE_THICKNESS = 1;

const SLIDER_THUMB_WIDTH = 12;
const SLIDER_THUMB_HEIGHT = 36;
const SLIDER_THUMB_CORNER_RADIUS = 3;
const SLIDER_THUMB_STROKE_THICKNESS = 2;
const SLIDER_TRACK_THICKNESS = 3;
const SLIDER_TRACK_REACHABLE_FRACTION_RANGE_PROB = [0.1, 0.9];
const SLIDER_TRACK_REACHABLE_FRACTION_RANGE_POS = [0., 1.0];
const SLIDER_TRACK_UNREACHABLE_SEGMENTS_HATCH_LENGTH = 9;
const SLIDER_TRACK_UNREACHABLE_SEGMENTS_HATCH_ANGLE = 60;
const SLIDER_TRACK_UNREACHABLE_SEGMENTS_HATCH_THICKNESS = 3;
const SLIDER_TRACK_UNREACHABLE_SEGMENTS_DASH_LENGTH = 3;
const SLIDER_TRACK_UNREACHABLE_SEGMENTS_DASH_SPACE = 3;
const SLIDER_TRACK_UNREACHABLE_SEGMENTS_HATCH_SPACE_FRACTION = 0.1 / 6;
const SLIDER_TICKS_FRACTIONS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
const SLIDER_TICKS_LENGTH = 9;
const SLIDER_TICKS_THICKNESS = SLIDER_TRACK_THICKNESS;

const COLORED_WHEEL_CUE_OUTER_RADIUS = 10;
const COLORED_WHEEL_CUE_OUTER_STROKE_THICKNESS = 2;
const COLORED_WHEEL_CUE_INNER_RADIUS = 2;
const COLORED_WHEEL_CUE_INNER_STROKE_THICKNESS = 1;
const COLOR_DIRECTION_CUE_CENTER_TO_SLIDER_EDGE = 20;

const SCORE_METER_WIDTH = 138;
const SCORE_METER_STROKE_THICKNESS = 1;
const SCORE_LABEL_RIGHT_TO_METER_LEFT = 10;
const SCORE_METER_RIGHT_TO_VALUE_LABEL_LEFT = 8;

const TRUE_HIDVAR_LINE_THICKNESS = 3;
const TRUE_HIDVAR_DOT_RADIUS = POS_OUTCOME_DOT_RADIUS;
const TRUE_HIDVAR_DOT_STROKE_THICKNESS = POS_OUTCOME_DOT_STROKE_THICKNESS;
const TRUE_HIDVAR_RANK_RIGHT_TO_INDICATOR_LEFT = 8;
const TRUE_HIDVAR_ESTIMATE_MARKER_HEIGHT = 10;
const TRUE_HIDVAR_ESTIMATE_MARKER_THICKNESS = 1;

const MOUSE_TARGET_PADDING_Y = 10;
const MOUSE_TARGET_PADDING_X = 10;

const TASK_SESSION_START_MOUSE_TARGET_WIDTH = SLIDER_THUMB_WIDTH + 2 * MOUSE_TARGET_PADDING_X;
const TASK_SESSION_START_MOUSE_TARGET_HEIGHT = SLIDER_THUMB_HEIGHT + 2 * MOUSE_TARGET_PADDING_Y;
const TASK_SESSION_START_MOUSE_TARGET_DISPLAY_OUTER_RADIUS = 7;
const TASK_SESSION_START_MOUSE_TARGET_DISPLAY_OUTER_STROKE_THICKNESS = 2;
const TASK_SESSION_START_MOUSE_TARGET_DISPLAY_INNER_RADIUS = 3;

const FEEDBACK_FORM_IFRAME_WIDTH = 700;
const FEEDBACK_FORM_IFRAME_HEIGHT = 430;

// Define the length of the slider track: 640px.
// This value is obtained from the below calculation
// - Desired degrees of visual angle between the left end and the middle of the
// slider track reachable segment
const SLIDER_TRACK_REACHABLE_SEGMENT_HALF_DEG_VISUAL_ANGLE = 6;// 5;
// - Degrees of visual angle corresponding to the eye's blind spot
const BLIND_SPOT_DEG_VISUAL_ANGLE = 13.5;
const SLIDER_LENGTH = (() => {
  const blindSpotToScreenCenterDistancePx = 584; // this was empirically measured
  const eyeToScreenCenterDistancePx = (blindSpotToScreenCenterDistancePx 
    / Math.tan(BLIND_SPOT_DEG_VISUAL_ANGLE * Math.PI / 180));
  const sliderTrackReachableSegmentHalfLengthPx = (
    eyeToScreenCenterDistancePx * Math.tan(
      SLIDER_TRACK_REACHABLE_SEGMENT_HALF_DEG_VISUAL_ANGLE * Math.PI / 180));
  let sliderLength = (
    sliderTrackReachableSegmentHalfLengthPx * 2
    / (SLIDER_TRACK_REACHABLE_FRACTION_RANGE_PROB[1]
      - SLIDER_TRACK_REACHABLE_FRACTION_RANGE_PROB[0]) );
  // round to even number
  sliderLength = 2 * Math.round(sliderLength / 2);
  return sliderLength; 
  })(); // this returns 640px

// Layout (in CSS pixels)

// Vertical layout for the display while performing the task
const STIMULUS_CENTER_Y_TO_SLIDER_TRACK_CENTER_Y = 85;
const SLIDER_TRACK_CENTER_Y_TO_TICKS_LABEL_BASELINE = 39;
const SLIDER_BOTTOM_ANCHOR_Y_TO_PROMPT_BASELINE = 50;

// Vertical layout for the display at the end of one session of the task
const SESSION_END_TITLE_BASELINE_TO_SCORE_DISPLAY_TOP = 36;
const SCORE_DISPLAY_HEIGHT = 14;
const SCORE_DISPLAY_BOTTOM_TO_TRUE_HIDVARS_TITLE_BASELINE = 46;
const TRUE_HIDVARS_TITLE_BASELINE_TO_INDICATOR_CENTER_Y = 22;
const TRUE_HIDVARS_INDICATOR_INTERSPACE_Y = 26;
const TRUE_HIDVARS_INDICATOR_CENTER_Y_TO_SLIDER_THUMB_TOP = 50;
const PROMPT_BASELINE_TO_REVIEW_INSTRUCTIONS_BUTTON_BASELINE = 80;

// Vertical layout for the task display when the last session has been completed
const PROMPT_BASELINE_TO_UPLOAD_INDICATOR_BASELINE = 22;
const PROMPT_BASELINE_TO_NEXT_TASK_BUTTON_BASELINE = 26;
const PROMPT_BASELINE_TO_COMPLETE_PROLIFIC_SUBMISSION_BUTTON_BASELINE = 26;
const PROMPT_BASELINE_TO_DOWNLOAD_DATA_FILE_BUTTON_BASELINE = 26;
const PROMPT_BASELINE_TO_FEEDBACK_FORM_TOP = Math.max(65,
  PROMPT_BASELINE_TO_COMPLETE_PROLIFIC_SUBMISSION_BUTTON_BASELINE);
const CANVAS_DISPLAY_MAX_HEIGHT_ABOVE_CENTER = 400;
const HEIGHT_ABOVE_CENTER_FOR_SESSION_END_DISPLAY_WITHOUT_TRUE_HIDVARS_INDICATORS = (
  SLIDER_THUMB_HEIGHT / 2
  + TRUE_HIDVARS_INDICATOR_CENTER_Y_TO_SLIDER_THUMB_TOP
  + TRUE_HIDVARS_TITLE_BASELINE_TO_INDICATOR_CENTER_Y
  + TEXT_FONT_SIZE
  + SCORE_DISPLAY_BOTTOM_TO_TRUE_HIDVARS_TITLE_BASELINE
  + SESSION_END_TITLE_BASELINE_TO_SCORE_DISPLAY_TOP
  + SESSION_END_TITLE_FONT_SIZE);

//
// Class definitions
//

class Display {
  constructor() {
    this._hidden = false;
  }

  get hidden() {
    return this._hidden;
  }

  set hidden(value) {
    if (value != this._hidden) {
      this._hidden = value;
      setNeedsRedraw();
    }
  }

  setHiddenWithoutRedraw(value = true) {
    this._hidden = value;
  }
}

class StackLayoutDisplay extends Display {
  constructor(displays, topAnchorY,
    spacing) {
    super();
    this.displays = displays;
    this.spacing = spacing;
    this.layoutDisplaysFromTopAnchorY(topAnchorY, spacing);
  }

  layoutDisplaysFromTopAnchorY(topAnchorY) {
    for (let display of this.displays) {
      display.topAnchorY = topAnchorY;
      topAnchorY = display.bottomAnchorY + this.spacing;
    }
  }

  get topAnchorY() {
    return this.displays[0].topAnchorY;
  }

  get bottomAnchorY() {
    return this.displays.at(-1).bottomAnchorY;
  }

  set topAnchorY(value) {
    this.layoutDisplaysFromTopAnchorY(value);
  }

  set centerX(value) {
    for (let display of this.displays) {
      display.centerX = value;
    }
  }

  draw(ctx) {
    if (this.hidden) {
      return;
    }

    for (let display of this.displays) {
      display.draw(ctx);
    }
  }
}

class Button extends Display {
  constructor(ctx, centerX, baselineY, text, clickAction,
    enabled = true, fontSize = BUTTON_FONT_SIZE) {
    super();
    this.centerX = centerX;
    this._text = text;
    this._textMetrics = measureText(ctx, text, fontSize);
    this._ctx = ctx;
    this.topAnchorY = baselineY;
    this._highlighted = false;
    this._enabled = enabled;
    this.pressed = false;
    this.clickAction = clickAction;
    this.fontSize = fontSize;
  }

  get baselineY() {
    return this.topAnchorY;
  }

  get bottomAnchorY() {
    return this.topAnchorY;
  }

  get color() {
    if (this.enabled) {
      return (this.highlighted ? BUTTON_TEXT_COLOR_HIGHLIGHTED : BUTTON_TEXT_COLOR);
    } else {
      return BUTTON_TEXT_COLOR_DISABLED;
    }
  }

  get text() {
    return this._text;
  }

  set text(value) {
    if (value != this._text) {
      this._text = value;
      this._updateTextMetrics();
      setNeedsRedraw();
    }
  }

  _updateTextMetrics() {
    this._textMetrics = measureText(this._ctx, this._text, this.fontSize);
  }

  get highlighted() {
    return this._highlighted;
  }

  set highlighted(value) {
    if (value != this._highlighted) {
      this._highlighted = value;
      setNeedsRedraw();
    }
  }

  get enabled() {
    return this._enabled;
  }

  set enabled(value) {
    if (value != this._enabled) {
      this._enabled = value;
      setNeedsRedraw();
    }
  }

  draw(ctx) {
    if (this.hidden) {
      return;
    }

    drawText(ctx, this.text, this.centerX, this.baselineY, this.fontSize, this.color);

    if (this._ctx != ctx) {
      this._ctx = ctx;
      this._updateTextMetrics();
    }
  }

  isInside(x, y) {
    if (this.enabled) {
      const xLeft = this.centerX - this._textMetrics.actualBoundingBoxLeft - MOUSE_TARGET_PADDING_X;
      const xRight = this.centerX + this._textMetrics.actualBoundingBoxRight + MOUSE_TARGET_PADDING_X;
      const yTop = this.baselineY - this._textMetrics.actualBoundingBoxAscent - MOUSE_TARGET_PADDING_Y;
      const yBottom = this.baselineY + this._textMetrics.actualBoundingBoxDescent + MOUSE_TARGET_PADDING_Y;
      const inside = (x >= xLeft && x <= xRight && y >= yTop && y <= yBottom);
      return inside;
    } else {
      return false;
    }
  }

  mouseDidEnter() {
    this.highlighted = true;
  }

  mouseDidExit() {
    this.highlighted = false;
    this.pressed = false;
  }

  mouseDidPressDown() {
    this.pressed = true;
  }

  mouseDidPressUp() {
    if (this.pressed) {
      this.mouseDidClick();
    }
    this.pressed = false;
  }

  mouseDidClick() {
    this.clickAction?.();
  }
}

class TextDisplay extends Display { // single line of text
  constructor(ctx, anchorX, anchorY, text,
    fontSize = TEXT_FONT_SIZE, color = TEXT_COLOR, bold = false,
    textAlign = "center",
    textBaseline = "alphabetic") {
    super();
    this.anchorX = anchorX;
    this.anchorY = anchorY;
    this.text = text;
    this.fontSize = fontSize;
    this.color = color;
    this.bold = bold;
    this.textAlign = textAlign;
    this.textBaseline = textBaseline;
  }

  get topAnchorY() {
    return this.anchorY;
  }

  set topAnchorY(value) {
    this.anchorY = value;
  }

  get bottomAnchorY() {
    return this.anchorY;
  }

  set centerX(value) {
    this.anchorX = value
  }

  draw(ctx) {
    if (this.hidden) {
      return;
    }

    drawText(ctx, this.text, this.anchorX, this.anchorY, this.fontSize, this.color,
      this.textAlign, this.textBaseline, this.bold);
  }
}

class MultilineTextDisplay extends Display {
  constructor(ctx, anchorX, topAnchorY, text,
    width,
    fontSize = TEXT_FONT_SIZE, color = TEXT_COLOR,
    lineHeight = MULTILINE_TEXT_LINE_HEIGHT) {
    super();
    this.anchorX = anchorX;
    this.topAnchorY = topAnchorY;
    this.text = text;
    this.width = width;
    this.fontSize = fontSize;
    this.color = color;
    this.lineHeight = (lineHeight ? lineHeight
      : getTextHeight(ctx, this.text, this.fontSize));
    this.textLines = this.calculateTextLines(ctx);
    this.topToBottomAnchor = this.lineHeight * (this.textLines.length - 1);
  }

  get bottomAnchorY() {
    return this.topAnchorY + this.topToBottomAnchor;
  }

  set centerX(value) {
    this.anchorX = value
  }

  draw(ctx) {
    if (this.hidden) {
      return;
    }

    ctx.font = getFont(this.fontSize);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = this.color;

    // print all lines of text
    let txtY = this.topAnchorY;
    this.textLines.forEach(txtline => {
      txtline = txtline.trim();
      ctx.fillText(txtline, this.anchorX, txtY);
      txtY += this.lineHeight;
    });
  }

  // adapted from https://github.com/geongeorge/Canvas-Txt/blob/master/src/index.js
  calculateTextLines(ctx) {
    const width = this.width;
    const lineHeight = this.lineHeight;

    ctx.font = getFont(this.fontSize);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';

    // added one-line only auto linebreak feature
    let textarray = [];
    let temptextarray = this.text.split('\n');
    temptextarray.forEach(txtt => {
      let textwidth = ctx.measureText(txtt).width;
      if (textwidth <= width) {
        textarray.push(txtt)
      } else {
        let temptext = txtt;
        let linelen = width;
        let textlen;
        let textpixlen;
        let texttoprint;
        textwidth = ctx.measureText(temptext).width;
        while (textwidth > linelen) {
          textlen = 0;
          textpixlen = 0;
          texttoprint = '';
          while (textpixlen < linelen) {
            textlen++;
            texttoprint = temptext.substr(0, textlen);
            textpixlen = ctx.measureText(temptext.substr(0, textlen)).width;
          }
          // Remove last character that was out of the box
          textlen--;
          texttoprint = texttoprint.substr(0, textlen);
          //if statement ensures a new line only happens at a space, and not amidst a word
          const backup = textlen;
          if (temptext.substr(textlen, 1) != ' ') {
            while (temptext.substr(textlen, 1) != ' ' && textlen != 0) {
              textlen--;
            }
            if (textlen == 0) {
              textlen = backup;
            }
            texttoprint = temptext.substr(0, textlen);
          }

          temptext = temptext.substr(textlen);
          textwidth = ctx.measureText(temptext).width;
          textarray.push(texttoprint);
        }
        if (textwidth > 0) {
          textarray.push(temptext);
        }
      }
      // end foreach temptextarray
    })

    return textarray;
  }
}

class DotStimulus extends Display {
  constructor(ctx, centerX, topY, color, hidden = false, radius = PROB_OUTCOME_DOT_RADIUS,
    stroked = false, strokeThickness = 0, strokeColor = BLACK_COLOR) {
    super();
    this.centerX = centerX;
    this.topY = topY;
    this.radius = radius;
    this.stroked = stroked;
    this.strokeThickness = strokeThickness;
    this.strokeColor = strokeColor;
    this._color = color;
    this.setHiddenWithoutRedraw(hidden);
  }

  get topAnchorY() {
    return this.topY;
  }

  set topAnchorY(value) {
    this.topY = value;
  }

  get centerY() {
    return this.topY + this.radius;
  }

  set centerY(value) {
    this.topY = value - this.radius;
  }

  get bottomAnchorY() {
    return this.topY + 2 * this.radius;
  }

  get color() {
    return this._color;
  }

  set color(value) {
    if (this._color != value) {
      this._color = value;
      setNeedsRedraw();
    }
  }

  draw(ctx) {
    if (this.hidden) {
      return;
    }

    ctx.beginPath();
    ctx.arc(this.centerX, this.centerY, this.radius, 0, Math.PI * 2);
    ctx.fillStyle = this.color;
    ctx.fill();
    ctx.closePath();
    if (this.stroked) {
      ctx.lineWidth = this.strokeThickness;
      ctx.strokeStyle = this.strokeColor;
      ctx.stroke();
    }
  }
}

// Ada-Prob outcome stimulus
class ProbOutcomeDotStimulus extends DotStimulus {
  constructor(ctx, centerX, topY, outcome, hidden = false, radius = PROB_OUTCOME_DOT_RADIUS) {
    const color = dotColorForOutcome(outcome);
    super(ctx, centerX, topY, color, hidden, radius);
    this._outcome = outcome;
  }

  get outcome() {
    return this._outcome;
  }

  set outcome(value) {
    if (this._outcome != value) {
      this._outcome = value;
      const color = dotColorForOutcome(value);
      this.color = color;
    }
  }
}

function dotColorForOutcome(outcome) {
  return outcome == 1 ? BLUE_DOT_COLOR : YELLOW_DOT_COLOR;
}

// Ada-Pos outcome stimulus
class PosOutcomeDotStimulus extends DotStimulus {
  constructor(ctx, outcomeCenterXRange, topY, outcome,
    hidden = false, radius = POS_OUTCOME_DOT_RADIUS,
    fillColor = POS_OUTCOME_DOT_FILL_COLOR,
    stroked = true, strokeThickness = POS_OUTCOME_DOT_STROKE_THICKNESS,
    strokeColor = POS_OUTCOME_DOT_STROKE_COLOR) {
    const centerX = dotCenterXForOutcome(outcome,
        outcomeCenterXRange[0], outcomeCenterXRange[1]);
    super(ctx, centerX, topY, fillColor, hidden, radius,
      stroked, strokeThickness, strokeColor);
    this._outcome = outcome;
    this._outcomeCenterXRange = outcomeCenterXRange;
  }

  get outcome() {
    return this._outcome;
  }

  get outcomeCenterXRange() {
    return this._outcomeCenterXRange;
  }

  set outcome(value) {
    if (this._outcome != value) {
      this._outcome = value;
      this.updateCenterX();
    }
  }

  set outcomeCenterXRange(value) {
    if (this._outcomeCenterXRange != value) {
      this._outcomeCenterXRange = value;
      this.updateCenterX();
    }
  }

  updateCenterX() {
    const centerX = dotCenterXForOutcome(this.outcome,
        this.outcomeCenterXRange[0], this.outcomeCenterXRange[1]);
    this.centerX = centerX;
  }
}

function dotCenterXForOutcome(outcome, minOutcomeCenterX, maxOutcomeCenterX) {
  return minOutcomeCenterX + outcome * (maxOutcomeCenterX - minOutcomeCenterX);
}

// Ada-Prob variant of the slider
class Slider extends Display {
  constructor(ctx, centerX, centerY, length = SLIDER_LENGTH, task = ADA_PROB,
    fractionValue = 0.5) {
    super();
    this.centerX = centerX;
    this.centerY = centerY;
    this.length = length;
    this._fractionValue = fractionValue;
    const thumb = {
      width: SLIDER_THUMB_WIDTH,
      height: SLIDER_THUMB_HEIGHT,
      strokeThickness: SLIDER_THUMB_STROKE_THICKNESS,
      fillColor: SLIDER_THUMB_FILL_COLOR,
      strokeColor: SLIDER_THUMB_STROKE_COLOR,
      cornerRadius: SLIDER_THUMB_CORNER_RADIUS,
    }
    this.thumb = thumb;
    const isAdaProb = (task == ADA_PROB);
    const reachableFractionRange = (isAdaProb ? 
      SLIDER_TRACK_REACHABLE_FRACTION_RANGE_PROB
      : SLIDER_TRACK_REACHABLE_FRACTION_RANGE_POS);
    const track = {
      thickness: SLIDER_TRACK_THICKNESS,
      color: SLIDER_TRACK_COLOR,
      reachableFractionRange: reachableFractionRange,
      unreachableSegments: {
        hatch: {
          length: SLIDER_TRACK_UNREACHABLE_SEGMENTS_HATCH_LENGTH,
          angle: SLIDER_TRACK_UNREACHABLE_SEGMENTS_HATCH_ANGLE,
          thickness: SLIDER_TRACK_UNREACHABLE_SEGMENTS_HATCH_THICKNESS,
          spaceFraction: SLIDER_TRACK_UNREACHABLE_SEGMENTS_HATCH_SPACE_FRACTION,
        },
        dash: {
          length: SLIDER_TRACK_UNREACHABLE_SEGMENTS_DASH_LENGTH,
          space: SLIDER_TRACK_UNREACHABLE_SEGMENTS_DASH_SPACE,
        },
      }
    }
    this.track = track;
    const ticks = {
      fractions: SLIDER_TICKS_FRACTIONS,
      length: SLIDER_TICKS_LENGTH,
      thickness: SLIDER_TICKS_THICKNESS,
      labelFontSize: SLIDER_TICKS_LABEL_FONT_SIZE,
      labelColor: SLIDER_TICKS_LABEL_COLOR,
    }
    this.ticks = ticks;
    this._highlighted = false;
    this._tickLabelsHidden = (isAdaProb ? false : true);
    this._unreachableSegmentsHidden = (isAdaProb ? false : true);
  }

  get topAnchorY() {
    return this.centerY;
  }

  set topAnchorY(value) {
    this.centerY = value;
  }

  get bottomAnchorY() {
    if (this._tickLabelsHidden) {
      return this.tickBottomY;
    } else {
      return this.tickLabelBaselineY;
    }
  }

  get tickLabelBaselineY() {
    return this.centerY + SLIDER_TRACK_CENTER_Y_TO_TICKS_LABEL_BASELINE;
  }

  get tickBottomY() {
    return this.centerY + this.ticks.length
  }

  get leftX() {
    return this.centerX - Math.round(this.length / 2);
  }

  get fractionValue() {
    return this._fractionValue;
  }

  set fractionValue(value) {
    if (this._fractionValue != value) {
      this._fractionValue = value;
      setNeedsRedraw();
    }
  }

  get highlighted() {
    return this._highlighted;
  }

  set highlighted(value) {
    if (value != this._highlighted) {
      this._highlighted = value;
      this.thumb.strokeColor = value ? BLACK_HIGHLIGHTED_COLOR : BLACK_COLOR;
      setNeedsRedraw();
    }
  }

  drawHatch(ctx, hatchFraction, halign="center") {
    let hatch = this.track.unreachableSegments.hatch;
    let fractionX = this.leftX + hatchFraction * this.length;
    let hatchSpanX = Math.cos(hatch.angle * Math.PI / 180) * hatch.length;
    let hatchSpanY = Math.sin(hatch.angle * Math.PI / 180) * hatch.length;
    let hatchTopY = this.centerY - Math.round(hatchSpanY / 2);
    let hatchBottomY = hatchTopY + hatchSpanY;
    let hatchLeftX;
    if (halign === "center") {
      hatchLeftX = fractionX - Math.round(hatchSpanX / 2);
    } else if (halign === "left") {
      hatchLeftX = fractionX;
    } else if (halign === "right") {
      hatchLeftX = fractionX - hatchSpanX;
    }
    let hatchRightX = hatchLeftX + hatchSpanX;
    ctx.lineWidth = hatch.thickness;
    ctx.strokeStyle = this.track.color;
    ctx.beginPath();
    ctx.moveTo(hatchLeftX, hatchBottomY);
    ctx.lineTo(hatchRightX, hatchTopY);
    ctx.stroke();
  }

  draw(ctx) {
    if (this.hidden) {
      return;
    }

    ctx.lineWidth = this.track.thickness;
    ctx.strokeStyle = this.track.color;
    // track: reachable segment
    let trackReachableSegmentLeftX = (this.centerX +
      (this.track.reachableFractionRange[0] - 0.5) * this.length);
    let trackReachableSegmentRightX = (this.centerX +
      (this.track.reachableFractionRange[1] - 0.5) * this.length);
    ctx.beginPath();
    ctx.moveTo(trackReachableSegmentLeftX, this.centerY);
    ctx.lineTo(trackReachableSegmentRightX, this.centerY);
    ctx.stroke();
    // track: unreachable segments - dashes
    if (!this._unreachableSegmentsHidden) {
      ctx.setLineDash([this.track.unreachableSegments.dash.length,
        this.track.unreachableSegments.dash.space]);
      ctx.beginPath();
      ctx.moveTo(this.leftX, this.centerY);
      ctx.lineTo(trackReachableSegmentLeftX, this.centerY);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(trackReachableSegmentRightX, this.centerY);
      ctx.lineTo(this.leftX + this.length, this.centerY);
      ctx.stroke();
      ctx.setLineDash([]);
      // track: unreachable segments - hatches
      let leftHatchFractionMax = (this.track.reachableFractionRange[0] -
        this.track.unreachableSegments.hatch.spaceFraction);
      let rightHatchFractionMin = (this.track.reachableFractionRange[1] +
        this.track.unreachableSegments.hatch.spaceFraction);
      let hatchFraction = 0.;
      while (hatchFraction < leftHatchFractionMax) {
        this.drawHatch(ctx, hatchFraction);
        hatchFraction += this.track.unreachableSegments.hatch.spaceFraction;
      }
      hatchFraction = 1.;
      while (hatchFraction > rightHatchFractionMin) {
        this.drawHatch(ctx, hatchFraction);
        hatchFraction -= this.track.unreachableSegments.hatch.spaceFraction;
      }
      this.drawHatch(ctx, leftHatchFractionMax);
      this.drawHatch(ctx, rightHatchFractionMin);
    }
    // ticks
    ctx.lineWidth = this.ticks.thickness;
    for (let i = 0; i < this.ticks.fractions.length; i++) {
      let tickFraction = this.ticks.fractions[i];
      ctx.beginPath();
      ctx.moveTo(this.leftX + tickFraction * this.length, this.centerY);
      ctx.lineTo(this.leftX + tickFraction * this.length,
        (this.centerY + this.ticks.length));
      ctx.stroke();
    }
    // tick labels
    if (!this._tickLabelsHidden) {
        for (let i = 0; i < this.ticks.fractions.length; i++) {
        let tickFraction = this.ticks.fractions[i];
        let tickCenterX = this.leftX + tickFraction * this.length;
        let tickLabelText = String((tickFraction * 100).toFixed(0)) + "%";
        drawText(ctx, tickLabelText, tickCenterX, this.tickLabelBaselineY,
          this.ticks.labelFontSize, this.ticks.labelColor,
          "center", "alphabetic")
      }
    }
    // thumb
    ctx.fillStyle = this.thumb.fillColor;
    ctx.strokeStyle = this.thumb.strokeColor;
    ctx.lineWidth = this.thumb.strokeThickness;
    let thumbLeftX = (this.leftX
      + this.fractionValue * this.length
      - Math.round(this.thumb.width / 2));
    let thumbTopY = this.centerY - Math.round(this.thumb.height / 2);
    drawRoundedRect(ctx, thumbLeftX, thumbTopY, this.thumb.width, this.thumb.height,
      this.thumb.cornerRadius);
  }
}

// Ada-Prob color-direction cues for the slider
class ColorDirectionCues extends Display {
  constructor(ctx, centerX, centerY, distanceBetweenCueCenters) {
    super();
    this.distanceBetweenCueCenters = distanceBetweenCueCenters;
    this._centerX = centerX;
    this._centerY = centerY;

    const leftColor = dotColorForOutcome(0);
    const rightColor = dotColorForOutcome(1);
    this.leftCue = new ColoredWheelCue(ctx, this.leftCueCenterX, centerY, leftColor);
    this.rightCue = new ColoredWheelCue(ctx, this.rightCueCenterX, centerY, rightColor);
  }

  get centerX() {
    return this._centerX;
  }

  get centerY() {
    return this._centerY;
  }

  set centerX(value) {
    if (this._centerX != value) {
      this._centerX = value;
      this.leftCue.centerX = this.leftCueCenterX;
      this.rightCue.centerX = this.rightCueCenterX;
    }
  }

  set centerY(value) {
    if (this._centerY != value) {
      this._centerY = value;
      this.leftCue.centerY = this._centerY;
      this.rightCue.centerY = this._centerY;
    }

  }

  get leftCueCenterX() {
    return this._centerX - Math.round(this.distanceBetweenCueCenters / 2);
  }

  get rightCueCenterX() {
    return this._centerX + Math.round(this.distanceBetweenCueCenters / 2);
  }

  draw(ctx) {
    if (this.hidden) {
      return;
    }

    this.leftCue.draw(ctx);
    this.rightCue.draw(ctx);
  }
}

class ColoredWheelCue extends Display {
  constructor(ctx, centerX, centerY, color, hidden = false,
    outerRadius = COLORED_WHEEL_CUE_OUTER_RADIUS,
    outerStrokeThickness = COLORED_WHEEL_CUE_OUTER_STROKE_THICKNESS,
    innerRadius = COLORED_WHEEL_CUE_INNER_RADIUS,
    innerStrokeThickness = COLORED_WHEEL_CUE_INNER_STROKE_THICKNESS,
    stroked = false, strokeThickness = 0, strokeColor = BLACK_COLOR) {
    super();
    const outerCircle = new DotStimulus(ctx, 0, 0, color, hidden, outerRadius,
      true, outerStrokeThickness, BLACK_COLOR);
    const innerCircle = new DotStimulus(ctx, 0, 0, WHITE_COLOR, hidden, innerRadius,
      true, innerStrokeThickness, BLACK_COLOR);
    outerCircle.centerX = centerX;
    innerCircle.centerX = centerX;
    outerCircle.centerY = centerY;    
    innerCircle.centerY = centerY;
    this._outerCircle = outerCircle;
    this._innerCircle = innerCircle;
  }

  set centerX(value) {
    this._outerCircle.centerX = value;
    this._innerCircle.centerX = value;
  }

  set centerY(value) {
    this._outerCircle.centerY = value;
    this._innerCircle.centerY = value;
  }

  get topAnchorY() {
    return this._outerCircle.topAnchorY;
  }

  set topAnchorY(value) {
    let centerY = value + this._outerCircle.radius;
    this.setCenterY(centerY);
  }

  get bottomAnchorY() {
    return this._outerCircle.bottomAnchorY;
  }

  get color() {
    return this._outerCircle.color;
  }

  set color(value) {
    this._outerCircle.color = value;
  }
  

  draw(ctx) {
    if (this.hidden) {
      return;
    }

    this._outerCircle.draw(ctx);
    this._innerCircle.draw(ctx);
  }
}

// Display indicating one value of the hidden variable
class HidvarIndicator extends Display {
  constructor(ctx, centerX, centerY, length = SLIDER_LENGTH, fraction = 0.5,
    lineThickness = TRUE_HIDVAR_LINE_THICKNESS,
    indicatorDotRadius = TRUE_HIDVAR_DOT_RADIUS,
    indicatorStrokeThickness = TRUE_HIDVAR_DOT_STROKE_THICKNESS,
    indicatorFillColor = TRUE_HIDVAR_DOT_FILL_COLOR,
    indicatorStrokeColor = TRUE_HIDVAR_DOT_STROKE_COLOR) {
    super();
    this.centerX = centerX;
    this.centerY = centerY;
    this.length = length;
    this._fraction = fraction;
    this.lineThickness = lineThickness;
    this.indicatorDotRadius = indicatorDotRadius
    this.indicatorStrokeThickness = indicatorStrokeThickness;
    this.indicatorFillColor = indicatorFillColor;
    this.indicatorStrokeColor = indicatorStrokeColor;
  }

  get topAnchorY() {
    return this.centerY;
  }

  set topAnchorY(value) {
    this.centerY = value
  }

  get bottomAnchorY() {
    return this.centerY;
  }

  get leftX() {
    return this.centerX - Math.round(this.length / 2);
  }

  get fraction() {
    return this._fraction;
  }

  set fraction(value) {
    if (value != this._fraction) {
      this._fraction = value;
      setNeedsRedraw();
    }
  }

  draw(ctx) {
    if (this.hidden) {
      return;
    }

    let dotCenterX = this.leftX + this.fraction * this.length;
    let leftSegmentLength = (this.fraction * this.length - this.indicatorDotRadius);
    let rightSegmentLength = this.length - leftSegmentLength - 2 * this.indicatorDotRadius;
    let segmentTopY = this.centerY - Math.round(this.lineThickness / 2);
    // left segment
    ctx.fillStyle = this.indicatorStrokeColor;
    ctx.fillRect(
      this.leftX,
      segmentTopY,
      leftSegmentLength,
      this.lineThickness);
    // right segment
    ctx.fillStyle = this.indicatorStrokeColor;
    ctx.fillRect(
      this.leftX + this.length - rightSegmentLength,
      segmentTopY,
      rightSegmentLength,
      this.lineThickness);
    // dot marker indicator
    ctx.fillStyle = this.indicatorFillColor;
    ctx.lineWidth = this.indicatorStrokeThickness;
    ctx.strokeStyle = this.indicatorStrokeColor;
    ctx.beginPath();
    ctx.arc(dotCenterX, this.centerY, this.indicatorDotRadius, 0, Math.PI * 2);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
}

// Display showing the true value of the hidden variable for one stable period
// and its rank of occurrence in the session
class LeftLabeledHidvarIndicator extends Display {
  constructor(ctx, centerX, centerY, text, length = SLIDER_LENGTH, fraction = 0.5,
    labelFontSize = TRUE_HIDVAR_RANK_FONT_SIZE,
    labelColor = TRUE_HIDVAR_LABEL_COLOR) {
    super();
    const indicator = new HidvarIndicator(ctx, centerX, centerY, length,
      fraction);
    const labelRightX = (centerX - Math.round(length / 2)
      - TRUE_HIDVAR_RANK_RIGHT_TO_INDICATOR_LEFT);
    const label = new TextDisplay(ctx, labelRightX, centerY, text,
        TRUE_HIDVAR_RANK_FONT_SIZE, TRUE_HIDVAR_LABEL_COLOR, false,
        "right", "middle");
    this.indicator = indicator;
    this.label = label;
  }

  set centerX(value) {
    this.indicator.centerX = value;
    const labelRightX = (this.indicator.leftX
      - TRUE_HIDVAR_RANK_RIGHT_TO_INDICATOR_LEFT);
    this.label.anchorX = labelRightX; // because align="right", centerX acts as right anchor
  }

  get centerY() {
    return this.indicator.centerY;
  }

  set centerY(value) {
    this.indicator.centerY = value;
    this.label.anchorY = value;
  }

  get topAnchorY() {
    return this.centerY;
  }

  set topAnchorY(value) {
    this.centerY = value
  }

  get bottomAnchorY() {
    return this.centerY;
  }

  draw(ctx) {
    if (this.hidden) {
      return;
    }
    this.indicator.draw(ctx);
    this.label.draw(ctx);
  } 
}

// Display for one stable period showing the true value and subject's estimates
// of the hidden variable and subjects' estimates superimposed.
class TrueHidvarAndEstimatesIndicator extends LeftLabeledHidvarIndicator {
  get estimates() {
    return this._estimates;
  }

  set estimates(estimates) {
    if (this._estimates != estimates) {
      this._estimates = estimates;
      // setNeedsRedraw();
    }
  }

  draw(ctx) {
    if (this.hidden) {
      return;
    }
    
    super.draw(ctx);
    if (this.estimates?.length > 0) {
      ctx.fillStyle = TRUE_HIDVAR_ESTIMATE_MARKER_COLOR;
      for (let estimate of this.estimates) {
        const leftX = this.indicator.leftX + (estimate * this.indicator.length
          - Math.round(TRUE_HIDVAR_ESTIMATE_MARKER_THICKNESS / 2));
        const topY = (this.indicator.centerY
          - Math.round(TRUE_HIDVAR_ESTIMATE_MARKER_HEIGHT / 2));
        ctx.fillRect(
          leftX,
          topY,
          TRUE_HIDVAR_ESTIMATE_MARKER_THICKNESS,
          TRUE_HIDVAR_ESTIMATE_MARKER_HEIGHT);
      }
    }
  }
}

// Display showing all true values and subject's estimates of the hidden variable
// for the whole session, with each value for each stable period listed vertically
class TrueHidvarsDisplay extends Display {
  constructor(ctx, centerX, bottomAnchorY, width = SLIDER_LENGTH,
    task = ADA_PROB) {
    super();
    this.task = task;
    const titleText = (this.task == ADA_PROB) ? lang.trueProbabilitiesTitle : lang.truePositionsTitle;
    this.titleLabel = new TextDisplay(ctx, centerX, bottomAnchorY,
      titleText, TEXT_FONT_SIZE, TRUE_HIDVAR_LABEL_COLOR);
    this._bottomAnchorY = bottomAnchorY;
    this._centerX = centerX;
    this.width = width;
    this.setTrueValues(ctx, []);
  }

  get centerX() {
    return this._centerX;
  }

  set centerX(value) {
    if (value != this._centerX) {
      this._centerX = value;
      this.titleLabel.anchorX = value;
      for (let indicator of this.hidvarIndicators) {
        indicator.centerX = value;
      }
    }
  }

  get topAnchorY() {
    return this.titleLabel.anchorY;
  }

  get bottomAnchorY() {
    return this._bottomAnchorY;
  }

  set bottomAnchorY(value) {
    if (value != this._bottomAnchorY) {
      this._bottomAnchorY = value;
      this.layoutIndicatorsFromBottomAnchorY(this.hidvarIndicators, value);
    }
  }

  get hidvarIndicators() {
    return this._hidvarIndicators;
  }

  setTrueValues(ctx, trueValues, estimatesPerTrueValue) {
    let hidvarIndicators = [];
    for (let i = 0; i < trueValues.length; i++) {
      const hidvarIndicator = new TrueHidvarAndEstimatesIndicator(ctx,
        this.centerX, 0,
        lang.rankTextWithIndex(i),
        this.width, trueValues[i]);
      if (estimatesPerTrueValue && (i < estimatesPerTrueValue.length)) {
        hidvarIndicator.estimates = estimatesPerTrueValue[i];
      }

      hidvarIndicators.push(hidvarIndicator);
    }
    this._hidvarIndicators = hidvarIndicators;
    if (trueValues.length == 1) { // singular
      this.titleLabel.text = (this.task == ADA_PROB) ? lang.trueProbabilitySingularTitle : lang.truePositionSingularTitle;
    } else { // plural
      this.titleLabel.text = (this.task == ADA_PROB) ? lang.trueProbabilitiesTitle : lang.truePositionsTitle;
    }
    this.layoutIndicatorsFromBottomAnchorY(hidvarIndicators, this.bottomAnchorY);
  }

  draw(ctx) {
    if (this.hidden) {
      return;
    }

    this.titleLabel.draw(ctx);
    for (let display of this.hidvarIndicators) {
      display.draw(ctx);
    }
  }

  layoutIndicatorsFromBottomAnchorY(indicators, bottomAnchorY) {
    const availableHeightAboveCenter = Math.min(CANVAS_DISPLAY_MAX_HEIGHT_ABOVE_CENTER,
      window.innerHeight / 2);
    const maxHeight = Math.floor(availableHeightAboveCenter
      - HEIGHT_ABOVE_CENTER_FOR_SESSION_END_DISPLAY_WITHOUT_TRUE_HIDVARS_INDICATORS);
    const desiredHeight = (indicators.length > 0 ?
      (indicators.length - 1) * TRUE_HIDVARS_INDICATOR_INTERSPACE_Y
      : 0);
    let actualHeight = desiredHeight;
    let interspaceY = TRUE_HIDVARS_INDICATOR_INTERSPACE_Y;
    if (desiredHeight > maxHeight) {
      actualHeight = maxHeight;
      interspaceY = maxHeight / (indicators.length - 1);
    }
    let indicatorCenterY = bottomAnchorY;
    for (let i = indicators.length - 1; i >= 0; i--) {
      indicators[i].centerY = indicatorCenterY;
      indicatorCenterY -= interspaceY;
    }
    const titleBaselineToBottomAnchor = (indicators.length > 0 ?
      actualHeight + TRUE_HIDVARS_TITLE_BASELINE_TO_INDICATOR_CENTER_Y
      : 0);
    this.titleLabel.topAnchorY = bottomAnchorY - titleBaselineToBottomAnchor;
  }
}

// Display for the subject's score
class ScoreDisplay extends Display {
  constructor(ctx, centerX, centerY, height = SCORE_DISPLAY_HEIGHT,
    label = {
      text: lang.score,
      fontSize: SCORE_LABEL_FONT_SIZE,
      color: SCORE_DISPLAY_LABEL_COLOR
    },
    valueLabel = {
      text: "",
      fontSize: SCORE_LABEL_FONT_SIZE,
      color: SCORE_DISPLAY_LABEL_COLOR,
    },
    meter = {
      valueFrac: 0.61,
      width: SCORE_METER_WIDTH,
      fillColor: SCORE_METER_FILL_COLOR,
      strokeColor: SCORE_METER_STROKE_COLOR,
      thickness: SCORE_METER_STROKE_THICKNESS,
    }) {
    super();
    this.centerX = centerX;
    this.centerY = centerY;
    this.height = height;
    this.label = label;
    this.valueLabel = valueLabel;
    this.meter = meter;
  }

  get topY() {
    return this.centerY - Math.round(this.height / 2);
  }

  set topY(value) {
    this.centerY = value + Math.round(this.height / 2);
  }

  get topAnchorY() {
    return this.topY;
  }

  set topAnchorY(value) {
    this.topY = value;
  }

  get bottomY() {
    return this.topY + this.height;
  }

  get bottomAnchorY() {
    return this.bottomY;
  }

  set bottomY(value) {
    this.centerY = value - Math.round(this.height / 2);
  }

  get leftX() {
    return this.centerX - Math.round(this.meter.width / 2);
  }

  updateWithScoreFrac(scoreFrac,
    gain = null, accumulatedGain = null) {
    this.meter.valueFrac = scoreFrac;
    this.valueLabel.text = lang.valueLabelTextForScoreFrac(scoreFrac,
      gain, accumulatedGain); 
  }

  draw(ctx) {
    if (this.hidden) {
      return;
    }

    // meter
    const fillWidth = Math.round(this.meter.valueFrac * this.meter.width);
    const leftX = this.leftX;
    const topY = this.topY;
    ctx.fillStyle = this.meter.fillColor;
    ctx.strokeStyle = this.meter.strokeColor;
    ctx.fillRect(
      leftX,
      topY,
      fillWidth,
      this.height);
    ctx.strokeRect(
      leftX,
      topY,
      fillWidth,
      this.height);
    ctx.strokeRect(
      leftX,
      topY,
      this.meter.width,
      this.height);
    // labels
    const scoreLabelRight = (leftX
      - SCORE_LABEL_RIGHT_TO_METER_LEFT);
    const valueLabelLeft = (leftX
      + this.meter.width
      + SCORE_METER_RIGHT_TO_VALUE_LABEL_LEFT);
    drawText(ctx, this.label.text, scoreLabelRight, this.centerY,
      this.label.fontSize, this.label.color,
      "right", "middle");
    if (this.valueLabel.text) {
      drawText(ctx, this.valueLabel.text, valueLabelLeft, this.centerY,
      this.valueLabel.fontSize, this.valueLabel.color,
      "left", "middle");
    }
  }
}

// Display for the task, shown both during and at the end of a task session,
// drawn on the <canvas> element.
// This is class shared for both tasks (Ada-Prob and Ada-Pos).
// Task-specific UI configuration is performed within the code of this class.
class TaskSessionDisplay extends Display {
  constructor(ctx, centerX, centerY, sliderLength, task = ADA_PROB, taskIdx = null,
    mouseTargetClickAction = null,
    completeProlificSubmissionButtonClickAction = null,
    downloadDataFileButtonClickAction = null,
    estimate = 0.5, showSessionEnd = false) {
    super();
    this._estimate = estimate;
    this._showSessionEnd = showSessionEnd;
    this.slider = new Slider(ctx, centerX, 0, sliderLength, task, estimate);
    if (task == ADA_PROB) {
      this.dotStimulus = new ProbOutcomeDotStimulus(ctx, centerX, 0, 1, true);
      const distanceBetweenCueCenters = (sliderLength + COLOR_DIRECTION_CUE_CENTER_TO_SLIDER_EDGE * 2);
      this.colorDirectionCues = new ColorDirectionCues(ctx, centerX, centerY, distanceBetweenCueCenters);
    } else {
      const outcomeCenterXRange = [
        Math.round(centerX - this.slider.length / 2),
        Math.round(centerX + this.slider.length / 2)];
      this.dotStimulus = new PosOutcomeDotStimulus(ctx, outcomeCenterXRange, 0, 0.5, true);
    }
    this.prompt = new TextDisplay(ctx, centerX, 0,
      lang.promptStartFirstSession, PROMPT_FONT_SIZE);
    this.mouseTarget = new TaskSessionStartMouseTarget(centerX, centerY, mouseTargetClickAction);
    this.mouseTarget.hidden = true;
    // displays for session end screen
    this.trueHidvarsDisplay = new TrueHidvarsDisplay(ctx, centerX,
      0, sliderLength, task);
    this.scoreDisplay = new ScoreDisplay(ctx, centerX, 0);
    this.sessionEndTitleLabel = new TextDisplay(ctx, centerX, 0, "",
      SESSION_END_TITLE_FONT_SIZE, SESSION_END_TITLE_COLOR, SESSION_END_TITLE_BOLD);
    this.reviewInstructionsButton = new Button(ctx, centerX, 0,
      lang.reviewInstructionsButtonText, null, null,
      REVIEW_INSTRUCTIONS_BUTTON_FONT_SIZE);
    this.nextTaskButton = new Button(ctx, centerX, 0,
      taskIdx !== null ? lang.nextTaskButtonTextWithIndex(taskIdx+1) : "",
      null, false,
      PROMPT_FONT_SIZE);
    this.uploadIndicator = new TextDisplay(ctx, centerX, 0, "", PROMPT_FONT_SIZE);
    this.uploadIndicator.hidden = true;
    this.nextTaskButton.hidden = true;
    this.completeProlificSubmissionButton = new Button(ctx, centerX, 0,
      lang.completeProlificSubmissionButtonText,
      completeProlificSubmissionButtonClickAction, false,
      PROMPT_FONT_SIZE);
    this.downloadDataFileButton = new Button(ctx, centerX, 0,
      lang.downloadDataFileButtonText,
      downloadDataFileButtonClickAction, false,
      PROMPT_FONT_SIZE);
    this.isCompleteProlificSubmissionButtonShown = false;
    this.isDownloadDataFileButtonShown = false;
    this.isFeedbackFormShown = false;
    this.task = task;
    
    this.layoutForCenterCoordinates(centerX, centerY);
  }

  get outcome() {
    return this.dotStimulus.outcome;
  }

  set outcome(value) {
    this.dotStimulus.outcome = value;
  }

  get estimate() {
    return this._estimate;
  }

  set estimate(value) {
    if (value != this._estimate) {
      this._estimate = value;
      this.slider.fractionValue = value;
    }
  }

  set estimateHighlighted(value) {
    this.slider.highlighted = value;
  }

  get isShowingSessionEnd() {
    return this._showSessionEnd;
  }

  set showSessionEnd(value) {
    if (value != this._showSessionEnd) {
      this._showSessionEnd = value;
      setNeedsRedraw();
    }
  }

  get centerX() {
    return this.slider.centerX;
  }

  get centerY() {
    return this.slider.centerY;
  }

  get mouseTargets() {
    if (this.isCompleteProlificSubmissionButtonShown) {
      return [this.mouseTarget,
              this.reviewInstructionsButton,
              this.nextTaskButton,
              this.completeProlificSubmissionButton];
    } else if (this.isDownloadDataFileButtonShown) {
      return [this.mouseTarget,
              this.reviewInstructionsButton,
              this.nextTaskButton,
              this.downloadDataFileButton];
    } else {
      return [this.mouseTarget,
              this.nextTaskButton,
              this.reviewInstructionsButton];
    }
  }

  setCenterCoordinates(centerX, centerY) {
    this.layoutForCenterCoordinates(centerX, centerY);
  }

  setTrueValues(ctx, trueValues, estimatesPerTrueValue) {
    this.trueHidvarsDisplay.setTrueValues(ctx, trueValues, estimatesPerTrueValue);
    this.layoutForCenterCoordinates(this.centerX, this.centerY);
  }

  layoutForCenterCoordinates(centerX, centerY) {
    this.slider.centerX = centerX;
    if (this.task == ADA_PROB) {
      this.colorDirectionCues.centerX = centerX;
      this.dotStimulus.centerX = centerX;
    } else {
      const outcomeCenterXRange = [
        Math.round(centerX - this.slider.length / 2),
        Math.round(centerX + this.slider.length / 2)];
      this.dotStimulus.outcomeCenterXRange = outcomeCenterXRange;
    }
    this.prompt.anchorX = centerX;
    this.trueHidvarsDisplay.centerX = centerX;
    this.scoreDisplay.centerX = centerX;
    this.sessionEndTitleLabel.anchorX = centerX;
    this.reviewInstructionsButton.centerX = centerX;
    this.nextTaskButton.centerX = centerX;
    this.uploadIndicator.centerX = centerX;
    if (this.isCompleteProlificSubmissionButtonShown) {
      this.completeProlificSubmissionButton.centerX = centerX;
    }
    if (this.isDownloadDataFileButtonShown) {
      this.downloadDataFileButton.centerX = centerX;
    }

    this.slider.centerY = centerY;
    if (this.task == ADA_PROB) {
      this.colorDirectionCues.centerY = centerY;
      this.dotStimulus.centerY = (this.slider.centerY
        - STIMULUS_CENTER_Y_TO_SLIDER_TRACK_CENTER_Y);
    } else {
      this.dotStimulus.centerY = centerY;
    }
    this.prompt.anchorY = (this.slider.bottomAnchorY
      + SLIDER_BOTTOM_ANCHOR_Y_TO_PROMPT_BASELINE);
    this.trueHidvarsDisplay.bottomAnchorY = (this.slider.centerY
      - SLIDER_THUMB_HEIGHT / 2
      - TRUE_HIDVARS_INDICATOR_CENTER_Y_TO_SLIDER_THUMB_TOP);
    this.scoreDisplay.bottomY = (this.trueHidvarsDisplay.topAnchorY
      - SCORE_DISPLAY_BOTTOM_TO_TRUE_HIDVARS_TITLE_BASELINE);
    this.sessionEndTitleLabel.anchorY = (this.scoreDisplay.topY
      - SESSION_END_TITLE_BASELINE_TO_SCORE_DISPLAY_TOP);
    this.reviewInstructionsButton.topAnchorY = (this.prompt.anchorY +
      PROMPT_BASELINE_TO_REVIEW_INSTRUCTIONS_BUTTON_BASELINE);
    this.nextTaskButton.topAnchorY = (this.prompt.anchorY +
        PROMPT_BASELINE_TO_NEXT_TASK_BUTTON_BASELINE);
    this.uploadIndicator.anchorY = (this.prompt.anchorY +
      PROMPT_BASELINE_TO_UPLOAD_INDICATOR_BASELINE);
    if (this.isCompleteProlificSubmissionButtonShown) {
      this.completeProlificSubmissionButton.topAnchorY = (this.prompt.anchorY +
        PROMPT_BASELINE_TO_COMPLETE_PROLIFIC_SUBMISSION_BUTTON_BASELINE);
    }
    if (this.isDownloadDataFileButtonShown) {
      this.downloadDataFileButton.topAnchorY = (this.prompt.anchorY +
        PROMPT_BASELINE_TO_DOWNLOAD_DATA_FILE_BUTTON_BASELINE);
    }
    if (this.isFeedbackFormShown) {
      this.feedbackForm.yTop = (this.prompt.anchorY
        + PROMPT_BASELINE_TO_FEEDBACK_FORM_TOP);
    }

    this.mouseTarget.setCenterCoordinates(centerX, centerY);
  }

  draw(ctx) {
    if (this.isShowingSessionEnd) {
      this.sessionEndTitleLabel.draw(ctx);
      this.scoreDisplay.draw(ctx);
      this.trueHidvarsDisplay.draw(ctx);
      if (this.task == ADA_PROB) {
        this.colorDirectionCues.draw(ctx);
      }
      this.slider.draw(ctx);
      this.prompt.draw(ctx);
      this.uploadIndicator.draw(ctx);
      this.reviewInstructionsButton.draw(ctx);
      this.nextTaskButton.draw(ctx);
      this.mouseTarget.draw(ctx);
      if (this.isCompleteProlificSubmissionButtonShown) {
        this.completeProlificSubmissionButton.draw(ctx);
      }
      if (this.isDownloadDataFileButtonShown) {
        this.downloadDataFileButton.draw(ctx);
      }
    } else {
      if (this.task == ADA_PROB) {
        this.colorDirectionCues.draw(ctx);
      }
      this.slider.draw(ctx);
      this.prompt.draw(ctx);
      this.dotStimulus.draw(ctx);
    }
  }

  startUploadAnimation(character = ".", length = 4, delay = 500) {
    this._loadingAnimation = new AnimationCycleLoop(length, delay, (n) => {
      this.uploadIndicator.text = character.repeat(n);
      setNeedsRedraw();
    });
    this.uploadIndicator.hidden = false;
    this._loadingAnimation.start(true);
  }

  stopUploadAnimation() {
    this._loadingAnimation?.stop();
    this.uploadIndicator.hidden = true;
    setNeedsRedraw();
  }

  showCompleteProlificSubmissionButton() {
    this.completeProlificSubmissionButton.enabled = true;
    this.isCompleteProlificSubmissionButtonShown = true;
    this.layoutForCenterCoordinates(this.centerX, this.centerY);
    setNeedsRedraw();
  }

  showDownloadDataFileButton() {
    this.downloadDataFileButton.enabled = true;
    this.isDownloadDataFileButtonShown = true;
    this.layoutForCenterCoordinates(this.centerX, this.centerY);
    setNeedsRedraw();
  }

  showFeedbackForm() {
    if (!this.isFeedbackFormShown) {
      this.feedbackForm = new FeedbackForm(studyParams.feedbackFormSrc, 0);
      this.isFeedbackFormShown = true;
      this.layoutForCenterCoordinates(this.centerX, this.centerY);
      document.body.appendChild(this.feedbackForm.domElement);
    }    
  }
}

// Display and mouse interaction for the target at the center of the slider
// that subjects must click to start the next session.
class TaskSessionStartMouseTarget extends Display {
  constructor(centerX, centerY,
    clickAction,
    enabled = true,
    hidden = true,
    width = TASK_SESSION_START_MOUSE_TARGET_WIDTH,
    height = TASK_SESSION_START_MOUSE_TARGET_HEIGHT) {
    super();
    this.width = width;
    this.height = height;
    this.pressed = false;
    this.clickAction = clickAction;
    this.setCenterCoordinates(centerX, centerY);
    this.enabled = enabled;
    this.hidden = hidden;
    this._highlighted = false;
  }

  setCenterCoordinates(centerX, centerY) {
    this.xLeft = centerX - this.width / 2;
    this.xRight = centerX + this.width / 2;
    this.yTop = centerY - this.height / 2;
    this.yBottom = centerY + this.height / 2;
  }

  isInside(x, y) {
    if (this.enabled) {
      const inside = (x >= this.xLeft && x <= this.xRight && y >= this.yTop && y <= this.yBottom);
      return inside;
    }
  }

  mouseDidEnter() {
    if (!this.hidden) {
      this.highlighted = true;
    }
  }

  mouseDidExit() {
    this.highlighted = false;
    this.pressed = false;
  }

  mouseDidPressDown() {
    this.pressed = true;
  }

  mouseDidPressUp() {
    if (this.pressed) {
      this.mouseDidClick();
    }
    this.pressed = false;
  }

  mouseDidClick() {
    this.highlighted = false;
    this.clickAction?.();
  }

  get highlighted() {
    return this._highlighted;
  }

  set highlighted(value) {
    if (value != this._highlighted) {
      this._highlighted = value;
      if (!this.hidden) {
        setNeedsRedraw();
      }
    }
  }

  get centerX() {
    return (this.xLeft + this.xRight) / 2;
  }

  get centerY() {
    return (this.yTop + this.yBottom) / 2;
  }

  get color() {
    if (this.enabled) {
      return (this.highlighted ? BUTTON_TEXT_COLOR_HIGHLIGHTED : BUTTON_TEXT_COLOR);
    } else {
      return BUTTON_TEXT_COLOR_DISABLED;
    }
  }

  draw(ctx) {
    if (this.hidden) {
      return;
    }

    const centerX = this.centerX;
    const centerY = this.centerY;
    const color = this.color;
    ctx.beginPath();
    ctx.arc(centerX, centerY, TASK_SESSION_START_MOUSE_TARGET_DISPLAY_INNER_RADIUS,
      0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.closePath();
    
    ctx.beginPath();
    ctx.arc(centerX, centerY, TASK_SESSION_START_MOUSE_TARGET_DISPLAY_OUTER_RADIUS,
      0, Math.PI * 2);
    ctx.lineWidth = TASK_SESSION_START_MOUSE_TARGET_DISPLAY_OUTER_STROKE_THICKNESS;
    ctx.strokeStyle = color;
    ctx.stroke();
  }
}

// Used to track when the mouse pointer enters/exits or moves within
// a certain target location among some defined targets on the canvas.
class MouseTracker {
  constructor(updateForMouseMovement,
    getCurrentMouseTargets) {
    this.isTrackingMouseMovement = false;
    this.updateForMouseMovement = updateForMouseMovement;
    this.getCurrentMouseTargets = getCurrentMouseTargets;
  }

  get canvas() {
    return this._canvas;
  }

  set canvas(canvas) {
    this._canvas = canvas;
    canvas.onmousemove = (e) => { this.mouseDidMove(e) };
    canvas.onmousedown = (e) => { this.mouseDidPressDown(e) };
    canvas.onmouseup = (e) => { this.mouseDidPressUp(e) };
  }

  mouseDidMove(mouseEvent) {
    if (this.isTrackingMouseMovement) {
      // updateEstimateForMouseMovement(mouseEvent.movementX)
      // setNeedsRedraw();
      this.updateForMouseMovement?.(mouseEvent.movementX);
    }
    if (this.areMouseTargetsEnabled) {
      const mouseX = mouseEvent.pageX;
      const mouseY = mouseEvent.pageY;
      const mouseTargets = this.getCurrentMouseTargets();
      let mouseIsInsideAnyTarget = false;
      for (let target of mouseTargets) {
        const mouseWasInsideTarget = target.mouseIsInside;
        const mouseIsInsideTarget = target.isInside(mouseX, mouseY);
        if (mouseIsInsideTarget && !mouseWasInsideTarget) {
          target.mouseDidEnter?.();
        }
        if (mouseIsInsideTarget && mouseWasInsideTarget) {
          target.mouseDidMove?.();
        }
        if (!mouseIsInsideTarget && mouseWasInsideTarget) {
          target.mouseDidExit?.();
        }
        target.mouseIsInside = mouseIsInsideTarget;
        mouseIsInsideAnyTarget = (mouseIsInsideAnyTarget || mouseIsInsideTarget);
      }
      if (mouseIsInsideAnyTarget) {
        this.canvas.style.cursor = "pointer";
      } else {
        this.canvas.style.cursor = "auto";
      }
    }
  }
  
  get areMouseTargetsEnabled() {
    return !this.isTrackingMouseMovement;
  }

  mouseDidPressDown(mouseEvent) {
    if (this.areMouseTargetsEnabled) {
      const mouseX = mouseEvent.pageX;
      const mouseY = mouseEvent.pageY;
      const mouseTargets = this.getCurrentMouseTargets();
      for (let target of mouseTargets) {
        if (target.isInside(mouseX, mouseY)) {
          target?.mouseDidPressDown();
        }
      }
    }
  }

  mouseDidPressUp(mouseEvent) {
    if (this.areMouseTargetsEnabled) {
      const mouseX = mouseEvent.pageX;
      const mouseY = mouseEvent.pageY;
      const mouseTargets = this.getCurrentMouseTargets();
      for (let target of mouseTargets) {
        if (target.isInside(mouseX, mouseY)) {
          target?.mouseDidPressUp();
        }
      }
    }
  }

  startTrackingMouseMovement() {
    this.isTrackingMouseMovement = true;
    this.canvas.style.cursor = "none";
  }

  stopTrackingMouseMovement() {
    this.isTrackingMouseMovement = false;
    this.canvas.style.cursor = "auto";
  }
}

class AnimationCycleLoop {
  constructor(length, delay, handler) {
    this.length = length;
    this.delay = delay;
    this.handler = handler;
  }

  start(doInitialCallback = true) {
    this._counter = 0;
    this._intervalID = setInterval(() => {
      this._counter = (this._counter + 1) % this.length;
      this.handler?.(this._counter);
    }, this.delay);
    if (doInitialCallback) {
      this.handler?.(this._counter);
    }
  }

  stop() {
    if (this._intervalID !== null && this._intervalID !== undefined) {
      clearInterval(this._intervalID);
    }
    this._intervalID = null;
  }
}

// Display for the task instructions (DOM-based)
class InstructionDisplay {
  constructor(studyName, hidden = true) {
    let containerElement = document.createElement('div');
    containerElement.id = "instruction-container";
    containerElement.innerHTML = this.getInnerHTMLString();
    containerElement.hidden = hidden;
    this.containerElement = containerElement;
    this.instructionImg = containerElement.querySelector("#instruction-image");
    this.previousInstructionButton = containerElement.querySelector("#previous-instruction-button");
    this.nextInstructionButton = containerElement.querySelector("#next-instruction-button");
    this.startOrResumeTaskButton = containerElement.querySelector("#start-or-resume-task-button");
    this.progressLabel = containerElement.querySelector("#instruction-progress-label");
    this.previousInstructionButton.textContent = lang.previousInstructionButtonText;
    this.nextInstructionButton.textContent = lang.nextInstructionButtonText;
    this.studyName = studyName;
  }

  set hidden(value) {
    if (value != this.containerElement.hidden) {
      this.containerElement.hidden = value;
    }
  }

  showInstructionIndex(idx, nInstructions, taskIdx = null) {
    const eventValue = (taskIdx !== null) ? [idx, taskIdx] : idx;
    studyRunData.recordStudyRunEvent(window.performance.now(),
      "showTaskInstructionIndex", eventValue);
    this.showImageForInstructionIndex(idx, taskIdx);
    this.previousInstructionButton.disabled = (idx == 0);
    this.nextInstructionButton.disabled = (idx == (nInstructions-1));
    this.progressLabel.textContent = this.progressLabelTextForIndex(idx, nInstructions);
  }

  showImageForInstructionIndex(idx, taskIdx = null) {
    const src = lang.instructionImgSrc(this.studyName, idx, taskIdx);
    this.instructionImg.src = src;
  }

  progressLabelTextForIndex(idx, nInstructions) {
    return `${idx+1}/${nInstructions}`;
  }

  getInnerHTMLString() {
    return (
      `<img id="instruction-image" src="" />
       <div id="instruction-image-to-navbutton-interspace"></div>
       <div>
         <button id="previous-instruction-button" class="instruction-nav-button" style="" disabled></button>
         <span id="instruction-progress-label"></span>
         <button id="next-instruction-button" class="instruction-nav-button"></button>
       </div>
       <div>
        <button id="start-or-resume-task-button" class="instruction-nav-button" hidden></button>
       </div>`);
  }
}

// Display for asking for subject's informed consent (DOM-based)
class InformedConsentDisplay {
  constructor(studyParams) {
    let domElement = document.createElement('div');
    domElement.id = "consent-container";
    domElement.innerHTML = this.getInnerHTMLString(studyParams);
    this.domElement = domElement;
  }

  getInnerHTMLString(studyParams) {
    return lang.getConsentInnerHTMLString(studyParams);
  }

}

// Display for the google feedback form shown at the end of the task (DOM-based)
class FeedbackForm {
  constructor(src, yTop) {
    this.src = src;
    let domElement = document.createElement('div');
    let iframeHTML = `<iframe src="${src}" width="${FEEDBACK_FORM_IFRAME_WIDTH}" height="${FEEDBACK_FORM_IFRAME_HEIGHT}" frameborder="0" marginheight="0" marginwidth="0"></iframe>`;
    domElement.id = "feedback-form-container";
    domElement.innerHTML = iframeHTML;
    domElement.style.position = "absolute";
    domElement.style.overflow = "auto";
    domElement.style.transform = "translate(-50%, 0%)";
    domElement.style.left = "50%";
    domElement.style.top = `${yTop}px`;
    this.domElement = domElement;
  }

  set yTop(value) {
    this.domElement.style.top = `${value}px`;
  }

  set hidden(value) {
    this.domElement.hidden = value;
  }
}

