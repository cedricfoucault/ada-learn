//
// Utility functions
//

SOFTPLUS_DEFAULT_SHARPNESS = 5;

function getFractionalValue(v, vA, vB,
  vAFrac = 0, vBFrac = 1, clipped = true,
  do_softplus = true, softplus_sharpness = SOFTPLUS_DEFAULT_SHARPNESS) {
  let slope = (vB - vA) / (vBFrac - vAFrac);
  let intercept = vA - slope * vAFrac;
  let vFrac = (v - intercept) / slope;
  if (do_softplus) {
    vFrac = softplus(vFrac, softplus_sharpness);
  }
  if (clipped) {
    return Math.max(0, Math.min(1, vFrac)); // clip to [0, 1]
  } else {
    return vFrac;
  }
}

function meanAbsoluteError(array1, array2) {
  let sumAbsoluteErrors = 0;
  array1.forEach((v1, i) => sumAbsoluteErrors += Math.abs(v1 - array2[i]));
  return sumAbsoluteErrors / array1.length;
}

function meanSquaredError(array1, array2) {
  let sumSquaredErrors = 0;
  array1.forEach((v1, i) => sumSquaredErrors += (v1 - array2[i]) ** 2);
  return sumSquaredErrors / array1.length;
}

function softplus(x, sharpness = SOFTPLUS_DEFAULT_SHARPNESS) {
  // Softplus function (Dugas et al., 2001)
  // Soft version of a hard threshold function at 0.
  // Allows to keep values always strictly positive.
  // When x < 0, the function takes very small but non-null positive values.
  // When x > 0, the function quickly converges to the identity function.
  return (1 / sharpness) * Math.log(1 + Math.exp(sharpness * x));
}

function nowDateString(includeTimeOfDay = false) {
  let length = 10; // YYYY-MM-DD
  if (includeTimeOfDay) {
    length += 9; // YYYY-MM-DDTHH:mm:ss
  } 
  return new Date().toISOString().substring(0, length);
}

// ref https://stackoverflow.com/questions/951021/what-is-the-javascript-version-of-sleep
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}


// inspired from: https://github.com/Inist-CNRS/node-csv-string/blob/master/src/CSV.ts
function csvStringify(input, sep = ',') {
  const EOL = '\r\n';
  let ret;
  if (Array.isArray(input)) {
    if (input.length === 0) {
      ret = EOL;
    } else if (!Array.isArray(input[0])) {
      for (let loop = 0; loop < input.length; loop++) {
        ret = _csvStringifyReducer(input[loop], ret, sep, loop > 0);
      }
      ret += EOL;
    } else if (Array.isArray(input[0])) {
      ret = input.map((item) => csvStringify(item, sep)).join('');
    }
  } else if (typeof input == 'object') {
    for (const key in input) {
      if (input.hasOwnProperty(key)) {
        ret = _csvStringifyReducer(input[key], ret, sep);
      }
    }
    ret += EOL;
  } else {
    ret = _csvStringifyReducer(input, ret, sep) + EOL;
  }
  return ret;
};

function _csvStringifyReducer(item, memo, sep = ',', prependSep = false) {
  item = _csvStringifySingleValue(item);
  return (
    (memo !== undefined || prependSep ? `${memo}${sep}` : '') +
    _csvStringifyQuoteIfRequired(item, sep)
  );
};

function _csvStringifySingleValue(item) {
  if (item === 0) {
    item = '0';
  } else if (item === undefined || item === null) {
    item = '';
  }
  if (typeof item != 'string') {
    const s = item.toString();
    if (s == '[object Object]') {
      item = JSON.stringify(item);
      if (item == '{}') {
        item = '';
      }
    } else {
      item = s;
    }
  }
  return item;
};

function _csvStringifyQuoteIfRequired(value, sep) {
  const quoteCharRegex = new RegExp('"', 'g');
  const specialCharRegex = new RegExp('["\r\n]', 'g');
  let shouldBeQuoted = value.search(specialCharRegex) >= 0 || value.includes(sep);
  return (shouldBeQuoted ?
    '"' + value.replace(quoteCharRegex, '""') + '"'
    : value);
}

function promptToSaveJson(obj, fname, space = 2) {
  let str = JSON.stringify(obj, null, space);
  promptToSaveStr(str, fname, 'application/json');
}

function promptToSaveStr(str, fname, type = 'text') {
  let blob = new Blob( [ str ], { type });
  
  let url = URL.createObjectURL( blob );
  let link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', fname);
  let event = document.createEvent( 'MouseEvents' );
  event.initMouseEvent('click', true, true, window, 1, 0, 0, 0, 0, false, false, false, false, 0, null);
  link.dispatchEvent(event);
}


function drawText(ctx, text, x, y, fontSize, color,
  textAlign = "center",
  textBaseline = "alphabetic",
  bold = false,
  font = FONT) {
  ctx.font = (bold ? "bold " : "") + String(fontSize) + "px " + font;
  ctx.textAlign = textAlign;
  ctx.textBaseline = textBaseline;
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

function measureText(ctx, text, fontSize,
  textAlign = "center",
  textBaseline = "alphabetic",
  bold = false,
  font = FONT) {
  ctx.font = getFont(fontSize, bold, font);
  ctx.textAlign = textAlign;
  ctx.textBaseline = textBaseline;
  return ctx.measureText(text);
}

function getFont(fontSize, bold = false, font = FONT) {
  return (bold ? "bold " : "") + String(fontSize) + "px " + font;
}

// Calculate height for the text displayed on a single line
function getTextHeight(ctx, text, fontSize) {
  const previousTextBaseline = ctx.textBaseline;
  const previousFont = ctx.font;

  ctx.textBaseline = 'bottom';
  ctx.font = getFont(fontSize, bold, font);
  const textMetrics = ctx.measureText(text);
  const height = textMetrics.actualBoundingBoxAscent;

  ctx.textBaseline = previousTextBaseline;
  ctx.font = previousFont;

  return height;
}

// source: https://stackoverflow.com/questions/1255512/how-to-draw-a-rounded-rectangle-using-html-canvas
function drawRoundedRect(
  ctx,
  x,
  y,
  width,
  height,
  radius,
  fill = true,
  stroke = true
) {
  if (typeof radius === 'number') {
    radius = {tl: radius, tr: radius, br: radius, bl: radius};
  } else {
    radius = {...{tl: 0, tr: 0, br: 0, bl: 0}, ...radius};
  }
  ctx.beginPath();
  ctx.moveTo(x + radius.tl, y);
  ctx.lineTo(x + width - radius.tr, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius.tr);
  ctx.lineTo(x + width, y + height - radius.br);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius.br, y + height);
  ctx.lineTo(x + radius.bl, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius.bl);
  ctx.lineTo(x, y + radius.tl);
  ctx.quadraticCurveTo(x, y, x + radius.tl, y);
  ctx.closePath();
  if (fill) {
    ctx.fill();
  }
  if (stroke) {
    ctx.stroke();
  }
}

// adapted from ref: https://github.com/jashkenas/underscore/blob/ffabcd443fd784e4bc743fff1d25456f7282d531/underscore.js
function randomSample(array, n) {
  if (n == null) {
    return array[randomInt(array.length - 1)];
  }
  let sample = array.slice();
  let length = sample.length;
  n = Math.max(Math.min(n, length), 0);
  let last = length - 1;
  for (let index = 0; index < n; index++) {
    let rand = randomInt(index, last);
    let temp = sample[index];
    sample[index] = sample[rand];
    sample[rand] = temp;
  }
  return sample.slice(0, n);
}

function randomInt(min, max) {
  if (max == null) {
    max = min;
    min = 0;
  }
  return min + Math.floor(Math.random() * (max - min + 1));
}

// from https://github.com/jashkenas/underscore/blob/ffabcd443fd784e4bc743fff1d25456f7282d531/underscore.js
function range(start, stop, step) {
  if (stop == null) {
    stop = start || 0;
    start = 0;
  }
  if (!step) {
    step = stop < start ? -1 : 1;
  }

  let length = Math.max(Math.ceil((stop - start) / step), 0);
  let range = Array(length);

  for (var idx = 0; idx < length; idx++, start += step) {
    range[idx] = start;
  }

  return range;
}

function smootherStep(t) {
  // ref https://en.wikipedia.org/wiki/Smoothstep
  let ts = t * t;
  let tc = ts * t;
  return Math.max(0, Math.min(1, 6*tc*ts - 15*ts*ts + 10*tc));
}

// ref https://easings.net/#easeOutQuint
function easeOutQuint(t) {
  return 1 - Math.pow(1 - t, 5);
}