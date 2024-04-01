// Script to check that minimum requirements for participation in the study are met

const REQUIREMENTS_MIN_SCREEN_WIDTH = 800; // in css pixels
const REQUIREMENTS_MIN_SCREEN_HEIGHT = 768; // in css pixels

const RETURN_PROLIFIC_SUBMISSION_MESSAGE = "return your submission on Prolific by clicking the 'stop without completing' button";

class RequirementsCheckResults {
  constructor(passed, failureMessage = null) {
    this.passed = passed;
    this.failureMessage = failureMessage;    
  }
}

function checkNotMobileRequirement(simulateFailure = false) {
  let userAgentString = navigator.userAgent;
  const isMobileDevice = userAgentString.indexOf("Mobi") > -1;
  if (!isMobileDevice && !simulateFailure) {
    // Pass
    return new RequirementsCheckResults(true);
  } else {
    // Fail
    let message = `It seems you are using a mobile device. \
      To participate in this study, you need to use a laptop or a desktop computer. \
      Please come back with such a device or ${RETURN_PROLIFIC_SUBMISSION_MESSAGE}.`;
    return new RequirementsCheckResults(false, message);
  }
}

function checkPointerRequirement(simulateFailure = false) {
  const isPointerRequirementMet = window.matchMedia("(pointer:fine)").matches;
  if (isPointerRequirementMet && !simulateFailure) {
    // Pass
    return new RequirementsCheckResults(true);
  } else {
    // Fail
    let message = `To participate in this study, you need a computer with \
      a pointing device such as a touchpad or a mouse. \
      It seems you do not have one at the moment. \
      Please come back with such a device or ${RETURN_PROLIFIC_SUBMISSION_MESSAGE}.`;
    return new RequirementsCheckResults(false, message);
  }
}

function checkScreenSizeRequirement(simulateFailure = false) {
  if ((window.screen.width >= REQUIREMENTS_MIN_SCREEN_WIDTH)
    && (window.screen.height >= REQUIREMENTS_MIN_SCREEN_HEIGHT)
    && !simulateFailure) {
    // Pass
    return new RequirementsCheckResults(true);
  } else {
    // Fail
    let message = `To participate in this study, you need a screen resolution of at least \
      ${REQUIREMENTS_MIN_SCREEN_WIDTH}px wide and ${REQUIREMENTS_MIN_SCREEN_HEIGHT} high. \
      It seems your screen is ${window.screen.width}px wide and ${window.screen.height}px high. \
      Please come back with a larger screen or ${RETURN_PROLIFIC_SUBMISSION_MESSAGE}.`;
    return new RequirementsCheckResults(false, message);
  }
}

function checkBrowserRequirement(simulateFailure = false) {
  let userAgentString = navigator.userAgent;

  // Ref: https://developer.mozilla.org/en-US/docs/Web/HTTP/Browser_detection_using_the_user_agent#making_the_best_of_user_agent_sniffing
  const isSeamonkey = userAgentString.indexOf("Seamonkey") > -1;
  const isFirefox = userAgentString.indexOf("Firefox") > -1 && !isSeamonkey;
  const isEdge = userAgentString.indexOf("Edg") > -1;
  const isChromium = userAgentString.indexOf("Chromium") > -1;
  const isChrome = (userAgentString.indexOf("Chrome") > -1) && !isChromium && !isEdge;
  const isSafari = (userAgentString.indexOf("Safari") > -1) && !isChrome && !isChromium && !isEdge;
  const isOpera = (userAgentString.indexOf("OPR") > -1) || (userAgentString.indexOf("Opera") > -1);
  const isInternetExplorer = (userAgentString.indexOf("Trident") > -1) || (userAgentString.indexOf("MSIE") > -1);

  if ((isChrome || isFirefox) && !simulateFailure) {
    // Pass
    return new RequirementsCheckResults(true);
  } else {
    // Fail
    let browserName = null;
    if (isSeamonkey) {
      browserName = "SeaMonkey";
    } else if (isEdge) {
      browserName = "Microsoft Edge";
    } else if (isChromium) {
      browserName = "Chromium";
    } else if (isChrome) {
      browserName = "Chrome";
    } else if (isSafari) {
      browserName = "Safari";
    } else if (isOpera) {
      browserName = "Opera";
    } else if (isInternetExplorer) {
      browserName = "Internet Explorer";
    }
    let message = browserName ? `It seems you are using ${browserName}. ` : '';
    message += `To participate in this study, you need to use Google Chrome \
      or another browser in the Chrome family, or Firefox. \
      Please come back using one of these browsers, or ${RETURN_PROLIFIC_SUBMISSION_MESSAGE}.`;
    return new RequirementsCheckResults(false, message);
  }
}

// Check that all requirements are met.
// If this is the case, requirementsCheckResults.passed will be true.
// If not, the evluation will stop at the first check that failed
// requirementsCheckResults.passed will be false
// and requirementsCheckResults.failureMessage will explain to the user
// what requirement is not met and how to meet the requirement if they can.
let requirementsCheckResults;
requirementsCheckResults = checkNotMobileRequirement();
if (requirementsCheckResults.passed) {
  requirementsCheckResults = checkPointerRequirement();
}
if (requirementsCheckResults.passed) {
  requirementsCheckResults = checkScreenSizeRequirement();
}
if (requirementsCheckResults.passed) {
  requirementsCheckResults = checkBrowserRequirement();
}

// If the requirements failed, replace the whole HTML document body with the failure message 
if (!requirementsCheckResults.passed) {
  document.body.innerHTML = `<div id='requirements-check-failure-message'><p>${requirementsCheckResults.failureMessage}</p</div>`;
}

