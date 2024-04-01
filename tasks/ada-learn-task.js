"use strict";

//
// Define constants (known prior to execution or from the URL search parameters)
//

//
// Objects creation
//

let pavloviaManager = new PavloviaManager(saveIncompleteStudyRunData);

// Object in which to record all the data to be saved for this study run
// (to the server or locally on disk)
let studyRunData = new AdaStudyRunData(studyParams.studyName, studyParams.nTasks);
if (isProlificSubject) {
  studyRunData.prolificPid = prolificInfo.prolificPid;
  studyRunData.studyId = prolificInfo.studyId;
  studyRunData.sessionId = prolificInfo.sessionId;
}
studyRunData.recordStudyRunEvent(window.performance.now(),
  "arriveStudyHomepage", null);

// Subject id
let subjectId = getSubjectId();

// Id for the current run of the study.
let runId = getStudyRunId();

// Accumulator in which to record the subject's total accumulated money gains
let accumulatedGain = 0;

// Randomly select sequences for each session of this experiment from the ones
// available in the sequence data object
let selectedSequenceIndices = null;
let selectedSequenceIndicesByTask = null;
if (studyParams.nTasks > 1) {
  selectedSequenceIndicesByTask = new Array(studyParams.nTasks);
  for (let i = 0; i < studyParams.nTasks; i++) {
    const nSessions = studyParams.nSessionsByTask[i];
    const taskSequenceData = getSequenceData(i);
    selectedSequenceIndicesByTask[i] = randomSample(range(taskSequenceData.nSessions),
      nSessions);
  }
} else {
  const taskSequenceData = getSequenceData();
  selectedSequenceIndices = randomSample(range(taskSequenceData.nSessions),
    studyParams.nSessions);
}

function getSequenceData(taskIdx = null) {
  if (studyParams.nTasks > 1) {
    const taskName = studyParams.taskNames[taskIdx];
    return sequenceDataByTaskName[taskName];
  } else {
    return sequenceData;
  }
}

// Navigation and handling of the current experiment step

let initialExperimentStep;
const initialTaskIdx = studyParams.nTasks > 1 ? 0 : null;
if (studyParams.doGetSubjectConsent) {
  // Start with informed consent
  initialExperimentStep = new ExperimentStep(EXPERIMENT_STEP_TYPE_INFORMED_CONSENT, 0,
    initialTaskIdx);
} else if (studyParams.doInstructions) {
  // Start with task instructions
  initialExperimentStep = new ExperimentStep(EXPERIMENT_STEP_TYPE_INSTRUCTION, 0,
    initialTaskIdx);
} else {
  // Start with main task
  initialExperimentStep = new ExperimentStep(EXPERIMENT_STEP_TYPE_MAIN_TASK, 0,
    initialTaskIdx);
}
let currentExperimentStep = new CurrentExperimentStep(initialExperimentStep,
  window);

window.didMoveFromStepToStep = function(previousStep, nextStep) {
  if ((previousStep.taskIdx !== null) && (previousStep.taskIdx < nextStep.taskIdx)) {
    // We are moving on from one task to the next one.
    // Recreate the objects tied to the current task to reflect the change.
    createTaskSessionObjects(canvasWidthCss, canvasHeightCss, nextStep.taskIdx);
  }

  const nInstructions = ((studyParams.nTasks > 1) ?
    studyParams.nInstructionsByTask[nextStep.taskIdx]
    : studyParams.nInstructions);
  if (previousStep.isMainTask && nextStep.isMainTask) {
    const isFirstSession = nextStep.index == 0;
    startTaskSession(isFirstSession);
  } else if ((previousStep.isInstruction && nextStep.isInstruction)
    || (previousStep.isReviewInstruction && nextStep.isReviewInstruction)) {
    instructionDisplay.showInstructionIndex(nextStep.index, nInstructions, nextStep.taskIdx);
    updateStartOrResumeTaskButtonForExperimentStep(nextStep);
  } else if ((previousStep.isInstruction && nextStep.isMainTask)) {
    instructionDisplay.hidden = true;
    canvas.hidden = false;
    startTaskSession(true);
  } else if (previousStep.isMainTask && nextStep.isReviewInstruction) {
    instructionDisplay.showInstructionIndex(nextStep.index, nInstructions, nextStep.taskIdx);
    updateStartOrResumeTaskButtonForExperimentStep(nextStep);
    canvas.hidden = true;
    instructionDisplay.hidden = false;
  } else if (previousStep.isReviewInstruction && nextStep.isMainTask) {
    instructionDisplay.hidden = true;
    canvas.hidden = false;
  } else if (previousStep.isInformedConsent && nextStep.isInstruction) {
    instructionDisplay.showInstructionIndex(0, nInstructions, nextStep.taskIdx);
    updateStartOrResumeTaskButtonForExperimentStep(nextStep);
    if (consentElement) {
      consentElement.hidden = true;
    }
    instructionDisplay.hidden = false;
  } else if (previousStep.isInformedConsent && nextStep.isMainTask) {
    if (consentElement) {
      consentElement.hidden = true;
    }
    canvas.hidden = false;
    startTaskSession(true);
  } else if (previousStep.isMainTask && nextStep.isInstruction) {
    console.assert((previousStep.taskIdx < nextStep.taskIdx),
      "Moving from main task to initial instruction phase should only happen when moving to the next task");
    instructionDisplay.showInstructionIndex(nextStep.index, nInstructions, nextStep.taskIdx);
    updateStartOrResumeTaskButtonForExperimentStep(nextStep);
    canvas.hidden = true;
    instructionDisplay.hidden = false;
  } else {
    console.error(`forbidden move from step ${previousStep} to step ${nextStep}`);
  }

  setNeedsRedraw();
};

function startTaskSession(isFirstSession = true) {
  taskSessionDisplay.prompt.text = lang.promptStartFirstSession;
  taskSessionDisplay.estimate = 0.5;
  taskSessionDisplay.showSessionEnd = false;
  taskSessionDisplay.reviewInstructionsButton.enabled = false;
  taskSessionDisplay.nextTaskButton.enabled = false;
  if (isFirstSession) {
    taskSessionDisplay.mouseTarget.enabled = true;
    taskSessionDisplay.mouseTarget.clickAction = startTaskSessionRunLoop;
  } else {
    startTaskSessionRunLoop();
  }
  setNeedsRedraw();
}

function updateStartOrResumeTaskButtonForExperimentStep(step) {
  const nInstructions = (studyParams.nTasks > 1 ?
    studyParams.nInstructionsByTask[step.taskIdx]
    : studyParams.nInstructions);
  if (step.isInstruction && (step.index == nInstructions - 1)) {
    instructionDisplay.startOrResumeTaskButton.textContent = lang.startMainTaskButtonText;
    instructionDisplay.startOrResumeTaskButton.hidden = false;
  } else if (step.isReviewInstruction) {
    instructionDisplay.startOrResumeTaskButton.textContent = lang.resumeMainTaskButtonText;
    instructionDisplay.startOrResumeTaskButton.hidden = false;
  } else {
    instructionDisplay.startOrResumeTaskButton.hidden = true;
  }
}

// Informed consent element

let consentElement;
if (studyParams.doGetSubjectConsent) {
  let consentDisplay = new InformedConsentDisplay(studyParams);
  consentElement = consentDisplay.domElement;
  const acceptConsentButton = consentElement.querySelector("#accept-consent-button");
  acceptConsentButton.onclick = didClickAcceptConsentButton;
}

function didClickAcceptConsentButton() {
  studyRunData.recordStudyRunEvent(window.performance.now(),
        "acceptConsent", null);
  const taskIdx = studyParams.nTasks > 1 ? 0 : null;
  const nextStep = (studyParams.doInstructions ?
    new ExperimentStep(EXPERIMENT_STEP_TYPE_INSTRUCTION, 0, taskIdx)
    : new ExperimentStep(EXPERIMENT_STEP_TYPE_MAIN_TASK, 0, taskIdx));
  currentExperimentStep.moveToStep(nextStep);
}

// Instruction display

let instructionDisplay = new InstructionDisplay(studyParams.studyName);
instructionDisplay.nextInstructionButton.onclick = didClickNextInstructionButton;
instructionDisplay.previousInstructionButton.onclick = didClickPreviousInstructionButton;
instructionDisplay.startOrResumeTaskButton.onclick = didClickStartOrResumeTaskButton;
instructionDisplay.hidden = studyParams.doGetSubjectConsent || (!studyParams.doInstructions);
if (studyParams.doInstructions && (!studyParams.doGetSubjectConsent)) {
  const taskIdx = studyParams.nTasks > 1 ? 0 : null;
  const nInstructions = ((studyParams.nTasks > 1) ?
    studyParams.nInstructionsByTask[taskIdx]
    : studyParams.nInstructions);
  instructionDisplay.showInstructionIndex(0, nInstructions, taskIdx);
};

function didClickNextInstructionButton() {
  const nInstructions = ((studyParams.nTasks > 1) ?
    studyParams.nInstructionsByTask[currentExperimentStep.taskIdx]
    : studyParams.nInstructions);
  const nextInstructionIndex = (currentExperimentStep.index < (nInstructions - 1) ?
    currentExperimentStep.index + 1
    : currentExperimentStep.index);
  moveCurrentExperimentStepToIdx(nextInstructionIndex);
}

function didClickPreviousInstructionButton() {
  const previousInstructionIndex = (currentExperimentStep.index > 0 ?
    currentExperimentStep.index - 1
    : currentExperimentStep.index);
  moveCurrentExperimentStepToIdx(previousInstructionIndex);
}

function moveCurrentExperimentStepToIdx(idx) {
  let step = new ExperimentStep(currentExperimentStep.type, idx, currentExperimentStep.taskIdx);
  step.indexTaskSessionToResume = currentExperimentStep.step.indexTaskSessionToResume;
  currentExperimentStep.moveToStep(step);
}

function didClickStartOrResumeTaskButton() {
  if (currentExperimentStep.isInstruction) {
    const step = new ExperimentStep(EXPERIMENT_STEP_TYPE_MAIN_TASK, 0,
      currentExperimentStep.taskIdx);
    currentExperimentStep.moveToStep(step);
  } else if (currentExperimentStep.isReviewInstruction) {
    let taskSessionIdx = currentExperimentStep.step.indexTaskSessionToResume;
    if (taskSessionIdx === null || taskSessionDisplay === undefined) {
      console.error("undefined task session index to resume upon clicking the resume task button");
    }
    const step = new ExperimentStep(EXPERIMENT_STEP_TYPE_MAIN_TASK, taskSessionIdx,
      currentExperimentStep.taskIdx);
    currentExperimentStep.moveToStep(step);
  }
}

// Canvas, in which the task session display is drawn

let canvas = document.getElementById("canvas");
canvas.hidden = studyParams.doInstructions || studyParams.doGetSubjectConsent;

let canvasWidthCss;
let canvasHeightCss;
function resizeCanvasToWindow() {
  // resize the canvas to fill the browser's window
  canvasWidthCss = window.innerWidth;
  canvasHeightCss = window.innerHeight;
  // account for device DPI, drawing with higher resolution on higher DPI screens
  // see https://www.kirupa.com/canvas/canvas_high_dpi_retina.html
  let pxRatioPhysToCss = window.devicePixelRatio;
  canvas.width = canvasWidthCss * pxRatioPhysToCss;
  canvas.height = canvasHeightCss * pxRatioPhysToCss;
  canvas.style.width = canvasWidthCss + 'px';
  canvas.style.height = canvasHeightCss + 'px';
  canvas.getContext("2d").scale(pxRatioPhysToCss, pxRatioPhysToCss);
}

// Task session objects

let taskSessionDisplay;
let taskSessionRunLoop;
function createTaskSessionObjects(canvasWidth, canvasHeight,
  taskIdx = null) {
  const taskName = (studyParams.nTasks > 1 ?
    studyParams.taskNames[taskIdx] : studyParams.taskName);
  const ctx = canvas.getContext("2d");
  const canvasCenterX = Math.round(canvasWidth / 2);
  const canvasCenterY = Math.round(canvasHeight / 2);
  const mouseTargetClickAction = startTaskSessionRunLoop;
  taskSessionDisplay = new TaskSessionDisplay(ctx, canvasCenterX, canvasCenterY,
      SLIDER_LENGTH, taskName, taskIdx,
      mouseTargetClickAction,
      didClickCompleteProlificSubmissionButton,
      didClickDownloadDataFileButton);

  const setOutcome = (outcome) => {
    taskSessionDisplay.outcome = outcome;
  };
  const setStimulusVisible = (visible) => {
    taskSessionDisplay.dotStimulus.hidden = !visible;
  };
  const setEstimateHighlighted = (highlighted) => {
    taskSessionDisplay.estimateHighlighted = highlighted;
  };
  const getEstimate = () => {
    return taskSessionDisplay.estimate;
  };
  const canRunNextCallback = () => { return true };
  taskSessionRunLoop = new TaskSessionRunLoop(drawIfNeeded,
    setOutcome,
    setStimulusVisible,
    null,
    setEstimateHighlighted,
    getEstimate,
    taskSessionRunLoopStopCallback,
    canRunNextCallback);
}

function updateTaskSessionObjectsForCanvasSize(canvasWidth, canvasHeight) {
  let ctx = canvas.getContext("2d");
  let canvasCenterX = Math.round(canvasWidth / 2);
  let canvasCenterY = Math.round(canvasHeight / 2);
  taskSessionDisplay.setCenterCoordinates(canvasCenterX, canvasCenterY);
}

function startTaskSessionRunLoop() {
  pavloviaManager.initServerSessionIfNeeded();
  const currentStep = currentExperimentStep.step;
  let taskName = (studyParams.nTasks > 1 ?
    studyParams.taskNames[currentStep.taskIdx] : studyParams.taskName);
  if (taskName == ADA_PROB) {
    taskSessionDisplay.prompt.text = lang.promptEstimateProbability;
   } else if (taskName == ADA_POS) {
    taskSessionDisplay.prompt.text = lang.promptEstimatePosition;
   }
  const sessionData = newSessionDataForExperimentStep(currentStep);
  taskSessionRunLoop.start(sessionData);
  mouseTracker.startTrackingMouseMovement();
  setNeedsRedraw();
}

function newSessionDataForExperimentStep(step) {
  let taskName = (studyParams.nTasks > 1 ?
    studyParams.taskNames[step.taskIdx] : studyParams.taskName);
  let taskSequenceIndices = (studyParams.nTasks > 1 ?
    selectedSequenceIndicesByTask[step.taskIdx]
    : selectedSequenceIndices);
  let taskSequenceIdx;
  switch (step.type) {
    case EXPERIMENT_STEP_TYPE_MAIN_TASK: {
      taskSequenceIdx = taskSequenceIndices[step.index];
      break;
    }
    default: {
      console.assert(false, "error: trying to create a task session for an experiment step that is not a task session step");
      return;
    }
  }
  const taskSequenceData = getSequenceData(step.taskIdx);
  const nTrials = studyParams.debugShortenSequences ? 2 : taskSequenceData.nTrials;

  if (taskName == ADA_PROB) {
    return new AdaProbTaskSessionData(
      subjectId,
      runId,
      step.typeString,
      step.index,
      nTrials,
      taskSequenceData.outcome[taskSequenceIdx],
      taskSequenceData.p1[taskSequenceIdx],
      taskSequenceData.didP1Change[taskSequenceIdx],
      taskSequenceData.idealObserverEstimate[taskSequenceIdx],
      SEQUENCE_DATA_FILE,
      taskSequenceIdx,
      taskSequenceData["pC"],
      taskSequenceData["p1Min"],
      taskSequenceData["p1Max"],
      taskSequenceData["minRunLength"],
      taskSequenceData["maxRunLength"],
      taskSequenceData["minOddChange"]);
  } else if (taskName == ADA_POS) {
    return new AdaPosTaskSessionData(
      subjectId,
      runId,
      step.typeString,
      step.index,
      nTrials,
      taskSequenceData.outcome[taskSequenceIdx],
      taskSequenceData.mean[taskSequenceIdx],
      taskSequenceData.didMeanChange[taskSequenceIdx],
      SEQUENCE_DATA_FILE,
      taskSequenceIdx,
      taskSequenceData["pC"],
      taskSequenceData["std"],
      taskSequenceData["meanMin"],
      taskSequenceData["meanMax"],
      taskSequenceData["minRunLength"],
      taskSequenceData["maxRunLength"],
      taskSequenceData["minOddChange"]);  
  } else {
    console.assert(false, "unknown task name");
  }
}

function taskSessionRunLoopStopCallback() {
  const sessionData = taskSessionRunLoop.sessionData;
  const sessionIdx = currentExperimentStep.index;
  const taskIdx = currentExperimentStep.taskIdx;
  const nSessions = (studyParams.nTasks > 1 ?
    studyParams.nSessionsByTask[taskIdx]
    : studyParams.nSessions);
  const idxFirstSessionCountedInMoneyBonus = (studyParams.nTasks > 1 ?
    studyParams.idxFirstSessionCountedInMoneyBonusByTask[taskIdx]
    : studyParams.idxFirstSessionCountedInMoneyBonus);
  // Compute and register the subject's score for this session
  const scoreFrac = getScoreFracFromSessionData(sessionData);
  sessionData.scoreFrac = scoreFrac;
  // Compute and record the corresponding monetary gain 
  const sessionGain = (sessionIdx >= idxFirstSessionCountedInMoneyBonus ?
    gainFromScoreFrac(scoreFrac) : 0);
  accumulatedGain += sessionGain;
  // Record the task session data into the run data object
  studyRunData.recordTaskSessionData(sessionData, taskIdx);
  // Show session end screen
  updateScoreDisplay(scoreFrac, sessionGain, accumulatedGain, sessionIdx);
  updateTrueValuesDisplayForSessionData(sessionData);
  taskSessionDisplay.showSessionEnd = true;
  taskSessionDisplay.sessionEndTitleLabel.text = sessionEndTitleTextForSessionIndex(
    sessionIdx, nSessions); // TBD
  const wasLastSessionOfTask = (sessionIdx == (nSessions - 1));
  const wasLastSessionOfStudyRun = (studyParams.nTasks > 1 ?
    (wasLastSessionOfTask && (taskIdx == (studyParams.nTasks - 1)))
    : wasLastSessionOfTask);
  if (wasLastSessionOfStudyRun) {
    // this was the last session of the whole study run
    taskSessionDisplay.prompt.text = lang.promptSavingData;
    taskSessionDisplay.startUploadAnimation();
    taskSessionDisplay.mouseTarget.enabled = false;
    taskSessionDisplay.mouseTarget.hidden = true;
    taskSessionDisplay.mouseTarget.clickAction = startTaskSessionRunLoop;
    taskSessionDisplay.reviewInstructionsButton.hidden = true;
    taskSessionDisplay.reviewInstructionsButton.enabled = false;
    taskSessionDisplay.nextTaskButton.hidden = true;
    taskSessionDisplay.nextTaskButton.enabled = false;
    if (studyParams.doShowFeedbackForm) {
      taskSessionDisplay.showFeedbackForm();
    }
  } else if (wasLastSessionOfTask) {
    // this was the last session of the current task
    // and there is another task to perform next
    taskSessionDisplay.prompt.text = lang.promptTaskCompleteWithIndex(taskIdx);
    taskSessionDisplay.mouseTarget.enabled = false;
    taskSessionDisplay.mouseTarget.hidden = true;
    taskSessionDisplay.mouseTarget.clickAction = startTaskSessionRunLoop;
    taskSessionDisplay.reviewInstructionsButton.hidden = true;
    taskSessionDisplay.reviewInstructionsButton.enabled = false;
    const nextStepType = (studyParams.doInstructions ?
      EXPERIMENT_STEP_TYPE_INSTRUCTION : EXPERIMENT_STEP_TYPE_MAIN_TASK);
    const nextStep = new ExperimentStep(nextStepType, 0, taskIdx+1);
    taskSessionDisplay.nextTaskButton.clickAction = () => { 
      currentExperimentStep.moveToStep(nextStep); };
    taskSessionDisplay.nextTaskButton.hidden = false;
    taskSessionDisplay.nextTaskButton.enabled = true;
  } else {
    // there is another session to perform next for the same task
    taskSessionDisplay.prompt.text = lang.promptStartNextSession;
    const nextTaskSessionStep = new ExperimentStep(
      EXPERIMENT_STEP_TYPE_MAIN_TASK, sessionIdx+1, taskIdx);
    const reviewInstructionsStep = new ExperimentStep(
      EXPERIMENT_STEP_TYPE_REVIEW_INSTRUCTION, 0, taskIdx);
    reviewInstructionsStep.indexTaskSessionToResume = sessionIdx;
    taskSessionDisplay.mouseTarget.clickAction = () => { 
      currentExperimentStep.moveToStep(nextTaskSessionStep); };
    taskSessionDisplay.mouseTarget.enabled = true;
    taskSessionDisplay.mouseTarget.hidden = false;
    const reviewEventValue = (taskIdx !== null) ? [sessionIdx, taskIdx] : sessionIdx;
    taskSessionDisplay.reviewInstructionsButton.clickAction = () => {
      studyRunData.recordStudyRunEvent(window.performance.now(),
        "reviewTaskInstructionsAtSessionIdx", reviewEventValue);
      currentExperimentStep.moveToStep(reviewInstructionsStep);
    };
    taskSessionDisplay.reviewInstructionsButton.hidden = false;
    taskSessionDisplay.reviewInstructionsButton.enabled = true;
    taskSessionDisplay.nextTaskButton.hidden = true;
    taskSessionDisplay.nextTaskButton.enabled = false;
  }
  setNeedsRedraw();
  // stop tracking subject's estimates
  mouseTracker.stopTrackingMouseMovement();

  if (wasLastSessionOfStudyRun) {
    studyRunData.totalMoneyBonus = accumulatedGain;
    saveStudyRunDataThenCompleteTask();
  }
}

async function saveStudyRunDataThenCompleteTask() {
  studyRunData.recordStudyRunEvent(window.performance.now(),
    "startStudyDataUpload", null);
  let didSaveSucceed;
  try {
    await pavloviaManager.saveStudyRunData(studyRunData);
    // to simulate delay: saveStudyRunData(studyRunData, false, true);
    didSaveSucceed = true;
  } catch (error) {
    didSaveSucceed = false;
  }
  console.log("saveStudyRunData done");
  await pavloviaManager.finishServerSessionIfNeeded();
  console.log("finishServerSessionIfNeeded done");
  taskSessionDisplay.stopUploadAnimation();
  if (didSaveSucceed) {
    taskSessionDisplay.prompt.text = lang.promptStudyComplete;
    if (isProlificSubject) {
      taskSessionDisplay.showCompleteProlificSubmissionButton();
    }
  } else {
    if (isProlificSubject) {
      taskSessionDisplay.prompt.text = lang.promptStudyDataUploadFailedProlific;
      taskSessionDisplay.showDownloadDataFileButton();
    } else {
      taskSessionDisplay.prompt.text = lang.promptStudyComplete;
    }
  }
  setNeedsRedraw();
}

function saveIncompleteStudyRunData() {
  console.log("saving incomplete study run data");
  if (taskSessionRunLoop.isRunning && taskSessionRunLoop.sessionData) {
    // record task session data even though the session is incomplete
    studyRunData.recordTaskSessionData(taskSessionRunLoop.sessionData,
      currentExperimentStep.taskIdx);
  }
  studyRunData.recordStudyRunEvent(window.performance.now(),
    "startIncompleteStudyDataUpload", null);
  pavloviaManager.saveStudyRunData(studyRunData, true);
}

function sessionEndTitleTextForSessionIndex(sessionIndex, nSessions) {
  if (nSessions > 1) {
    return `${lang.sessionComplete} (${sessionIndex+1}/${nSessions})`;
  } else {
    return lang.sessionComplete;
  }
}

function updateTrueValuesDisplayForSessionData(sessionData) {
  const ctx = canvas.getContext("2d");
  const [trueValues, estimatesPerTrueValue] = getTrueValuesAndEstimatesPerRunLength(sessionData);
  taskSessionDisplay.setTrueValues(ctx, trueValues, estimatesPerTrueValue);
}

function getTrueValuesAndEstimatesPerRunLength(sessionData) {
  const isAdaProb = (sessionData.taskName == ADA_PROB_TASK_NAME_DASH_CASE);
  let trueValuePerRunLength = new Array();
  let estimatesPerRunLength = new Array();
  let runLengthIndex = -1;
  let trueValues = isAdaProb ? sessionData.p1 : sessionData.mean;
  let didTrueValueChange = isAdaProb ? sessionData.didP1Change : sessionData.didMeanChange;
  for (var i = 0; i < trueValues.length; i++) {
    if (didTrueValueChange[i]) {
      trueValuePerRunLength.push(trueValues[i]);
      estimatesPerRunLength.push(new Array());
      runLengthIndex += 1;
    }
    estimatesPerRunLength[runLengthIndex].push(sessionData.estimate[i]);
  }
  return [trueValuePerRunLength, estimatesPerRunLength];
}

function updateScoreDisplay(scoreFrac, gain, locAccumulatedGain, sessionIdx) {
  if (sessionIdx == 0) {
    gain = null;
    locAccumulatedGain = null;
  } else if (sessionIdx == 1) {
    locAccumulatedGain = null;
  }
  taskSessionDisplay.scoreDisplay.updateWithScoreFrac(scoreFrac, gain, locAccumulatedGain);
}

function gainFromScoreFrac(scoreFrac) {
  return scoreFrac * studyParams.maxMoneyBonusPerSession;
}

// Prolific end of study action buttons

function didClickCompleteProlificSubmissionButton () {
  // open link in a new window so that the subject
  // can still see the final task screen and provide input in the feedback form
  window.open(prolificInfo.completionURL);
}

function didClickDownloadDataFileButton() {
  promptToSaveStudyRunData(studyRunData, 'csv');
}

// 
// Event handlers
//

// Mouse events

let mouseTracker = new MouseTracker(
    updateEstimateForMouseMovement,
    currentMouseTargets);
mouseTracker.canvas = canvas;

function currentMouseTargets() {
  if (currentExperimentStep.isMainTask) {
    return taskSessionDisplay.mouseTargets;
  } else {
    return [];
  }
}

function updateEstimateForMouseMovement(movementX) {
  const pxPerFractionUnit = taskSessionDisplay.slider.length;
  const probaUpdate = (movementX / pxPerFractionUnit);
  const newValue = taskSessionDisplay.estimate + probaUpdate;
  updateEstimate(newValue);
  // record estimate updates within a trial
  let tNow = window.performance.now();
  taskSessionRunLoop.sessionData.recordEvent(tNow, "estimateUpdate", newValue);
  setNeedsRedraw();
}

function updateEstimate(newValue) {
  const min = taskSessionDisplay.slider.track.reachableFractionRange[0];
  const max = taskSessionDisplay.slider.track.reachableFractionRange[1];
  newValue = Math.min(Math.max(newValue, min), max);
  taskSessionDisplay.estimate = newValue;
}

//  Window resize events

window.addEventListener('resize', windowDidResize, false);
function windowDidResize() {
  resizeCanvasToWindow(canvas);
  updateTaskSessionObjectsForCanvasSize(canvasWidthCss, canvasHeightCss);
  setNeedsRedraw();
}

//
// Drawing
//

let needsRedraw;
function setNeedsRedraw() {
  needsRedraw = true;
  if (!taskSessionRunLoop?.willUpdateFrame) {
    // Execute drawIfNeeded on the next event cycle,
    // so that multiple calls to setNeedsRedraw within the current cycle
    // will result in only one execution draw() in the first execution of
    // drawIfNeeded() (the other calls being shortcuted due to needsRedraw being false).
    // This is more efficient than calling drawIfNeeded directly here,
    // which would result in multiple redundant draw() executions if setNeedsRedraw()
    // is called several times in one cycle.
    setTimeout(drawIfNeeded, 0);
  }
}

function drawIfNeeded() {
  if (needsRedraw) {
    draw();
  }
  needsRedraw = false;
}

function draw() {
  let ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (currentExperimentStep.isMainTask) {
    taskSessionDisplay.draw(ctx);
  }
}

//
// Initial setup
//

document.body.appendChild(instructionDisplay.containerElement);
if (studyParams.doGetSubjectConsent) {
  document.body.appendChild(consentElement);
}

resizeCanvasToWindow(canvas);
createTaskSessionObjects(canvasWidthCss, canvasHeightCss,
  (studyParams.nTasks > 1 ? 0 : null));
setNeedsRedraw();
