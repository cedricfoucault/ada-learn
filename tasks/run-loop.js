//
// Run loop
//

const SOA = 1500; // ms
const ISI = 200; // ms
const TIME_BETWEEN_SESSION_START_AND_FIRST_TRIAL = 1000; // ms

const LOOP_STATE_READY = 1;
const LOOP_STATE_STARTING = 2;
const LOOP_STATE_RUNNING = 3;
const LOOP_STATE_PAUSED = 4;
const LOOP_STATE_STOPPING = 5;
const LOOP_STATE_STOPPED_READY_NEXT = 6;
const LOOP_STATE_STOPPED_FINAL = 7;

class RunLoopManager {
  constructor(firstFrameCallback, updateTimeFrameCallback, stopCallback,
    canRunNextCallback) {
    this.firstFrameCallback = firstFrameCallback;
    this.updateTimeFrameCallback = updateTimeFrameCallback;
    this.stopCallback = stopCallback;
    this.canRunNextCallback = canRunNextCallback;
    this.state = LOOP_STATE_READY;
  }

  get shouldRequestNextTimeFrame() {
    return (this.state == LOOP_STATE_RUNNING);
  }

  get canStart() {
    return ((this.state == LOOP_STATE_READY)
      || (this.state == LOOP_STATE_STOPPED_READY_NEXT));
  }

  get isRunning() {
    return (this.state == LOOP_STATE_RUNNING);
  }

  get willUpdateFrame() {
    return (this.state == LOOP_STATE_STARTING
      || this.state == LOOP_STATE_RUNNING);
  }

  get hasBeenStopped() {
    return ((this.state == LOOP_STATE_STOPPED_READY_NEXT)
      || (this.state == LOOP_STATE_STOPPED_FINAL));
  }

  start() {
    if (this.canStart) {
      this.state = LOOP_STATE_STARTING;
      this.requestNextTimeFrame();
    }
  }

  stop() {
    this.state = LOOP_STATE_STOPPING;
    this.stopCallback?.();
    this.state = (this.canRunNextCallback?.() ?
      LOOP_STATE_STOPPED_READY_NEXT
      : LOOP_STATE_STOPPED_FINAL);
  }

  requestNextTimeFrame() {
    window.requestAnimationFrame((time) => { this.updateTimeFrame(time) });
  }

  updateTimeFrame(time) {
    if (this.hasBeenStopped) {
      return;
    }

    if (this.state == LOOP_STATE_STARTING) {
      this.state = LOOP_STATE_RUNNING;
      this.firstFrameCallback?.(time);
    }

    this.updateTimeFrameCallback?.(time);
    
    if (this.shouldRequestNextTimeFrame) {
      this.requestNextTimeFrame();
    }
  }
}

class TaskSessionRunLoop {
  constructor(drawIfNeededCallback,
    setOutcome,
    setStimulusVisible,
    setChangeLabelVisible,
    setEstimateHighlighted,
    getEstimate,
    stopCallback,
    canRunNextCallback,
    showChanges = false,
    trialHasChangeCallback = null) {
    this.drawIfNeededCallback = drawIfNeededCallback;
    this.setOutcome = setOutcome;
    this.setStimulusVisible = setStimulusVisible;
    this.setChangeLabelVisible = setChangeLabelVisible;
    this.setEstimateHighlighted = setEstimateHighlighted;
    this.getEstimate = getEstimate;
    this.showChanges = showChanges;
    this.trialHasChangeCallback = trialHasChangeCallback;
    this.runLoop = new RunLoopManager((time) => { this.firstFrameCallback(time) },
      (time) => { this.updateTimeFrameCallback(time) },
      stopCallback,
      canRunNextCallback);
  }

  get canStart() {
    return this.runLoop.canStart;
  }

  get isRunning() {
    return this.runLoop.isRunning;
  }

  get hasBeenStopped() {
    return this.runLoop.hasBeenStopped;
  }

  start(sessionData) {
    this.sessionData = sessionData;
    this.sessionData.trialFrameTime = new Array(sessionData.nTrials);
    this.runLoop.start();
  }

  stop() {
    this.runLoop.stop();
  }

  firstFrameCallback(time) {
    this.sessionData.currentTrialIndex = -1;
    this.sessionData.startTime = time;
    this.sessionData.recordEvent?.(time, "sessionStart", null);
  }

  updateTimeFrameCallback(tFrame) {
    const sessionData = this.sessionData;
    // Case 0: Time frames at the start of the session before the first trial
    if (tFrame < (sessionData.startTime + TIME_BETWEEN_SESSION_START_AND_FIRST_TRIAL)) {
      return;
    }

    // Case 1: First time frame of the first trial
    if (sessionData.currentTrialIndex == -1) {
      this.startFirstTrial(tFrame);
      return;
    }

    // Case 2: Any time frame after that
    const trialHasVisibleChange = (this.showChanges
      && this.trialHasChangeCallback?.(sessionData.currentTrialIndex));
    const trialDuration = trialHasVisibleChange ? 2 * SOA : SOA;

    const trialFrameTime = sessionData.trialFrameTime[sessionData.currentTrialIndex];
    const tSinceTrialFrameTime = tFrame - trialFrameTime;
    const isNextTrialTime = tSinceTrialFrameTime >= trialDuration;
    const isInterTrialTime = !isNextTrialTime && (tSinceTrialFrameTime >= (trialDuration - ISI));

    if (isNextTrialTime) {
      if (sessionData.currentTrialIndex < sessionData.nTrials - 1) {
        this.moveToNextTrial(tFrame);
      } else {
        this.endSession(tFrame);
      }
    }

    if (trialHasVisibleChange) {
      const changeLabelOnsetFromTrialStart = 0;
      const changeLabelOffsetFromTrialStart = SOA - ISI;
      const shouldShowChangeLabel = (
        (tSinceTrialFrameTime >= changeLabelOnsetFromTrialStart)
        && (tSinceTrialFrameTime < changeLabelOffsetFromTrialStart));
      this.setChangeLabelVisible?.(shouldShowChangeLabel);
    }

    const stimulusOnsetFromTrialStart = trialHasVisibleChange ? SOA : 0;
    const stimulusOffsetFromTrialStart = trialDuration - ISI;
    const shouldBeShown = (
        (tSinceTrialFrameTime >= stimulusOnsetFromTrialStart)
        && (tSinceTrialFrameTime < stimulusOffsetFromTrialStart))
    this.setStimulusVisible?.(shouldBeShown);

    const shouldBeHighlighted = isInterTrialTime;
    this.setEstimateHighlighted?.(shouldBeHighlighted);

    this.drawIfNeededCallback();

    // if (DEBUG_IDEAL_OBSERVER) {
    //   const trialIndex = sessionData.currentTrialIndex;
    //   const isWithinTrialTime = tSinceTrialFrameTime <= (SOA - ISI);
    //   if (tSinceTrialFrameTime <= (SOA - ISI) && trialIndex >= 0 && trialIndex < sessionData.nTrials) {
    //     const ioEstimates = sessionData.idealObserverEstimate;
    //     const ioPrevEstimate = trialIndex > 0 ? ioEstimates[trialIndex-1] : 0.5;
    //     const ioNewEstimate = ioEstimates[trialIndex];
    //     const frac = Math.max(0, Math.min(1, tSinceTrialFrameTime / (SOA - ISI - 100)));
    //     const currentEstimate = (frac * ioNewEstimate + (1 - frac) * ioPrevEstimate);
    //     updateEstimate(currentEstimate);
    //     setNeedsRedraw();
    //   }
    // }
  }

  startFirstTrial(time) {
    const sessionData = this.sessionData;
    sessionData.currentTrialIndex = 0;
    sessionData.trialFrameTime[sessionData.currentTrialIndex] = time;
    this.sessionData.recordEvent?.(time, "trialStart", sessionData.currentTrialIndex);
    this.updateOutcomeForCurrentTrial();
  }

  moveToNextTrial(time) {
    const sessionData = this.sessionData;
    this.recordEstimateForCurrentTrial();
    // increment trial index 
    sessionData.currentTrialIndex += 1;
    sessionData.trialFrameTime[sessionData.currentTrialIndex] = time;
    this.sessionData.recordEvent?.(time, "trialStart", sessionData.currentTrialIndex);
    this.updateOutcomeForCurrentTrial();
  }

  endSession(time) {
    const sessionData = this.sessionData;
    sessionData.endTime = time;
    this.sessionData.recordEvent?.(time, "sessionEnd", null);
    this.recordEstimateForCurrentTrial();
    this.stop()
  }

  recordEstimateForCurrentTrial() {
    const sessionData = this.sessionData;
    // record subject's estimate for the current trial
    sessionData.estimate[sessionData.currentTrialIndex] = this.getEstimate();
  }

  updateOutcomeForCurrentTrial() {
    // update outcome for current trial
    const sessionData = this.sessionData;
    this.setOutcome?.(sessionData.outcome[sessionData.currentTrialIndex]); 
  }
}

