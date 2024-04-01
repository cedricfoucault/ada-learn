const EXPERIMENT_STEP_TYPE_INSTRUCTION = 1;
const EXPERIMENT_STEP_TYPE_TRAINING_WITHOUT_CHANGE = 2;
const EXPERIMENT_STEP_TYPE_TRAINING_WITH_LABELED_CHANGES = 3;
const EXPERIMENT_STEP_TYPE_TRAINING_WITH_HIDDEN_CHANGES = 4;
const EXPERIMENT_STEP_TYPE_MAIN_TASK = 5;
const EXPERIMENT_STEP_TYPE_REVIEW_INSTRUCTION = 6;
const EXPERIMENT_STEP_TYPE_INFORMED_CONSENT = 7;

class ExperimentStep {
  constructor(type, index, taskIdx = null) {
    this.type = type;
    this.index = index;
    this.taskIdx = taskIdx;
  }

  get isInstruction() {
    return (this.type == EXPERIMENT_STEP_TYPE_INSTRUCTION);
  }

  get isMainTask() {
    return this.type == EXPERIMENT_STEP_TYPE_MAIN_TASK;
  }

  get isReviewInstruction() {
    return (this.type == EXPERIMENT_STEP_TYPE_REVIEW_INSTRUCTION);
  }

  get isInformedConsent() {
    return (this.type == EXPERIMENT_STEP_TYPE_INFORMED_CONSENT);
  }

  get isTaskSession() {
    return !this.isInstruction;
  }

  get isTraining() {
    switch (this.type) {
      case EXPERIMENT_STEP_TYPE_TRAINING_WITHOUT_CHANGE:
      case EXPERIMENT_STEP_TYPE_TRAINING_WITH_HIDDEN_CHANGES:
      case EXPERIMENT_STEP_TYPE_TRAINING_WITH_LABELED_CHANGES: {
        return true;
      }
      default: {
        return false;
      }
    }
  }

  toString() {
    return `${this.typeString}-${this.index}`;
  }

  get typeString() {
    switch (this.type) {
      case EXPERIMENT_STEP_TYPE_MAIN_TASK: {
        return "main-task";
      }
      case EXPERIMENT_STEP_TYPE_TRAINING_WITHOUT_CHANGE: {
        return "training-without-change";
      }
      case EXPERIMENT_STEP_TYPE_TRAINING_WITH_HIDDEN_CHANGES: {
        return "training-with-hidden-change";
      }
      case EXPERIMENT_STEP_TYPE_TRAINING_WITH_LABELED_CHANGES: {
        return "training-with-labeled-change";
      }
      case EXPERIMENT_STEP_TYPE_INSTRUCTION: {
        return "instruction";
      }
      case EXPERIMENT_STEP_TYPE_REVIEW_INSTRUCTION: {
        return "review-instruction";
      }
      case EXPERIMENT_STEP_TYPE_INFORMED_CONSENT: {
        return "informed-consent";
      }
    }
  }
}

class CurrentExperimentStep {
  constructor(step, delegate = null) {
    this._step = step;
    this.delegate = delegate;
  }

  get step() {
    return this._step;
  }

  get type() {
    return this._step.type;
  }

  get index() {
    return this._step.index;
  }

  get taskIdx() {
    return this._step.taskIdx;
  }

  get isInstruction() {
    return this._step.isInstruction;
  }

  get isMainTask() {
    return this._step.isMainTask;
  }

  get isReviewInstruction() {
    return this._step.isReviewInstruction;
  }

  get isInformedConsent() {
    return this._step.isInformedConsent;
  }

  moveToStep(step) {
    const previousStep = this._step;
    const nextStep = step;
    this.delegate?.willMoveFromStepToStep?.(previousStep, nextStep);

    this._step = step;

    this.delegate?.didMoveFromStepToStep?.(previousStep, nextStep);
  }
}
