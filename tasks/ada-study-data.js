const ADA_PROB_TASK_NAME_DASH_CASE = "ada-prob";
const ADA_POS_TASK_NAME_DASH_CASE = "ada-pos";

class AdaStudyRunData {
  constructor(studyName, nTasks = 1) {
    this.studyName = studyName;
    this.studyRunEvents = {};
    this.nTasks = nTasks;
    if (this.nTasks > 1) {
      this.taskSessionDataArrays = new Array(nTasks)
      for (let i = 0; i < this.taskSessionDataArrays.length; i++) {
        this.taskSessionDataArrays[i] = [];
      }
    } else {
      this.taskSessionDataArray = [];
    }
  }

  recordTaskSessionData(data, taskIndex = null) {
    if (this.nTasks > 1) {
      this.taskSessionDataArrays[taskIndex].push(data);
    } else {
      this.taskSessionDataArray.push(data);
    }
  }

  recordStudyRunEvent(time, type, value) {
    if (!(type in this.studyRunEvents)) {
      // initialize empty array for this new event type
      this.studyRunEvents[type] = new Array();
    }
    this.studyRunEvents[type].push([time, value]);
  }

  toCSVString() {
    const EOL = '\r\n';
    const rows = new Array(2);
    rows[0] = "studyRunData";
    rows[1] = _csvStringifyQuoteIfRequired(this.toJSONString());
    return rows.join(EOL);
  }

  toJSONString() {
    return JSON.stringify(this, null, 0);
  }

  get jsonFilename() {
    return getStudyRunDataFilename(
      this.studyName,
      this._getValueForFirstTaskSession("subjectId"),
      this._getValueForFirstTaskSession("runId"),
      this._getValueForFirstTaskSession("dayCreated"),
      null,
      'json')
  }

  get csvFilename() {
    return getStudyRunDataFilename(
      this.studyName,
      this._getValueForFirstTaskSession("subjectId"),
      this._getValueForFirstTaskSession("runId"),
      this._getValueForFirstTaskSession("dayCreated"),
      null,
      'csv')
  }

  _getValueForFirstTaskSession(key) {
    if (this.nTasks > 1) {
      return this.taskSessionDataArrays[0][0]?.[key];
    } else {
      return this.taskSessionDataArray[0]?.[key];
    }
  }
}

class AdaTaskSessionData {
  constructor(taskName,
    subjectId,
    runId,
    taskSessionType,
    taskSessionIdx,
    nTrials,
    outcome,
    ) {
    this.dateCreated = nowDateString(true);
    this.taskName = taskName;
    this.subjectId = subjectId;
    this.runId = runId;
    this.taskSessionType = taskSessionType;
    this.taskSessionIdx = taskSessionIdx;
    this.nTrials = nTrials;
    this.trialIdx = range(nTrials);
    this.outcome = outcome;
    this.estimate = new Array(nTrials);
    this.scoreFrac = null;
    this.events = {};
  }

  get trialDataKeys() {
    return [
      "taskName",
      "subjectId",
      "runId",
      "dateCreated",
      "taskSessionType",
      "taskSessionIdx",
      "nTrials",
      "trialIdx",
      "outcome",
      "estimate",
      "scoreFrac"
      ];
  }

  get trialDataArraysByKey() {
    const trialDataKeys = this.trialDataKeys;
    const nTrials = this.nTrials;
    let ret = {};
    for (let key of trialDataKeys) {
      if (Array.isArray(this[key])) {
        ret[key] = this[key];
      } else {
        ret[key] = new Array(nTrials);
        for (let i = 0; i < nTrials; i++) {
          ret[key][i] = this[key]; // copy same value for all rows
        }
      }
    }
    return ret;
  }

  get trialDataTable2DArray() {
    const trialDataKeys = this.trialDataKeys;
    const nTrials = this.nTrials;
    const nColumns = trialDataKeys.length;
    const trialDataArraysByKey = this.trialDataArraysByKey;
    let rows = new Array(nTrials+1); // +1 for header row
    rows[0] = trialDataKeys;
    for (let i = 0; i < nTrials; i++) {
      let row = new Array(nColumns);
      for (let j = 0; j < nColumns; j++) {
        row[j] = trialDataArraysByKey[trialDataKeys[j]][i];
      }
      rows[i+1] = row;
    }
    return rows
  }

  get trialDataCSVString() {
    return csvStringify(this.trialDataTable2DArray);
  }

  get eventDataTable2DArray() {
    const eventTypes = Object.keys(this.events);
    let rows = new Array();
    rows.push(["eventType", "eventTime", "eventValue"]);
    for (let type of eventTypes) {
      let timeAndValues = this.events[type];
      for (let timeAndValue of timeAndValues) {
        rows.push([type, timeAndValue[0], timeAndValue[1]]);
      }
    }
    return rows;
  }

  get eventDataCSVString() {
    return csvStringify(this.eventDataTable2DArray);
  }

  recordEvent(time, type, value) {
    if (!(type in this.events)) {
      // initialize empty array for this new event type
      this.events[type] = new Array();
    }
    this.events[type].push([time, value]);
  }

  get dayCreated() {
    return this.dateCreated.substring(0, 10); // YYYY-MM-DD
  }

  get trialDataCSVFilename() {
    return getTaskSessionDataFilename(
      this.taskName,
      this.subjectId,
      this.runId,
      this.taskSessionType,
      this.taskSessionIdx,
      this.dayCreated,
      'trialData',
      'csv');
  }

  get eventDataCSVFilename() {
    return getTaskSessionDataFilename(
      this.taskName,
      this.subjectId,
      this.runId,
      this.taskSessionType,
      this.taskSessionIdx,
      this.dayCreated,
      'eventData',
      'csv');
  }

  get jsonFilename() {
    return getTaskSessionDataFilename(
      this.taskName,
      this.subjectId,
      this.runId,
      this.taskSessionType,
      this.taskSessionIdx,
      this.dayCreated,
      null,
      'json');
  }
}

class AdaProbTaskSessionData extends AdaTaskSessionData {
  constructor(
    subjectId,
    runId,
    taskSessionType,
    taskSessionIdx,
    nTrials,
    outcome,
    p1,
    didP1Change,
    idealObserverEstimate,
    sequenceDataFile,
    sequenceIdx,
    sequencePC,
    sequenceP1Min,
    sequenceP1Max,
    sequenceMinRunLength,
    sequenceMaxRunLength,
    sequenceMinOddChange,
    ) {
    super(ADA_PROB_TASK_NAME_DASH_CASE,
      subjectId,
      runId,
      taskSessionType,
      taskSessionIdx,
      nTrials,
      outcome);
    this.p1 = p1;
    this.didP1Change = didP1Change;
    this.idealObserverEstimate = idealObserverEstimate;
    this.sequenceDataFile = sequenceDataFile;
    this.sequenceIdx = sequenceIdx;
    this.sequencePC = sequencePC;
    this.sequenceP1Min = sequenceP1Min;
    this.sequenceP1Max = sequenceP1Max;
    this.sequenceMinRunLength = sequenceMinRunLength;
    this.sequenceMaxRunLength = sequenceMaxRunLength;
    this.sequenceMinOddChange = sequenceMinOddChange;
  }

  get trialDataKeys() {
    let keys = super.trialDataKeys;
    return keys.concat([
      "p1",
      "didP1Change",
      "idealObserverEstimate",
      "sequenceDataFile",
      "sequenceIdx",
      "sequencePC",
      "sequenceP1Min",
      "sequenceP1Max",
      "sequenceMinRunLength",
      "sequenceMaxRunLength",
      "sequenceMinOddChange"]);
  }
}


class AdaPosTaskSessionData extends AdaTaskSessionData {
  constructor(
    subjectId,
    runId,
    taskSessionType,
    taskSessionIdx,
    nTrials,
    outcome,
    mean,
    didMeanChange,
    sequenceDataFile,
    sequenceIdx,
    sequencePC,
    sequenceStd,
    sequenceMeanMin,
    sequenceMeanMax,
    sequenceMinRunLength,
    sequenceMaxRunLength,
    sequenceMinOddChange,
    ) {
    super(ADA_POS_TASK_NAME_DASH_CASE,
      subjectId,
      runId,
      taskSessionType,
      taskSessionIdx,
      nTrials,
      outcome);
    this.mean = mean;
    this.didMeanChange = didMeanChange;
    this.sequenceDataFile = sequenceDataFile;
    this.sequenceIdx = sequenceIdx;
    this.sequencePC = sequencePC;
    this.sequenceStd = sequenceStd;
    this.sequenceMeanMin = sequenceMeanMin;
    this.sequenceMeanMax = sequenceMeanMax;
    this.sequenceMinRunLength = sequenceMinRunLength;
    this.sequenceMaxRunLength = sequenceMaxRunLength;
    this.sequenceMinOddChange = sequenceMinOddChange;
  }

  get trialDataKeys() {
    let keys = super.trialDataKeys;
    return keys.concat([
      "mean",
      "didMeanChange",
      "sequenceDataFile",
      "sequenceIdx",
      "sequencePC",
      "sequenceStd",
      "sequenceMeanMin",
      "sequenceMeanMax",
      "sequenceMinRunLength",
      "sequenceMaxRunLength",
      "sequenceMinOddChange"]);
  }
}

function getStudyRunDataFilename(
  taskName,
  subjectId,
  runId,
  dateStr,
  suffix=null,
  fmt='json') {
  const fbasenameParts = [
    taskName,
    `subject-${subjectId}`,
    `run-${runId}`,
    dateStr];
  if (suffix) {
    fbasenameParts.push(suffix);
  }
  const fbasename = fbasenameParts.join("_");
  return `${fbasename}.${fmt}`;
}

function getTaskSessionDataFilename(
  taskName,
  subjectId,
  runId,
  taskSessionType,
  taskSessionIdx,
  dateStr,
  suffix=null,
  fmt='json') {
  const fbasenameParts = [
    taskName,
    `subject-${subjectId}`,
    `run-${runId}`,
    taskSessionType,
    `session-${taskSessionIdx}`,
    dateStr];
  if (suffix) {
    fbasenameParts.push(suffix);
  }
  const fbasename = fbasenameParts.join("_");
  return `${fbasename}.${fmt}`;
}

function promptToSaveStudyRunData(studyRunData, fmt='json') {
  if (fmt == 'json') {
    promptToSaveStr(studyRunData.toJSONString(), studyRunData.jsonFilename,
      'application/json');
  } else if (fmt == 'csv') {
    promptToSaveStr(studyRunData.toCSVString(), studyRunData.csvFilename,
      'text/csv');
  }
}

function promptToSaveTaskSessionData(sessionData, fmt='json') {
  if (fmt == 'json') {
    promptToSaveJson(sessionData, sessionData.jsonFilename, 0);
  } else if (fmt == 'csv') {
    promptToSaveStr(sessionData.trialDataCSVString, sessionData.trialDataCSVFilename, 'text/csv');
    promptToSaveStr(sessionData.eventDataCSVString, sessionData.eventDataCSVFilename, 'text/csv');
  }
}

function getSubjectId() {
  // Parse the subject ID from the URL parameters if it is provided.
  // If the subject is coming from Prolific, use Prolific's participant id.
  const DEFAULT_SUBJECT_ID = "default";
  let subjectId;
  if (isProlificSubject) {
    subjectId = prolificInfo.prolificPid;
  } else {
    subjectId = (urlSearchParams.has('subjectId') ?
      urlSearchParams.get('subjectId')
      : DEFAULT_SUBJECT_ID);
  }
  return subjectId;
}

let _studyRunId = null;
function getStudyRunId() {
  // Id for the current run of the study.
  // If the subject is coming from Prolific, use Prolific's session id.
  // Otherwise, generate one randomly when this function is first called.
  if (isProlificSubject) {
    _studyRunId = prolificInfo.sessionId;
  } else if (_studyRunId === null) {
    _studyRunId = randomInt(1e9);
  }
  return _studyRunId;
}
