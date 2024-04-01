class PavloviaManager {
  constructor(saveIncompleteResultsCallback) {
    this.shouldUsePavlovia = isHostPavlovia();
    this.pavloviaPlugin = jsPsych.plugins['pavlovia'];
    this.saveIncompleteResultsCallback = saveIncompleteResultsCallback;
    this.hasServerSession = false;
  }

  async initServerSessionIfNeeded() {
    if (this.shouldUsePavlovia
      && !this.hasServerSession) {
      return await this.initServerSession();
    }
  }

  async finishServerSessionIfNeeded() {
    if (this.shouldUsePavlovia
      && this.hasServerSession) {
      return await this.finishServerSession();
    }
  }

  async initServerSession() {
    console.log("PavloviaManager.initServerSession called");
    let response = this.pavloviaPlugin._init(null, 'config.json',
      this.saveIncompleteResultsCallback);
    this.hasServerSession = true;
    return await response;
  }

  async finishServerSession() {
    console.log("PavloviaManager.finishServerSession called");
    let response = this.pavloviaPlugin._finish();
    this.hasServerSession = false;
    return await response;
  }

  async saveStudyRunData(studyRunData, sync = false,
    simulateDelay = false, simulatedDelayMs = 11800) {
    if (simulateDelay) {
      await sleep(simulatedDelayMs);
    }
    // save the run data to the server or locally depending on the current
    // host and running mode
    if (this.shouldUsePavlovia) {
      try {
        await this.saveStudyRunDataToServer(studyRunData, sync);
      }
      catch (error) {
        console.error('PavloviaManager.saveStudyRunDataToServer failed with error',
          error);
        throw error
      }
    } else {
      promptToSaveStudyRunData(studyRunData, 'csv');
    }
  }

  async saveStudyRunDataToServer(studyRunData, sync = false) {
    console.log("PavloviaManager.saveStudyRunDataToServer called");
    if (!this.hasServerSession) {
      let errorMsg = "A server session must be initiated \
        before any data can be saved to the server";
      console.error(errorMsg);
      throw errorMsg;
    }
    return await this.pavloviaPlugin._save(null,
      studyRunData.toCSVString(), sync,
      studyRunData.csvFilename);
  }
}

function isHostPavlovia() {
  const hostname = window.location.hostname;
  return hostname.includes("pavlovia");
}
