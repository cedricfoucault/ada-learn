const ADA_POS = "ada-pos";
const ADA_PROB = "ada-prob";
const ADA_POS_PROB = "ada-pos-prob";
const ADA_PROB_POS = "ada-prob-pos";

const ADA_POS_DEFAULT_STUDY_PARAMS = {
  studyName: ADA_POS,
  estimatedCompletionTime: 28, // in minutes
  baseMoney: 4.20, // in pounds
  maxMoneyBonus: 2.10, // in pounds
  maxMoneyBonusPercent: 50, // in % of base money
  consentNoticeURL: "resources/ada-pos_online_notice_informed_consent_b-50_en.pdf",
  feedbackFormSrc: "https://docs.google.com/forms/d/e/1FAIpQLSczG6qoADdAW3kdOOcGX-0nTEgMZ5iJvMc0Xq3pnuZ54lsAhw/viewform?embedded=true",
  nTasks: 1,
  taskName: ADA_POS,
  nSessions: 10,
  nInstructions: 16,
  idxFirstSessionCountedInMoneyBonus: 1,
};

const ADA_POS_CUSTOM_STUDY_PARAMS_BY_ID = {
  "63a325dcc1156b1f99e248a9": {},
  "63a9abaf72822ffc8cdc0098": {
    maxMoneyBonus: 1.05, // in pounds
    maxMoneyBonusPercent: 25, // in % of base money
    consentNoticeURL: "resources/ada-pos_online_notice_informed_consent_b-25_en.pdf",
  }
};

const ADA_PROB_DEFAULT_STUDY_PARAMS = {
  studyName: ADA_PROB,
  estimatedCompletionTime: 54, // in minutes
  baseMoney: 8.10, // in pounds
  maxMoneyBonus: 4.05, // in pounds
  maxMoneyBonusPercent: 50, // in % of base money
  consentNoticeURL: "resources/ada-prob_online_notice_informed_consent_b-50_en.pdf",
  feedbackFormSrc: "https://docs.google.com/forms/d/e/1FAIpQLScUKcjNOVVoKFipY9z5RYV5kbsWnX1Owlbe3Wkt_TY9x8V6Ig/viewform?embedded=true",
  nTasks: 1,
  taskName: ADA_PROB,
  nSessions: 22,
  nInstructions: 23,
  idxFirstSessionCountedInMoneyBonus: 1,
};

const ADA_PROB_CUSTOM_STUDY_PARAMS_BY_ID = {
  "63bebd5cd191ad47924bb51e": {}
};

const ADA_POS_PROB_DEFAULT_STUDY_PARAMS = {
  studyName: ADA_POS_PROB,
  estimatedCompletionTime: 56,
  baseMoney: 8.40, // in pounds
  maxMoneyBonus: 4.20, // in pounds
  maxMoneyBonusPercent: 50, // in % of base money
  consentNoticeURL: "resources/ada-pos-prob_online_notice_informed_consent_b-50_en.pdf",
  feedbackFormSrc: "https://docs.google.com/forms/d/e/1FAIpQLSeQUIXL7-6VYLxTbb7lLYYBPJmwJspqkTrEUMMmk8APumk-GA/viewform?embedded=true",
  nTasks: 2,
  taskNames: [ADA_POS, ADA_PROB],
  nSessionsByTask: [6, 15],
  nInstructionsByTask: [16, 23],
  idxFirstSessionCountedInMoneyBonusByTask: [1, 1],
  estimatedInstructionTimeByTask: [2.5, 5.0],
  estimatedSessionTime: 2.25,
};

const ADA_POS_PROB_CUSTOM_STUDY_PARAMS_BY_ID = {
  "63d95c5a78080dfe9805f20f": {},
  "63dccbfa4325f63630e93414": {},
  "63e7ddd8db276ad46a42a291": {},
  "63ebe1a3178121b385ab86c2": {},
  "63ee6ac609643e895a98abc5": {}
};

const ADA_PROB_POS_DEFAULT_STUDY_PARAMS = {
  studyName: ADA_PROB_POS,
  estimatedCompletionTime: 56,
  baseMoney: 8.40, // in pounds
  maxMoneyBonus: 4.20, // in pounds
  maxMoneyBonusPercent: 50, // in % of base money
  consentNoticeURL: "resources/ada-prob-pos_online_notice_informed_consent_b-50_en.pdf",
  feedbackFormSrc: "https://docs.google.com/forms/d/e/1FAIpQLScQncm_V5eqE8TnEXwaL8JcSCShimdEWf3Qq81vQ8BDj7RarA/viewform?embedded=true",
  nTasks: 2,
  taskNames: [ADA_PROB, ADA_POS],
  nSessionsByTask: [15, 6],
  nInstructionsByTask: [23, 16],
  idxFirstSessionCountedInMoneyBonusByTask: [1, 1],
  estimatedInstructionTimeByTask: [2.5, 5.0],
  estimatedSessionTime: 2.25,
};

const ADA_PROB_POS_CUSTOM_STUDY_PARAMS_BY_ID = {
  "63d95f80f4cc7b1b41dc978f": {},
  "63dcccfb0203be81afa1026a": {},
  "63e7dde8ad75f35697ff0cdd": {},
  "63ebe26f463910c503e037a9": {},
  "63ee6accfa2c99defa333ee7": {}
};

const DEFAULT_STUDY_PARAMS_BY_NAME = {
  [ADA_POS]: ADA_POS_DEFAULT_STUDY_PARAMS,
  [ADA_PROB]: ADA_PROB_DEFAULT_STUDY_PARAMS,
  [ADA_POS_PROB]: ADA_POS_PROB_DEFAULT_STUDY_PARAMS,
  [ADA_PROB_POS]: ADA_PROB_POS_DEFAULT_STUDY_PARAMS,
};

const CUSTOM_STUDY_PARAMS_BY_ID_BY_NAME = {
  [ADA_POS]: ADA_POS_CUSTOM_STUDY_PARAMS_BY_ID,
  [ADA_PROB]: ADA_PROB_CUSTOM_STUDY_PARAMS_BY_ID,
  [ADA_POS_PROB]: ADA_POS_PROB_CUSTOM_STUDY_PARAMS_BY_ID,
  [ADA_PROB_POS]: ADA_PROB_POS_CUSTOM_STUDY_PARAMS_BY_ID,
};

const studyParams = {};

function initStudyParams(studyName) {
  // Set default properties
  const defaultStudyParams = DEFAULT_STUDY_PARAMS_BY_NAME[studyName];
  for (let k in defaultStudyParams) {
    studyParams[k] = defaultStudyParams[k];
  }

  // Set properties based on the prolific study id given in the url search parameters
  const customStudyParamsById = CUSTOM_STUDY_PARAMS_BY_ID_BY_NAME[studyName];
  const studyId = isProlificSubject ? prolificInfo.studyId : null;
  if ((studyId !== null) && (studyId in customStudyParamsById)) {
    const customStudyParams = customStudyParamsById[studyId];
    for (let k in customStudyParams) {
      studyParams[k] = customStudyParams[k];
    }
  }

  // Set individual properties based on specific url search parameters
  // - Whether to show the consent notice and get subject's informed consent before
  // proceeding to the instructions
  studyParams.doGetSubjectConsent = (urlSearchParams.has('skipConsent') ?
    false : true);;
  // - Whether to show the feedback form at the end of the task
  studyParams.doShowFeedbackForm = (urlSearchParams.has('skipFeedbackForm') ?
    false : true);
  // - Whether to show task instructions
  studyParams.doInstructions = (urlSearchParams.has('skipInstructions') ?
    false : true);
  // - Whether to shorten the sequences to have only a couple trials (used for debugging)
  studyParams.debugShortenSequences = (urlSearchParams.has('shortenSequences') ?
    true : false);
  // - Number of sessions to perform for each task
  if (urlSearchParams.has('nSessions')) {
    if (studyParams.nTasks > 1) {
      const nSessionsByTaskStr = urlSearchParams.getAll('nSessions');
      studyParams.nSessionsByTask = nSessionsByTaskStr.map(x => parseInt(x, 10));
    } else {
      const nSessionsStr = urlSearchParams.get('nSessions');
      studyParams.nSessions = parseInt(nSessionsStr, 10);
    }
  }

  // Set properties derived from the existing ones
  // - Maximum money bonus that the subject can get in each task session
  if (studyParams.nTasks > 1) {
    const nSessionsSum = studyParams.nSessionsByTask.reduce(
      (accu, n) => accu + n, 0);
    const nSessionsNotCounted = studyParams.idxFirstSessionCountedInMoneyBonusByTask.reduce(
      (accu, idx) => accu + idx, 0)
    const nSessionsCounted = nSessionsSum - nSessionsNotCounted;
    studyParams.maxMoneyBonusPerSession = (studyParams.maxMoneyBonus
      / nSessionsCounted);
    const estimatedCompletionTimesByTask = studyParams.nSessionsByTask.map(
      (x, i) => (x * studyParams.estimatedSessionTime));
    const estimatedCompletionTimesSum = estimatedCompletionTimesByTask.reduce(
      (accu, t) => accu + t, 0);
    studyParams.estimatedCompletionTimeRatios = estimatedCompletionTimesByTask.map(
      t => (t / estimatedCompletionTimesSum));
  } else {
    const nSessionsCounted = (studyParams.nSessions
      - studyParams.idxFirstSessionCountedInMoneyBonus);
    studyParams.maxMoneyBonusPerSession = (studyParams.maxMoneyBonus
      / nSessionsCounted);
  }
}
