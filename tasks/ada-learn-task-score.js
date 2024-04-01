// Shared Ada-Prob/Ada-Pos functions

function getScoreFracFromSessionData(sessionData) {
  const isAdaProb = (sessionData.taskName == ADA_PROB_TASK_NAME_DASH_CASE);
  let trueValue = isAdaProb ? sessionData.p1 : sessionData.mean;
  let estimate = sessionData.estimate;
  let idealObserverEstimate = isAdaProb ? sessionData.idealObserverEstimate : null;
  let outcome = sessionData.outcome;
  let didTrueValueChange = isAdaProb ? sessionData.didP1Change : sessionData.didMeanChange;
  if (studyParams.debugShortenSequences) {
    trueValue = trueValue.slice(0, sessionData.nTrials);
    estimate = estimate.slice(0, sessionData.nTrials);
    idealObserverEstimate = idealObserverEstimate?.slice(0, sessionData.nTrials);
    outcome = outcome?.slice(0, sessionData.nTrials);
    didTrueValueChange = didTrueValueChange?.slice(0, sessionData.nTrials);
  }
  if (isAdaProb) {
    return getAdaProbScoreFrac(trueValue, estimate, idealObserverEstimate);
  } else {
    return getAdaPosScoreFrac(trueValue, estimate, outcome, didTrueValueChange);
  }
  
}

// Scoring in Ada-Prob

const ADA_PROB_SCORE_FRAC_STAY_STILL = 0.0; // this is the score (before applying the softplus)
                                    // that a subject should have when giving
                                    // a constant estimate of 0.5
                                    // (i.e., staying still with the slider at the starting position)
                                    // after applying softplus, 0 -> 0.14
const ADA_PROB_SCORE_FRAC_IDEAL_OBSERVER = 0.9; // this is the score (before applying the softplus)
                                       // that the ideal observer obtains
                                       // (note that this ideal observer does not know about the
                                       // min odd change constraint)

// expected mean absolute error of the stay still strategy.
const ADA_PROB_EXPECTED_MAE_STAY_STILL = 0.2;
// expected mean absolute error of the ideal observer (computed empirically).
const ADA_PROB_EXPECTED_MAE_IDEAL_OBSERVER = 0.125;

function getAdaProbScoreFrac(trueValue, estimate, idealObserverEstimate) {
  const scoreMae = meanAbsoluteError(trueValue, estimate);
  // We compare the subject's mean absolute error to that of two references
  // - the ideal observer, which provides the high score reference
  // - the 'stay still' strategy, which provides the low score refrence
  // For the high score reference, we use the mean absolute error of the ideal
  // observer obtained for the particular sequence the subject observed in this session.
  // For the low score reference, we use the expected mean absolute error of
  // the "stay still" strategy and shift it so as to keep a constant slope
  // between the low and the high score reference across sessions
  let scoreIdealObserverMae = meanAbsoluteError(trueValue, idealObserverEstimate);
  const highScoreMae = scoreIdealObserverMae;  
  const lowScoreMae = (ADA_PROB_EXPECTED_MAE_STAY_STILL
    + (scoreIdealObserverMae - ADA_PROB_EXPECTED_MAE_IDEAL_OBSERVER));
  return getFractionalValue(scoreMae, lowScoreMae, highScoreMae,
    ADA_PROB_SCORE_FRAC_STAY_STILL, ADA_PROB_SCORE_FRAC_IDEAL_OBSERVER);
}

// Scoring in Ada-Pos 

const ADA_POS_SCORE_FRAC_LAST_OUTCOME = 0.33; // this is the score (before applying the softplus)
                                      // that a subject should have
                                      // when simply giving as estimate
                                      // the last observed outcome
const ADA_POS_SCORE_FRAC_ORACLE_SAMPLE_MEAN = 1.0; // this is the score of a virtual "oracle" observer
                                             // who perfectly knows when the change points occur
                                             // and computes its estimate as the sample mean of the outcomes
                                             // since the last change point
const ADA_POS_GENERATIVE_STD = 10/300; // the standard deviation of the outcome-generative distribution
const ADA_POS_GENERATIVE_MAE = ADA_POS_GENERATIVE_STD * Math.sqrt(2 / Math.PI); // the mean absolute deviation of the outcome-generative distribution
const ADA_POS_GENERATIVE_MEAN_RUN_LENGTH = 10;
// expected mean absolute error of the last outcome strategy
// this is used to threshold the MAE used for the low-score reference,
// so that in sequences where "last outcome" happens to be a very good strategy,
// e.g. when there are very many changes,  the subject is not penalized for it.
const ADA_POS_EXPECTED_MAE_LAST_OUTCOME = ADA_POS_GENERATIVE_MAE;
// expected mean absolute error of the oracle sample mean strategy in one average run length
// this is used to threshold the MAE used for the high-score reference,
// so that in sequences where the oracle can do better than usually expected
// by performing very minute adjustments, e.g. because there are very few changes in the sequence,
// the subject is not penalized for performing such minute adjustments.
const ADA_POS_EXPECTED_MAE_ORACLE_SAMPLE_MEAN = (ADA_POS_GENERATIVE_MAE
  * 1 / ADA_POS_GENERATIVE_MEAN_RUN_LENGTH
  * range(1, ADA_POS_GENERATIVE_MEAN_RUN_LENGTH+1).reduce( // sum(1/sqrt(i), i=1..10)
            (acc, i) => (acc + 1 / Math.sqrt(i)), 0));

function getAdaPosScoreFrac(trueValue, estimate, outcome, didMeanChange) {
  const scoreMae = meanAbsoluteError(trueValue, estimate);
  // We compare the subject's mean absolute error to that of two references
  // - the oracle, which provides the high score reference
  // - the 'last outcome' strategy, which provides the low score refrence
  // For each of these two strategies, we use as error the maximum of
  // the mean absolute error obtained for the particular sequence
  // the subject observed in this session and the expected mean absolute error
  // across many sequences. Taking the maximum ensures that subjects
  // are not penalized when the low-score strategy performs well, or
  // penalized for not performing very minute estimate adjustments
  // as the  high-score strategy does when the sequence is stable.
  let scoreLastOutcomeMae = meanAbsoluteError(trueValue, outcome);
  const oracleEstimate = getAdaPosOracleSampleMeanEstimate(outcome, didMeanChange);
  let scoreOracleMae = meanAbsoluteError(trueValue, oracleEstimate);
  scoreLastOutcomeMae = Math.max(scoreLastOutcomeMae, ADA_POS_EXPECTED_MAE_LAST_OUTCOME); 
  scoreOracleMae = Math.max(scoreOracleMae, ADA_POS_EXPECTED_MAE_ORACLE_SAMPLE_MEAN);
  return getFractionalValue(scoreMae, scoreLastOutcomeMae, scoreOracleMae,
    ADA_POS_SCORE_FRAC_LAST_OUTCOME, ADA_POS_SCORE_FRAC_ORACLE_SAMPLE_MEAN);
}

function getAdaPosOracleSampleMeanEstimate(outcome, didMeanChange) {
  let estimate = new Array(outcome.length + 1);
  let runLength = 0;
  for (let i = 0; i < outcome.length; i++) {
    if (didMeanChange[i] || (i == 0)) {
      estimate[i] = outcome[i];
      runLength = 1;
    } else {
      estimate[i] = (outcome[i] + estimate[i-1] * runLength) / (runLength + 1);
      runLength++;
    }
  }
  return estimate;
}
