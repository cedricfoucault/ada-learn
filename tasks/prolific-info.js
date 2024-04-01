const isProlificSubject = urlSearchParams.has('PROLIFIC_PID');

const prolificInfo = isProlificSubject ? {
  prolificPid: urlSearchParams.get('PROLIFIC_PID'),
  studyId: urlSearchParams.get('STUDY_ID'),
  sessionId: urlSearchParams.get('SESSION_ID'),
  completionURL: "https://app.prolific.co/submissions/complete?cc=CJ8EQCN9"
} : null;

