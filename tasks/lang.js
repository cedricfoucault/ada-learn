const langId  = (urlSearchParams.has("lang") ? urlSearchParams.get("lang") : "en");

const lang_en = {
  acceptConsentSingleTaskButtonText: "Accept and continue",
  acceptConsentMultipleTasksButtonText: "Accept and continue to the first task",
  nextInstructionButtonText: "Next →",
  previousInstructionButtonText: "← Previous",
  startMainTaskButtonText: "Begin the task ⇥",
  resumeMainTaskButtonText: "Resume the task ⇥",
  reviewInstructionsButtonText: "⇤ Review the instructions",
  completeProlificSubmissionButtonText: "Complete your submission on Prolific",
  downloadDataFileButtonText: "Download the data file",

  promptStartFirstSession: "Click on the slider to start the session",
  promptStartNextSession: "Click on the middle of the track to start the next session",
  promptSavingData: "We are saving your data. You will be able to complete the study when this is finished. Please wait.",
  promptStudyComplete: "You have completed the study. Thank you for your participation!",
  promptEstimateProbability: "Estimate the hidden proportion",
  promptEstimatePosition: "Estimate the hidden position",
  promptStudyDataUploadFailedProlific: "The data upload failed. Please download the file and send us a message on Prolific to complete your submission.",
  
  sessionComplete: "Session completed!",
  score: "Score",

  trueProbabilitiesTitle: "The true hidden proportions (and in thin strokes, your estimates) were:",
  trueProbabilitySingularTitle: "The true hidden proportion (and in thin strokes, your estimates) were:",
  truePositionsTitle: "The hidden positions (and in thin strokes, your estimates) were:",
  truePositionSingularTitle: "The hidden position (and in thin strokes, your estimates) were:",

  localeIdentifier: "en-GB",
  currency: "GBP",

  instructionImgSrc(studyName, idx, taskIdx = null) {
    if (taskIdx !== null && taskIdx !== undefined) {
      return `img/${studyName}-instruction-image-en-${taskIdx+1}-${idx+1}@2x.png`;
    } else {
      return `img/${studyName}-instruction-image-en-${idx+1}@2x.png`;
    }
  },

  promptTaskCompleteWithIndex(taskIdx) {
    const ordinalNum = this.ordinalNumWithIndex(taskIdx);
    return `You have completed the ${ordinalNum} task. Well done.`;
  },

  nextTaskButtonTextWithIndex(nextTaskIdx) {
    const ordinalNum = this.ordinalNumWithIndex(nextTaskIdx);
    return `Proceed to the ${ordinalNum} task ⇥`;
  },

  ordinalNumWithIndex(idx) {
    switch (idx + 1) {
      case 1:
        return "first";
      case 2:
        return "second";
      case 3:
        return "third";
      case 4:
        return "fourth";
      case 5:
        return "fifth";
      case 6:
        return "sixth";
      case 7:
        return "seventh";
      case 8:
        return "eigth";
      case 9:
        return "ninth";
      case 10:
        return "tenth";
      default:
        console.assert("ordinalNumWIthIndex not implemented for idx > 9");
        return this.rankTextWithIndex(idx);
    }
  },

  rankTextWithIndex(idx) {
    let suffix;
    switch ((idx % 10)) {
      case 0:
        suffix = idx != 10 ? "st" : "th";
        break;
      case 1:
        suffix = idx != 11 ? "nd" : "th";
        break;
      case 2:
        suffix = idx != 12 ? "rd" : "th";
        break;
      default:
        suffix = "th";
        break;
    }
    return `${idx+1}${suffix}`;
  },

  currencyStringFromNumber(number) {
    return Intl.NumberFormat(this.localeIdentifier,
      {style: 'currency', currency: this.currency}).format(number);
  },

  valueLabelTextForScoreFrac(scoreFrac,
    gain = null, accumulatedGain = null) {
    let scorePercentText = `${parseFloat(scoreFrac*100).toFixed(0)}%`;
    if (gain != null) {
      let gainText = `+${this.currencyStringFromNumber(gain)}`;
      if (accumulatedGain != null) {
        gainText = gainText.concat(`, total accumulated: +${this.currencyStringFromNumber(accumulatedGain)}`);
      }
      return `${scorePercentText} (${gainText})`;
    } else {
      return scorePercentText;
    }
  },

  getTrackedQuantity(taskName) {
    if (taskName == ADA_PROB) {
      return "proportion";
    } else if (taskName == ADA_POS) {
      return "position";
    }
  },

  getRatioText(ratio) {
    if (ratio < ((1/3+1/4)/2)) {
      return "one quarter";
    } else if (ratio < ((1/3+1/2)/2)) {
      return "one third";
    } else if (ratio < ((1/2+2/3)/2)) {
      return "one half";
    } else if (ratio < ((2/3+3/4)/2)) {
      return "two thirds";
    } else {
      return "three quarters";
    }
  },

  getConsentInnerHTMLString(studyParams) {
    const payHTMLString = `You will be paid \
       £<span id="base-money">${studyParams.baseMoney.toFixed(2)}</span> \
       for your participation, plus a performance bonus of up to \
        £<span id="max-money-bonus">${studyParams.maxMoneyBonus.toFixed(2)}</span> \
       (<span id="max-money-bonus-percent">${studyParams.maxMoneyBonusPercent.toFixed(0)}</span>% \
       of the base pay)`;
    let par1HTMLString;
    if (studyParams.nTasks == 2) {
      par1HTMLString = `\
       <p>In this study, you will do two tasks one after the other, \
       a ${this.getTrackedQuantity(studyParams.taskNames[0])} tracking task \
       and a ${this.getTrackedQuantity(studyParams.taskNames[1])} tracking task. \
       You will spend about ${this.getRatioText(studyParams.estimatedCompletionTimeRatios[0])} \
       of the time on the first task
       and ${this.getRatioText(studyParams.estimatedCompletionTimeRatios[1])} \
       on the second task. \
       It will take approximately \
       <span id=completion-time>${studyParams.estimatedCompletionTime}</span> minutes total. \
       ${payHTMLString}.</p>`
    } else {
      par1HTMLString = `\
       <p>In this study, you will perform a tracking task. It will take \
       approximately \
       <span id=completion-time>${studyParams.estimatedCompletionTime}</span> minutes. \
       ${payHTMLString}.</p>`
    }

    const taskOrTasks = studyParams.nTasks > 1 ? "tasks" : "task";
    const acceptButtonTitle = (studyParams.nTasks > 1 ?
      this.acceptConsentMultipleTasksButtonText : this.acceptConsentSingleTaskButtonText);

    return (
      `<h1 class=title>Informed consent for participation in this study</h1>

      ${par1HTMLString}

       <p>This is an academic research study aimed at understanding how people \
       make their estimates. During the ${taskOrTasks}, we will measure and record the \
       movements of your pointer. The data we collect is anonymous (it is \
       associated with a unique identifier unrelated to your identity). \
       <a id="consent-notice-link" \
          href="${studyParams.consentNoticeURL}" \
          target="_blank">Click here</a> \
      to read an information notice about the study.</p>

       <p>By clicking on the <b>accept</b> button below, you indicate that \
       you agree to the terms of the study described in the information notice \
       and consent to participate. If you do not agree, \
       simply close this browser window.</p>

       <button id="accept-consent-button">${acceptButtonTitle}</button>`);
  },
};

const lang_fr = {
  acceptConsentSingleTaskButtonText: "Accepter et continuer",
  acceptConsentMultipleTasksButtonText: "Accepter et procéder à la première tâche",
  nextInstructionButtonText: "Continuer →",
  previousInstructionButtonText: "← Revenir",
  startMainTaskButtonText: "Commencer la tâche ⇥",
  resumeMainTaskButtonText: "Reprendre la tâche ⇥",
  reviewInstructionsButtonText: "⇤ Revoir les instructions",
  completeProlificSubmissionButtonText: "Complétez votre soumission sur Prolific",
  downloadDataFileButtonText: "Télécharger le fichier de données",

  promptStartFirstSession: "Cliquez sur le curseur pour démarrer la session",
  promptStartNextSession: "Cliquez au milieu du rail pour démarrer la session suivante",
  promptSavingData: "Nous sauvegardons vos données. Vous pourrez terminer l'étude lorsque ce sera terminé. Veuillez patienter.",
  promptStudyComplete: "Vous avez terminé l'étude. Merci de votre participation !",
  promptEstimateProbability: "Estimez la proportion cachée",
  promptEstimatePosition: "Estimez la position cachée",
  promptStudyDataUploadFailedProlific: "La sauvegarde des données a échoué. Merci de télécharger le fichier et de nous envoyer un message sur Prolific pour compléter votre soumission.",
  
  sessionComplete: "Session terminée !",
  score: "Score",

  trueProbabilitiesTitle: "Les vraies proportions cachées (et en traits fins, vos estimations) étaient :",
  trueProbabilitySingularTitle: "La vraie proportion cachée (et en traits fins, vos estimations) étaient :",
  truePositionsTitle: "Les positions cachées (et en traits fins, vos estimations) étaient :",
  truePositionSingularTitle: "La position cachée (et en traits fins, vos estimations) étaient :",

  localeIdentifier: "fr-FR",
  currency: "EUR",

  instructionImgSrc(studyName, idx, taskIdx = null) {
    if (taskIdx !== null && taskIdx !== undefined) {
      return `img/${studyName}-instruction-image-fr-${taskIdx+1}-${idx+1}@2x.png`;
    } else {
      return `img/${studyName}-instruction-image-fr-${idx+1}@2x.png`;
    }
  },

  promptTaskCompleteWithIndex(taskIdx) {
    const ordinalNum = this.ordinalNumWithIndex(taskIdx, true);
    return `Vous avez terminé la ${ordinalNum} tâche. Bien joué.`;
  },

  nextTaskButtonTextWithIndex(nextTaskIdx) {
    const ordinalNum = this.ordinalNumWithIndex(nextTaskIdx, true);
    return `Passer à la ${ordinalNum} tâche ⇥`;
  },

  ordinalNumWithIndex(idx, isFeminine = false) {
    switch (idx + 1) {
      case 1:
        return isFeminine ? "première" : "premier";
      case 2:
        return "deuxième";
      case 3:
        return "troisième";
      case 4:
        return "quatrième";
      case 5:
        return "cinquième";
      case 6:
        return "sixième";
      case 7:
        return "septième";
      case 8:
        return "huitième";
      case 9:
        return "neuvième";
      case 10:
        return "dixième";
      default:
        console.assert("ordinalNumWIthIndex not implemented for idx > 9");
        return this.rankTextWithIndex(idx);
    }
  },

  rankTextWithIndex(idx) {
    let rankText;
    switch (idx) {
      case 0:
        rankText = "1ère";
        break;
      case 1:
        rankText = "2ème";
        break;
      case 2:
        rankText = "3ème";
        break;
      default:
        rankText = `${idx+1}ème`;
        break;
    }
    return rankText;
  },

  currencyStringFromNumber(number) {
    return Intl.NumberFormat(this.localeIdentifier,
      {style: 'currency', currency: this.currency}).format(number);
  },

  valueLabelTextForScoreFrac(scoreFrac,
    gain = null, accumulatedGain = null) {
    let scorePercentText = `${parseFloat(scoreFrac*100).toFixed(0)}%`;
    if (gain != null) {
      let gainText = `+${this.currencyStringFromNumber(gain)}`;
      if (accumulatedGain != null) {
        gainText = gainText.concat(`, total accumulé : +${this.currencyStringFromNumber(accumulatedGain)}`);
      }
      return `${scorePercentText} (${gainText})`;
    } else {
      return scorePercentText;
    }
  },

  getTrackedQuantity(taskName) {
    if (taskName == ADA_PROB) {
      return "proportion";
    } else if (taskName == ADA_POS) {
      return "position";
    }
  },

  getRatioText(ratio) {
    if (ratio < ((1/3+1/4)/2)) {
      return "un quart";
    } else if (ratio < ((1/3+1/2)/2)) {
      return "un tiers";
    } else if (ratio < ((1/2+2/3)/2)) {
      return "une moitié";
    } else if (ratio < ((2/3+3/4)/2)) {
      return "deux tiers";
    } else {
      return "trois quarts";
    }
  },

  getConsentInnerHTMLString(studyParams) {
    const payHTMLString = `Vous serez rémunéré \
       <span id="base-money">${studyParams.baseMoney.toFixed(2)}</span> € \
       pour votre participation, plus une prime de performance pouvant atteindre \
       <span id="max-money-bonus">${studyParams.maxMoneyBonus.toFixed(2)}</span> € \
       (<span id="max-money-bonus-percent">${studyParams.maxMoneyBonusPercent.toFixed(0)}</span>% \
       du montant de base)`;
    let par1HTMLString;
    if (studyParams.nTasks == 2) {
      par1HTMLString = `\
       <p>Dans cette étude, vous allez effectuer deux tâches l'une après l'autre, \
       une tâche de suivi de ${this.getTrackedQuantity(studyParams.taskNames[0])} \
       et une tâche de suivi de ${this.getTrackedQuantity(studyParams.taskNames[1])}. \
       You passerez environ ${this.getRatioText(studyParams.estimatedCompletionTimeRatios[0])} \
       du temps sur la première tâche et
       ${this.getRatioText(studyParams.estimatedCompletionTimeRatios[1])} \
       sur la seconde. \
       Cela prendra environ \
       <span id=completion-time>${studyParams.estimatedCompletionTime}</span> minutes au total. \
       ${payHTMLString}.</p>`
    } else {
      par1HTMLString = `\
       <p>Dans cette étude, vous allez effectuer une tâche de suivi de \
       ${this.getTrackedQuantity(studyParams.taskName)}. \
       Cela prendre environ \
       <span id=completion-time>${studyParams.estimatedCompletionTime}</span> minutes. \
       ${payHTMLString}.</p>`
    }

    const taskOrTasks = studyParams.nTasks > 1 ? "tasks" : "task";
    const acceptButtonTitle = (studyParams.nTasks > 1 ?
      this.acceptConsentMultipleTasksButtonText : this.acceptConsentSingleTaskButtonText);

    return (
      `<h1 class=title>Consentement éclairé pour la participation à cette étude</h1>

      ${par1HTMLString}

       <p>Il s'agit d'une étude de recherche fondamentale visant à comprendre \
       comment les gens font leurs estimations. Lors de la tâche, nous mesurerons
       et enregistrerons les mouvements de votre pointeur. \
       Les données que nous recueillons sont anonymes (elles sont associées à un
       identifiant unique sans rapport avec votre identité). \
       <a id="consent-notice-link" \
          href="${studyParams.consentNoticeURL}" \
          target="_blank">Cliquez ici</a> \
      pour lire une notice d'information sur l'étude.</p>

       <p>En cliquant sur le bouton <b>accepter</b> ci-dessous, \
       vous indiquez que vous acceptez les conditions de l'étude décrites \
       dans la notice et que vous consentez à y participer. \
       Si vous n'êtes pas d'accord, fermez simplement cette fenêtre du navigateur.</p>

       <button id="accept-consent-button">${acceptButtonTitle}</button>`);
  },
};

const lang_dict = {
  en: lang_en,
  fr: lang_fr
};

const lang = lang_dict[langId];
