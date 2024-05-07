Ada-Learn Behavioral Dataset
============================

This dataset accompanies the paper:

> **Two determinants of dynamic adaptive learning for magnitudes and probabilities**  
> Cédric Foucault, and Florent Meyniel  
> *Open Mind: Discoveries in Cognitive Science (2024)*   

Link to the article: [doi.org/10.1162/opmi_a_00139](https://doi.org/10.1162/opmi_a_00139)

If you use these data for your research, please cite the article.

The data were collected online from participants recruited on [Prolific](https://www.prolific.com).
Each participant completed an experiment in which they performed one or more learning tasks.
See [the paper](https://doi.org/10.1162/opmi_a_00139) and its companion [github repository](https://github.com/cedricfoucault/ada-learn) for more details about these experiments.

Study directories
-----------------

The data is organized into three directories.
Each directory contains one dataset corresponding to one study.
The data is structured in the same way for all studies.

[`ada-learn_study`](ada-learn_study)
: This is the main study presented in the above paper, involving 96 subjects. Each subject performed both the magnitude learning task and the probability learning task one after the other, in a task order that was counterbalanced across subjects.

[`ada-pos_study`](ada-pos_study)
: This is a pilot study in which subjects performed only the magnitude learning task (called "ada-pos" in this dataset). The data from the first 8 subjects were used to inform the design of the ada-learn study. Additional data were then collected to form a larger dataset, comprising 20 subjects in total.

[`ada-prob_study`](ada-prob_study)
: This is a pilot study in which subjects performed only the probability learning task (called "ada-prob" in this dataset). The data from the first 8 subjects were used to inform the design of the ada-learn study. Additional data were then collected to form a larger dataset, comprising 20 subjects in total.

File structure
---------------

Each study directory contains the following files. These files were created from the original data (contained in the companion github repository for the paper) using a python script (`analysis/data_build_aggregated_dataset.py` in that repository).

`data_subject-level.csv`
: This file contains the data that is uniquely defined at the level of each subject.
E.g., the subject's id, which task order the subject performed, and how long they took to complete the experiment.

`data_session-level.csv`
: This file contains the data uniquely defined at the level of each session.
E.g., the identifier of the observation/outcome sequence that was presented to the subject  in that session, and the score they obtained at the end of the session.

`data_outcome-level.csv`
: This file contains the data uniquely defined at the level of each observation
in the sequence, referred to as an "outcome" in this dataset. E.g., the value of the outcome/observation, its index in the sequence, and the value of the subject's estimate after having received that observation.

`task-parameters.csv`
: This file contains the parameter values that were used for each task during the study. E.g., the number of sessions subjects performed.

The content of each file is further detailed below.

Subject-level data (`data_subject-level.csv`)
---------------------------------------------

`subject`
: The subject's unique identifier.

`date`
: The date and time at which the subject started the experiment, written in UTC timezone and ISO 8601 format: YYYY-MM-DDTHH:MM:SS.

`completion_time_min`
: The time (in minutes) the subject took to complete the experiment (i.e., time elapsed between the time when the subject pressed the Accept button of the informed consent and started viewing the first task instruction, and the time when the subject finished the very last session and the data started to upload to the server).

`task_order`
: For the ada-learn study only. Describes the order in which the subject performed the two tasks: "pos-prob" if the subject performed the magnitude learning task first, "prob-pos" if the subject performed the probability learning task first.


Session-level data (`data_session-level.csv`)
---------------------------------------------

`subject`
: The subject's unique identifier.

`task`
: The task performed by the subject ('ada-pos' for the magnitude learning task, 'ada-prob' for the probability learning task).

`session_idx`
: The index of the session within the task for this subject.

`score`
: The subject's score for this session, normalized between 0 and 1.

`sequence_id`
: An identifier for the outcome sequence that was presented to the subject in this session. Identical outcome sequences have been presented across subjects; this identifier can be used to find them easily.
Note that the identifier must be considered only within one task:
across the two tasks, a same identifier may be used, but it does not refer
to the same outcome sequence.


Outcome-level data (`data_outcome-level.csv`)
---------------------------------------------

`subject`
: The subject's unique identifier.

`task`
: The task performed by the subject ('ada-pos' for the magnitude learning task, 'ada-prob' for the probability learning task).

`session_idx`
: The index of the session within the task for this subject.

`outcome_idx`
: The index of the outcome in this session.

`outcome`
: The value of the outcome (in the magnitude task, this corresponds to the normalized horizontal position of the snowball stimulus; in the probability task, this is 1 for a blue outcome stimulus or 0 for a yellow outcome stimulus).

`estimate`
: This is the subject's estimate after having observed that outcome, corresponding, in both tasks, to the normalized slider position at the end of the 1.5s interval
separating each outcome (i.e., immediately before the next outcome appears).

`hidden_parameter`
: This is the true generative value of the hidden parameter that the subject must estimate (called "hidden variable" in the paper). In the magnitude task, this is the mean of the Gaussian distribution used to generate the outcomes. In the probability task, this is the parameter of the Bernoulli distribution used to generate the outcomes, i.e. the probability that the outcome is equal to 1.

`did_change_point_occur`
: Boolean indicating whether a change point occurred, i.e. whether the hidden parameter changed, before the outcome was generated.

`sequence_id`
: Same as in the session-level data file; see above description.


Task parameters (`task-parameters.csv`)
---------------------------------------

This file contains the following fields.

`task`
: The name of the task ('ada-pos' for the magnitude learning task, 'ada-prob' for the probability learning task).

`n_sessions`
: The number of sessions each subject performed.

`n_outcomes`
: The number of outcomes per session.

`p_c`
: The generative probability of a change point occurring.

`outcome_sd`
: This field only applies to the ada-pos task (for ada-prob, it is left blank). The standard deviation of the Gaussian distribution used to generate outcomes.


Contact
-------

If you have any questions about this dataset or would like to use it, feel free to contact cedric.foucault@gmail.com.
