Ada-Learn
=========

This repository accompanies the paper:

> **Two determinants of dynamic adaptive learning for magnitudes and probabilities**  
> Cédric Foucault, and Florent Meyniel  
> *Open Mind: Discoveries in Cognitive Science (2024)*   

Link to the article: [doi.org/10.1162/opmi_a_00139](https://doi.org/10.1162/opmi_a_00139)

Three essential elements of this work are provided in this repository that allow
others to replicate and build upon this work.

1. [Tasks (`tasks` folder)](#tasks) : The code of the tasks that the participants
performed in the study presented in the above paper. It can be used to collect
new data, both in the lab and online.

2. [Data (`data` folder)](#data) : The data collected from participants. This includes
data from the main study presented in the paper and from two pilot studies in which
participants performed only one of the two tasks.

3. [Analysis (`analysis` folder)](#analysis) : The data analysis code, which allows to
reproduce all the results and figures of the paper.

Each element is described below in a dedicated section.

If you use this repository for your research, please cite the above paper.


Tasks
-----

The tasks run in the web browser. To run them, simply clone or download the repository
on your computer, and open the HTML file corresponding to the desired experiment in your
browser (e.g. Google Chrome or Firefox).

- [`tasks/ada-pos-study.html`](tasks/ada-pos-study.html) : experiment to perform the magnitude learning task alone
- [`tasks/ada-prob-study.html`](tasks/ada-prob-study.html) : experiment to perform the probability learning task alone
- [`tasks/ada-pos-prob-study.html`](tasks/ada-pos-prob-study.html)
or [`tasks/ada-prob-pos-study.html`](tasks/ada-prob-pos-study.html): experiment to perform the two tasks one after the other,
starting with the magnitude learning task (first file, 'pos-prob') or the probability learning task
(second file, 'prob-pos'). This is the experiment that participants did in the main study of the paper
(half of them did pos-prob, the other half did prob-pos).

There are a number of experiment parameters you can configure via the URL.
See below for a list of these URL parameters.

When you run the experiment this way, the data will be saved in a file on your
machine. You will be able to choose the location of this file at the end of the experiment
(it will appear as a download in the browser).

In the paper studies, the data were collected online. The experiment was hosted 
on Pavlovia and the data were sent to the Pavlovia server, instead of being saved locally.
The data saving mode (local or server) is determined automatically by the code. If you wish to
collect new data online and send it to a different server, you are free to adapt the existing code.

The tasks were created in javascript. Their javascript code is shared between the different
experiment files and can be found in the `.js` files (these files are imported from the HTML file
via the `<script>` elements).

The code was built from scratch in pure javascript. No framework was used because
the existing frameworks were too limited for the desired tasks.

The task implementation provided here, using javascript and `<canvas>`, enables experimenters
to retain complete control over the task design, and ensures a smooth experience for participants,
with minimal latency. In short, this implementation is similar to how most video game code works.
It is based on drawing the graphics and user interface in a `<canvas>` and updating them continuously
depending on task events and user interactions. The graphics are updated on a frame-by-frame basis,
with a loop synchronised to the screen refresh rate. User interactions are implemented by monitoring
the pointer events (mouse or trackpad movements and clicks) and updating the state of the experiment,
task, and user interface accordingly.

### URL parameters

A number of experimental parameters can be changed without modifying the code
by passing them in the URL via a query string. The query string is appended to
the HTML filename preceded by '?'. In this query string, each parameter is specified by
'<name>=<value>' and delimited from others by '&'. For example, to configure the number of sessions
to 5 and change the language to French in the 'ada-prob-study.html' experiment, the URL to write
would be, starting from the html filename: `ada-prob-study.html?nSessions=5&lang=fr`.

Here is a list of existing URL parameters:
- `subjectId`: Unique identifier for the subject (by default, a random identifier is generated).
- `nSessions`: Number of task sessions to perform. For the experiments with two tasks
('ada-pos-prob' and 'ada-prob-pos'), you must specify the number of sessions for each task.
This is done by writing, e.g., `nSessions=3&nSessions=5` to perform 3 sessions of the first task
and 5 sessions of the second task.
- `lang`: `en` for English version and `fr` for French version of the experiment.
- `skipConsent`, `skipInstructions`, `skipFeedbackForm`: Used to skip the consent form step of the
experiment, the task instructions step, and to hide the form displayed at the end of the experiment.
These parameters are provided without a value (no '=...'). 
- `shortenSequences`: This parameter is provided without a value. It activates a mode that is useful
to debug the code or to test the different experimental stages more quickly. When this mode is activated,
the length of sequences is drastically reduced, with only 2 observations per session. Some task features
may not work properly in this mode.

For more parameters that can be changed, see the code, starting with the objects `urlSearchParams`
and `studyParams`.


Data
----

The [`data`](data) folder contains all the behavioral data that were collected online for the above paper.
Subjects were recruited from Prolific. The data were collected over the course of three studies:
the main study, involving 96 subjects, and two pilot studies carried out to test the tasks individually,
involving 20 subjects each.

The data is included in two formats:

- The original acquisition format, consisting of one data file per subject. This is the format used
by the analysis code in this repository. The subject files are all stored in the data folder.
The list of subject files corresponding to the main study is provided in the file
[`data/group_defs/group_def_ada-learn_23-02-17.csv`](data/group_defs/group_def_ada-learn_23-02-17.csv)
(the files 'group_def_ada-pos_23-02-13.csv' and 'group_def_ada-prob_23-02-13.csv' correspond to the
two pilot studies).

- A more structured, aggregated format, organized by study, contained in
[`data/structured-dataset`](data/structured-dataset).
This dataset has also been put [on OSF](https://osf.io/2bk95).
It was created to facilitate the sharing of the data and its future use by the community.

If you want to look at the data and conduct new analysis, use the structured format,
as it is easier to understand and manipulate compared to the original format.
For more details, see the dedicated README in [`data/structured-dataset`](data/structured-dataset).

Note that the data in the original format contains additional data compared to the aggregated format,
but these are not used for the paper results. These additional data include,
for instance, data describing the slider's trajectory over time at the millisecond level. These data
are more heterogeneous. This is why the original format is less structured and easy to work with.


Analysis
--------

The data analyses are implemented by Python scripts stored in the [`analysis`](analysis) folder.
These allow to reproduce all the results and figures reported in the paper.

Each script performs one analysis. A docstring describing the analysis is often provided
at the top of the file. To run the analysis, simply execute it with Python from the root
of the repository. Using the terminal, the command to run would be:

```
python analysis/<script_name>.py
```

When the script is executed, the results of the analysis will be saved in the [`results`](results) folder.
For reference, the results of the paper are provided with the repository.

To reproduce all the paper results, you can use the following `make` commands.
The first command will delete the existing result files, and the second command
will automatically run all the necessary Python scripts to reproduce the paper result
files (instead of you running the Python scripts individually).

```
make clean_results
make paper_results
```

The `make` tool is provided with Linux and macOS.
The repository [Makefile](Makefile) specifies what result files should be made when the `paper_results`
target is provided, and how to produce these files from the Python scripts.
See the [Makefile](Makefile) for more information.

### Requirements

To run the analysis code, you will need to install Python and the necessary Python modules. The required
modules are listed in the [`requirements.txt`](requirements.txt) file. You can install them using `pip`
by running the following command in the terminal from the root of the repository:

```
pip install -r requirements.txt
```

The results of the paper were produced using Python 3.9 and the module versions listed in
[`requirements.txt`](requirements.txt). Using these versions ensures that the code runs correctly
and that the results are exactly reproduced. Other versions should work, but this has not been tested.

If you want to test the code using these versions without modifying the versions installed
in your default environment, you can create a new environment with [conda](https://docs.conda.io/) beforehand,
using the following commands:

```
conda create -n "ada-learn" python=3.9
conda activate ada-learn
pip install -r requirements.txt
```


Contact
-------

Feel free to contact cedric.foucault@gmail.com if you have questions or would like to collaborate.
