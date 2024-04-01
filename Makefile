#
# This Makefile allows to reproduce all the paper results using the command:
#
#      make paper_results
#
# When you run that command in the terminal, each of the paper result files,
# which are listed in the 'PAPER_RESULTS' variable below, will be checked,
# and the analysis code to produce it will be run if it needs to be reproduced
# (i.e., if it doesn't exist, or if it could have been modified due to, e.g., a change in the code).
# If a result file already exists and is up to date, the code to produce it will
# not be run again.
#
# If you want to run the code regardless of the existing results files,
# you can delete them beforehand, using the following command:
#
#      make clean_results
#
# An alternative would be to run `make paper_results1 with the '--always-make' option ,
# but this has the undersirable feature that when the same code is responsible
# for producing N files, it will be run N times instead of just once.
#

#
# Store the list of files that make up the results of the paper in a variable
#

# Fig. 1 G and H
PAPER_RESULTS := results/ada-learn_23-02-17/accuracy_subject-estimate_by_model-estimate_ada-pos.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/accuracy_subject-estimate_by_model-estimate_ada-prob.pdf
# Table S1 and S2
PAPER_RESULTS += results/ada-learn_23-02-17/accuracy_subject-estimate_by_model-estimate_stats_pearsonr_ada-pos.csv \
				 results/ada-learn_23-02-17/accuracy_subject-estimate_by_model-estimate_stats_pearsonr_ada-prob.csv
PAPER_RESULTS += results/ada-learn_23-02-17/accuracy_subject-estimate_by_model-estimate_stats_slope_ada-pos.csv \
				 results/ada-learn_23-02-17/accuracy_subject-estimate_by_model-estimate_stats_slope_ada-prob.csv
# Fig. 6
PAPER_RESULTS += results/ada-learn_23-02-17/bias-variance_decomposition_ada-pos.csv \
				 results/ada-learn_23-02-17/bias-variance_decomposition_ada-pos.pdf \
				 results/ada-learn_23-02-17/bias-variance_decomposition_ada-prob.csv \
				 results/ada-learn_23-02-17/bias-variance_decomposition_ada-prob.pdf \
				 results/ada-learn_23-02-17/bias-variance_decomposition_illustration-sequence_ada-pos.pdf \
				 results/ada-learn_23-02-17/bias-variance_decomposition_illustration-sequence_ada-prob.pdf
# Supplementary Text 4
PAPER_RESULTS += results/ada-learn_23-02-17/corr_across_subs_between_tasks.csv
PAPER_RESULTS += results/ada-learn_23-02-17/corr_across_subs_within_task.csv
# Fig. 2C
PAPER_RESULTS += results/ada-learn_23-02-17/illustration_adaptability_ada-pos.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/illustration_adaptability_ada-prob.pdf
# Fig. 2 A and B
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_around_cp_ada-pos.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_around_cp_ada-prob.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_around_cp_corr-with-model_ada-pos.csv
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_around_cp_corr-with-model_ada-prob.csv
# Fig. S2
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_around_cp_model_ada-pos.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_around_cp_model_ada-prob.pdf
# Fig. S4 A and B
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_around_cp_use-only-trials-with-update_ada-pos.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_around_cp_use-only-trials-with-update_ada-prob.pdf
# Fig. 5 A and B
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_regression_weights_cpp_uncertainty_ada-pos.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_regression_weights_cpp_uncertainty_ada-prob.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_regression_weights_cpp_uncertainty_stats_ada-pos.csv
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_regression_weights_cpp_uncertainty_stats_ada-prob.csv
# Fig. 4A
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_regression_weights_cpp_uncertainty_model_ada-pos.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_regression_weights_cpp_uncertainty_model_ada-prob.pdf
# Fig S4 C and D
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_regression_weights_cpp_uncertainty_stats_use-only-trials-with-update_ada-pos.csv
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_regression_weights_cpp_uncertainty_stats_use-only-trials-with-update_ada-prob.csv
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_regression_weights_cpp_uncertainty_use-only-trials-with-update_ada-pos.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_regression_weights_cpp_uncertainty_use-only-trials-with-update_ada-prob.pdf
# Supplementary Text 2
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_regression_weights_cpp_uncertainty_control-as-in-mcguire_ada-pos.csv
# Fig. 5 C, D, E, F
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_relationship_uncertainty_ada-pos.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_relationship_uncertainty_ada-prob.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_relationship_uncertainty_cpp_ada-pos.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_relationship_uncertainty_cpp_ada-prob.pdf
# Fig. 4 C, D
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_relationship_uncertainty_cpp_model_ada-pos.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_relationship_uncertainty_cpp_model_ada-prob.pdf
# Fig. S4 E, F, G, H
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_relationship_uncertainty_use-only-trials-with-update_ada-pos.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_relationship_uncertainty_use-only-trials-with-update_ada-prob.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_relationship_uncertainty_cpp_use-only-trials-with-update_ada-pos.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/learning-rate_relationship_uncertainty_cpp_use-only-trials-with-update_ada-prob.pdf
# Fig. S5B
PAPER_RESULTS += results/ada-learn_23-02-17/noisy_delta_rule_parameters_noise-level-constant_ada-pos.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/noisy_delta_rule_parameters_noise-level-constant_ada-prob.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/noisy_delta_rule_parameters_noise-level-prop-error_ada-pos.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/noisy_delta_rule_parameters_noise-level-prop-error_ada-prob.pdf
# Fig. S5C
PAPER_RESULTS += results/ada-learn_23-02-17/noisy_delta_rule_sim_learning-rate_regression_weights_noise-level-prop-error_ada-prob.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/noisy_delta_rule_sim_learning-rate_regression_weights_noise-level-prop-error_ada-pos.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/noisy_delta_rule_sim_learning-rate_regression_weights_noise-level-constant_ada-prob.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/noisy_delta_rule_sim_learning-rate_regression_weights_noise-level-constant_ada-pos.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/normative_effects_on_updating_illustration_determinants.pdf
# Fig 3
PAPER_RESULTS += results/ada-learn_23-02-17/normative_effects_on_updating_illustration_uncertainty-ada-pos_level-1_likelihood.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/normative_effects_on_updating_illustration_uncertainty-ada-pos_level-1_posterior.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/normative_effects_on_updating_illustration_uncertainty-ada-pos_level-1_prior.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/normative_effects_on_updating_illustration_uncertainty-ada-pos_level-2_likelihood.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/normative_effects_on_updating_illustration_uncertainty-ada-pos_level-2_posterior.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/normative_effects_on_updating_illustration_uncertainty-ada-pos_level-2_prior.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/normative_effects_on_updating_illustration_uncertainty-ada-prob_level-1_likelihood.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/normative_effects_on_updating_illustration_uncertainty-ada-prob_level-1_posterior.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/normative_effects_on_updating_illustration_uncertainty-ada-prob_level-1_prior.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/normative_effects_on_updating_illustration_uncertainty-ada-prob_level-2_likelihood.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/normative_effects_on_updating_illustration_uncertainty-ada-prob_level-2_posterior.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/normative_effects_on_updating_illustration_uncertainty-ada-prob_level-2_prior.pdf
# Fig. S1
PAPER_RESULTS += results/ada-learn_23-02-17/outcome_information_gain_studies.pdf
# Fig. S6
PAPER_RESULTS += results/ada-learn_23-02-17/performance_over_sessions_ada-pos.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/performance_over_sessions_ada-prob.pdf
# Fig. 4 E and F
PAPER_RESULTS += results/ada-learn_23-02-17/normative_variable_dynamics_example_ada-pos.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/normative_variable_dynamics_example_ada-prob.pdf
# Fig. 4B
PAPER_RESULTS += results/ada-learn_23-02-17/task_ranges_cpp.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/task_ranges_uncertainty.pdf
# Fig. S3
PAPER_RESULTS += results/ada-learn_23-02-17/update_interval_hist_ada-pos.csv
PAPER_RESULTS += results/ada-learn_23-02-17/update_interval_hist_ada-pos.pdf
PAPER_RESULTS += results/ada-learn_23-02-17/update_interval_hist_ada-prob.csv
PAPER_RESULTS += results/ada-learn_23-02-17/update_interval_hist_ada-prob.pdf

#
# Create a paper_results Make target to produce all those results
#

paper_results: $(PAPER_RESULTS)

#
# Create a clean_results Make target to delete them
#

.PHONY: clean_results
clean_results:
	rm results/ada-learn_23-02-17/*

#
# Specify how the files can be produced using the Python scripts
#

#
# About Make - Things to know to understand the below code.
#
# - A Makefile consists of a set of rules that specify how targets (files)
#   can be produced.
#
#  - The syntax for rules is:
#
#         targets: preqrequisites
#				command
#				command
#				command
#
# - '%' is a wildcard.
#      It allows to create patterns that can be used
#      to specify the rule for making many files that match the same pattern.
#
# - '$<' is an automatic variable which contains the name of the first prerequisite.
#
# - '\' is used to escape newlines (without it, targets and prerequisites must be
#       written on the same line)
# 
# - $(VAR): will be replaced by the value of the variable named 'VAR'
#
# - When a file (target) matches more than one pattern rule:
#   - in version 3.82 and higher: make will choose the one that is most specific. 
#   - in version 3.81 and lower: make will pick the first pattern rule that matches in the file.
#   To ensure that this Makefile works as intended regardless of the version of make,
#   I always put the more specific pattern rules first in this Makefile.
#

ODIR := results/ada-learn_23-02-17

$(ODIR)/accuracy_subject-estimate_by_model-estimate%: \
	analysis/accuracy_estimate_by_model_estimate_group.py
	python $<

$(ODIR)/bias-variance_decomposition%: \
	analysis/bias_variance_decomposition.py
	python $<

$(ODIR)/corr_across_subs_between_tasks%: \
	analysis/corr_across_subs_between_tasks.py
	python $<

$(ODIR)/corr_across_subs_within_task%: \
	analysis/corr_across_subs_within_task.py
	python $<

$(ODIR)/illustration_adaptability%: \
	analysis/illustration_adaptability.py
	python $<

$(ODIR)/learning-rate_around_cp_use-only-trials-with-update%: \
	analysis/learning_rate_around_cp_group.py
	python $< --use_only_trials_with_update --skip_model

$(ODIR)/learning-rate_around_cp%: \
	analysis/learning_rate_around_cp_group.py
	python $<

$(ODIR)/learning-rate_regression_weights_cpp_uncertainty_model%: \
	analysis/learning_rate_regression_cpp_uncertainty_group.py
	python $< --do_model

$(ODIR)/learning-rate_regression_weights_cpp_uncertainty_use-only-trials-with-update% \
$(ODIR)/learning-rate_regression_weights_cpp_uncertainty_stats_use-only-trials-with-update%: \
	analysis/learning_rate_regression_cpp_uncertainty_group.py
	python $< --use_only_trials_with_update

$(ODIR)/learning-rate_regression_weights_cpp_uncertainty_control%: \
	analysis/learning_rate_regression_cpp_uncertainty_group.py
	python $< --do_control

$(ODIR)/learning-rate_regression_weights_cpp_uncertainty%: \
	analysis/learning_rate_regression_cpp_uncertainty_group.py
	python $<

$(ODIR)/learning-rate_relationship_uncertainty_cpp_model%: \
	analysis/learning_rate_relationship_cpp_uncertainty_group.py
	python $< --do_model

$(ODIR)/learning-rate_relationship_uncertainty_use-only-trials-with-update% \
$(ODIR)/learning-rate_relationship_uncertainty_cpp_use-only-trials-with-update% : \
	analysis/learning_rate_relationship_cpp_uncertainty_group.py
	python $< --use_only_trials_with_update

$(ODIR)/learning-rate_relationship_%: \
	analysis/learning_rate_relationship_cpp_uncertainty_group.py
	python $<

$(ODIR)/noisy_delta_rule_parameters%: \
	analysis/noisy_delta_rule_parameters.py
	python $<

$(ODIR)/noisy_delta_rule_sim%: \
	analysis/noisy_delta_rule_sim_group_analyses.py
	python $<

$(ODIR)/normative_effects_on_updating_illustration%: \
	analysis/normative_effects_on_updating_illustration.py
	python $<

$(ODIR)/outcome_information_gain%: \
	analysis/outcome_information_gain.py
	python $<

$(ODIR)/performance_over_sessions%: \
	analysis/performance_over_sessions_group.py
	python $<

$(ODIR)/normative_variable_dynamics_example%: \
	analysis/normative_variable_dynamics_example.py
	python $<

$(ODIR)/task_ranges%: \
	analysis/task_ranges_cpp_uncertainty.py
	python $<

$(ODIR)/update_interval_hist%: \
	analysis/update_interval_hist_group.py
	python $<
