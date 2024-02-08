#!/usr/bin/env bash
# This script is used to generate the results for the final report.

# ----------------------------------------------------------------
# Train the models
# ----------------------------------------------------------------
python -m pinn_air.train_model --model BaselineModel --position_error 1.0 --velocity_error 1.0 --position_error_payload 1.0 --continuity_last_input_first_output 0.0 --output_continuity 0.0 --quaternion_norm 0.0 --quaternion_error 1.0 --physics_error 0.0 --exponential_decay_real 0.1 --exponential_decay_physics 0.1 --output_path trained_models/baseline --num_epochs 3000
python -m pinn_air.train_model --model PINNAirModel --position_error 20.0 --velocity_error 20.0 --position_error_payload 20.0 --continuity_last_input_first_output 0.0 --output_continuity 0.0 --quaternion_norm 10.0 --quaternion_error 20.0 --physics_error 5.0 --exponential_decay_real 0.1 --exponential_decay_physics 0.60 --slack_weight 0.1 --output_path trained_models/pinn_air --num_epochs 3000

# Generate some plots of the training trajectories
python -m pinn_air.stats_n_plots.plot_3dtrajectory

# ----------------------------------------------------------------
# Generate the results for the baseline model
# ----------------------------------------------------------------
# Results for TABLE I: RMSE of the different models, over the entire test
python -m pinn_air.stats_n_plots.rmse --model BaselineModel --output_dir trained_models/baseline > trained_models/baseline/rmse.txt

# Results for plots
python -m pinn_air.stats_n_plots.rmse_stats --model BaselineModel --output_dir trained_models/baseline
python -m pinn_air.stats_n_plots.plots --model BaselineModel --output_dir trained_models/baseline
python -m pinn_air.stats_n_plots.plots2 --model BaselineModel --output_dir trained_models/baseline
python -m pinn_air.stats_n_plots.plot_error_mean --model BaselineModel --output_dir trained_models/baseline

# ----------------------------------------------------------------
# Generate the results for the PINN-Air model
# ----------------------------------------------------------------
# Resuls for TABLE I: RMSE of the different models, over the entire test
python -m pinn_air.stats_n_plots.rmse --model PINNAirModel --output_dir trained_models/pinn_air > trained_models/pinn_air/rmse.txt

# Results for plots
python -m pinn_air.stats_n_plots.rmse_stats --model PINNAirModel --output_dir trained_models/pinn_air
python -m pinn_air.stats_n_plots.plots --model PINNAirModel --output_dir trained_models/pinn_air
python -m pinn_air.stats_n_plots.plots2 --model PINNAirModel --output_dir trained_models/pinn_air
python -m pinn_air.stats_n_plots.plots_slack --model PINNAirModel --output_dir trained_models/pinn_air
python -m pinn_air.stats_n_plots.plot_error_mean --model PINNAirModel --output_dir trained_models/pinn_air

# Generate the plots of the PINN-air against the baseline
python -m pinn_air.stats_n_plots.plot_error_mean_against_baseline --output_dir trained_models/pinn_air --output_baseline trained_models/baseline

# Generate predictions for the test set of the PINN-air model and save them to a .mat file
python -m pinn_air.stats_n_plots.save2mat --model PINNAirModel --output_dir trained_models/pinn_air

# Generate inference time statistics
python -m pinn_air.stats_n_plots.inference_time --model PINNAirModel --output_dir trained_models/pinn_air