# FedCSL-SimCLR Suite History

This file is appended automatically by `fedcsl_simclr_suite_runner.py`.

Suggested workflow:

1. Keep `fedcsl_simclr_suite_plans.yml` as the editable source of candidate plans.
2. All plans in the suite should preserve the FedCSL base mechanism:
   - multi-scale shapelet encoder
   - period scoring / `UseACF: true`
3. Only tune auxiliary choices such as `gamma`, `lr`, `wd`, `batch_size`,
   `beta`, alignment switches, local epoch count, and heterogeneity settings.
4. Use `GATE_ACC` to tighten or relax the fifth-evaluation screening threshold.
5. Review the latest section in this file after each batch run, then revise the YAML.
