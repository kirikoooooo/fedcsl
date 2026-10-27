"""FedCSL observability tooling (Argo Workflows + Prometheus + Grafana).

The package is intentionally dependency-light: the exporter and stage wrapper use
only the Python standard library, while dataset validation reuses torch from the
training environment.
"""

__version__ = "0.1.0"
