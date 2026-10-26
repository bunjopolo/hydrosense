# HydroSense scope

Implemented: CSV/Excel inspection, arbitrary tag-to-asset mapping, per-asset
conditional regression, chronological calibration/evaluation, saved models,
actual-versus-expected traces, file replay, editable local PI connection
settings, read-only PI polling, quality gates and persistent deviations.

Next deployment work requires the user's environment: verify PI authentication
and point access; train on reviewed healthy operating history; test against
known faults and operating transitions; establish acceptable false-alarm rates;
run a prospective comparison alongside the existing SmartSignal workflow.

Future extensions: temporal/lag features, multiple operating-regime selection,
model approval/rollback UI, durable event history, notification integrations,
network authentication and service packaging. These are not claimed as present.
