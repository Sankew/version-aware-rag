# Kittiwake architecture overview

Kittiwake is Brackwater's fleet manager. It assigns tasks to robots, plans paths and reports throughput.

## Components

- **Dispatcher:** assigns pick and transport tasks to robots every 200 ms
- **Path planner:** computes collision-free routes on the site map
- **Telemetry service:** ingests robot state at 10 Hz per robot
- **Site console:** web UI for warehouse supervisors

## Deployment

Each customer site runs an edge server with the dispatcher and path planner, so robots keep working if the internet connection drops. Telemetry and the console run in Brackwater's cloud.

## Limits

One edge server supports up to 150 robots.
