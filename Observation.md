- Position cost
  - Allow backtracking because it only cares about position
  - It forces the calculated velocity to match the spacing of the global plan (fixed delta t)
- Velocity ref
  - Tell the robot to follow the global plan at a certain speed
  - Might not allow backtracking because the ref velocity is positve
- Acceleration
  - Smooth velocity change
  - Might not stop fast enough to avoid collision
- Theta error

- omega W

  - Make the lidar and the map misalign
  - But limit the curvature of the turn
    - It makes the robot act like an ackermann

- Constraining the last horizon step

  - Might result in the bending of the mpc plan if it wants to match a certain velocity reference, because going off course will give higher velocity

- Add time as an optimization variable
