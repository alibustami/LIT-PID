"""A simple PID controller implementation."""


class PID:
    """A simple PID controller."""

    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.0):  # noqa
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, setpoint, pv, dt):
        """Compute the PID output value for given reference input and feedback."""
        error = setpoint - pv
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = (
            self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        )
        self.prev_error = error
        return output
