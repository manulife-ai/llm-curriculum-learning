from dataclasses import dataclass


@dataclass
class TeacherForcingSchedule:
    def get_p_ar(self, step: int, metrics: dict | None = None) -> float:
        return 0.0


@dataclass
class LinearSchedule:
    p_max: float
    ramp_steps: int

    def get_p_ar(self, step: int, metrics: dict | None = None) -> float:
        if self.ramp_steps <= 0:
            return self.p_max
        return min(self.p_max, self.p_max * (step / self.ramp_steps))


@dataclass
class ExponentialSchedule:
    p_max: float
    ramp_steps: int

    def get_p_ar(self, step: int, metrics: dict | None = None) -> float:
        if self.ramp_steps <= 0:
            return self.p_max
        return min(self.p_max, self.p_max * (1.0 - (2.718281828 ** (-step / self.ramp_steps))))


@dataclass
class MetricBasedSchedule:
    p_max: float
    ramp_steps: int

    def get_p_ar(self, step: int, metrics: dict | None = None) -> float:
        if not metrics:
            return 0.0
        return min(self.p_max, metrics.get("progress", 0.0) * self.p_max)


def build_schedule(config: dict):
    schedule_type = config["curriculum"]["schedule_type"]
    if schedule_type == "teacher_forcing":
        return TeacherForcingSchedule()
    if schedule_type == "linear":
        return LinearSchedule(config["curriculum"]["p_max"], config["curriculum"]["ramp_steps"])
    if schedule_type == "exponential":
        return ExponentialSchedule(config["curriculum"]["p_max"], config["curriculum"]["ramp_steps"])
    if schedule_type == "metric_based":
        return MetricBasedSchedule(config["curriculum"]["p_max"], config["curriculum"]["ramp_steps"])
    raise ValueError(f"Unsupported schedule type: {schedule_type}")
