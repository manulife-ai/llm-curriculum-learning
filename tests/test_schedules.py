from src.curriculum.schedules import LinearSchedule, TeacherForcingSchedule


def test_teacher_forcing_schedule_is_zero():
    schedule = TeacherForcingSchedule()
    assert schedule.get_p_ar(10) == 0.0


def test_linear_schedule_caps_at_p_max():
    schedule = LinearSchedule(p_max=0.4, ramp_steps=10)
    assert schedule.get_p_ar(0) == 0.0
    assert schedule.get_p_ar(5) == 0.2
    assert schedule.get_p_ar(50) == 0.4
