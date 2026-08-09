from app.gpu_resource_gate import foreign_compute, parse_pmon, should_defer


PMON = """# gpu pid type sm mem enc dec jpg ofa fb ccpm command
0 2628 C+G 82 51 - - - - 0 0 vk3d.exe
0 44620 C 16 9 - - - - 0 0 python.exe
"""


def test_gpu_gate_ignores_our_python_and_detects_foreign_renderer():
    rows = parse_pmon(PMON)
    assert len(rows) == 2
    assert [row.name for row in foreign_compute(rows)] == ["vk3d.exe"]
    assert should_defer(rows) is True


def test_gpu_gate_allows_idle_or_allowlisted_compute():
    rows = parse_pmon("0 44620 C 90 40 - - - - 0 0 python.exe\n")
    assert foreign_compute(rows) == []
    assert should_defer(rows) is False
