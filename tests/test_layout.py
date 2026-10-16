"""Layout input validation; numerical map checks live in test_kernels."""
import pytest
from vsa_layout import Segment, build_vsa_layout


@pytest.mark.parametrize("head_num", [0, -1, 1.5, True, None])
def test_layout_rejects_invalid_head_num(head_num):
    with pytest.raises(ValueError, match="head_num"):
        build_vsa_layout([[Segment("video", (1, 2, 3))]], head_num=head_num)


@pytest.mark.parametrize("dense_query_full", [None, 1, "False"])
def test_layout_rejects_invalid_dense_query_full(dense_query_full):
    with pytest.raises(ValueError, match="dense_query_full"):
        build_vsa_layout([[Segment("video", (1, 2, 3))]], head_num=2,
                         dense_query_full=dense_query_full)


@pytest.mark.parametrize("samples", [
    [], [[]], [Segment("video", (2, 4, 4))], [[(2, 4, 4)]],
    [[Segment("text", 4)]], None,
])
def test_invalid_layout_configuration(samples):
    with pytest.raises(ValueError):
        build_vsa_layout(samples, head_num=2)


@pytest.mark.parametrize("sparsity", [-.1, 1.1, float("nan"), float("inf"), [.5, .9], "0.5"])
def test_invalid_segment_sparsity(sparsity):
    with pytest.raises(ValueError, match="sparsity"):
        Segment("video", (2, 4, 4), sparsity=sparsity)


@pytest.mark.parametrize("kind", ["text", "audio", "keyframe"])
def test_dense_segments_have_no_sparsity(kind):
    assert Segment(kind, 3).sparsity is None
    with pytest.raises(ValueError, match="only supported for video"):
        Segment(kind, 3, sparsity=0)


@pytest.mark.parametrize("kind,size", [("audio", (1, 2, 3)), ("video", 12), ("video", (0, 4, 4)), ("text", -1)])
def test_invalid_segments(kind, size):
    with pytest.raises(ValueError):
        Segment(kind, size)
