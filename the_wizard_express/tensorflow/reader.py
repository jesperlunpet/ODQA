def shape(tensor, dim=None):
    """Gets the most specific dimension size(s) of the given tensor.
    This is a wrapper around the two ways to get the shape of a tensor: (1)
    t.get_shape() to get the static shape, and (2) tf.shape(t) to get the dynamic
    shape. This function returns the most specific available size for each
    dimension. If the static size is available, that is returned. Otherwise, the
    tensor representing the dynamic size is returned.
    Args:
        tensor: Input tensor.
        dim: Desired dimension. Use None to retrieve the list of all sizes.
    Returns:
        output = Most specific static dimension size(s).
    """
    static_shape = tensor.get_shape()
    dynamic_shape = tf.shape(tensor)
    if dim is not None:
        return tf.dimension_value(static_shape[dim]) or dynamic_shape[dim]
    else:
        return [
            tf.dimension_value(d) or dynamic_shape[i]
            for i, d in enumerate(static_shape)
        ]

def span_candidates(masks, max_span_width):
    """Generate span candidates.
    Args:
        masks: <int32> [num_retrievals, max_sequence_len]
        max_span_width: int
    Returns:
        starts: <int32> [num_spans]
        ends: <int32> [num_spans]
        span_masks: <int32> [num_retrievals, num_spans]
    """
    _, max_sequence_len = shape(masks)
    def _spans_given_width(width):
        current_starts = tf.range(max_sequence_len - width + 1)
        current_ends = tf.range(width - 1, max_sequence_len)
        return current_starts, current_ends

    starts, ends = zip(*(_spans_given_width(w + 1)
                        for w in range(max_span_width)))

    # [num_spans]
    starts = tf.concat(starts, 0)
    ends = tf.concat(ends, 0)

    # [num_retrievals, num_spans]
    start_masks = tf.gather(masks, starts, axis=-1)
    end_masks = tf.gather(masks, ends, axis=-1)
    span_masks = start_masks * end_masks

    return starts, ends, span_masks
