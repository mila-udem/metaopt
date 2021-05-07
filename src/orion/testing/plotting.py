# -*- coding: utf-8 -*-
"""
Plotting testing support module
===============================

Plotting testing support module providing defaults, functions and mocks.

"""
# pylint: disable=protected-access


def assert_regret_plot(plot):
    """Checks the layout of a regret plot"""
    assert plot.layout.title.text == "Regret for experiment 'experiment-name'"
    assert plot.layout.xaxis.title.text == "Trials ordered by suggested time"
    assert plot.layout.yaxis.title.text == "Objective 'loss'"

    trace1 = plot.data[0]
    assert trace1.type == "scatter"
    assert trace1.name == "trials"
    assert trace1.mode == "markers"
    assert len(trace1.y) == 1
    assert not trace1.x

    trace2 = plot.data[1]
    assert trace2.type == "scatter"
    assert trace2.name == "best-to-date"
    assert trace2.mode == "lines"
    assert len(trace2.y) == 1
    assert not trace2.x


def assert_regrets_plot(plot, names, balanced=10, with_avg=False):
    """Checks the layout of a regrets plot"""
    assert plot.layout.title.text == "Average Regret"
    assert plot.layout.xaxis.title.text == "Trials ordered by suggested time"
    assert plot.layout.yaxis.title.text == "loss"

    if with_avg:
        line_plots = plot.data[::2]
        err_plots = plot.data[1::2]
    else:
        line_plots = plot.data

    assert len(line_plots) == len(names)

    for name, trace in zip(names, line_plots):
        assert trace.type == "scatter"
        assert trace.name == name
        assert trace.mode == "lines"
        if balanced:
            assert len(trace.y) == balanced
            assert len(trace.x) == balanced

    if with_avg:
        assert len(err_plots) == len(names)
        for name, trace in zip(names, err_plots):
            assert trace.fill == "toself"
            assert trace.name == name
            assert not trace.showlegend
            if balanced:
                assert len(trace.y) == balanced * 2
                assert len(trace.x) == balanced * 2


def assert_rankings_plot(plot, names, balanced=10, with_avg=False):
    """Checks the layout of a rankings plot"""
    assert plot.layout.title.text == "Average Rankings"
    assert plot.layout.xaxis.title.text == "Trials ordered by suggested time"
    assert plot.layout.yaxis.title.text == "Ranking based on loss"

    if with_avg:
        line_plots = plot.data[::2]
        err_plots = plot.data[1::2]
    else:
        line_plots = plot.data

    assert len(line_plots) == len(names)
    for name, trace in zip(names, line_plots):
        assert trace.type == "scatter"
        assert trace.name == name
        assert trace.mode == "lines"
        if balanced:
            assert len(trace.y) == balanced
            assert len(trace.x) == balanced

    if with_avg:
        assert len(err_plots) == len(names)
        for name, trace in zip(names, err_plots):
            assert trace.fill == "toself"
            assert trace.name == name
            assert not trace.showlegend
            if balanced:
                assert len(trace.y) == balanced * 2
                assert len(trace.x) == balanced * 2


def assert_lpi_plot(plot, dims):
    """Checks the layout of a LPI plot"""
    assert plot.layout.title.text == "LPI for experiment 'experiment-name'"
    assert plot.layout.xaxis.title.text == "Hyperparameters"
    assert plot.layout.yaxis.title.text == "Local Parameter Importance (LPI)"

    trace = plot.data[0]
    assert trace["x"] == tuple(dims)
    assert trace["y"][0] > trace["y"][1]
    assert len(trace["error_y"]["array"]) == len(dims)


def assert_partial_dependencies_plot(
    plot,
    dims,
    custom_colorscale=False,
    smoothing=0.85,
    n_grid_points=5,
    n_samples=4,
    log_dims=None,
):
    """Checks the layout of a partial dependencies plot"""
    if not isinstance(n_grid_points, dict):
        n_grid_points = {dim: n_grid_points for dim in dims}
    if log_dims is None:
        log_dims = {}

    def _ax_label(axis, index):
        if index == 0:
            return f"{axis}axis"

        return f"{axis}axis{index + 1}"

    def _ax_layout(axis, index):
        return plot.layout[_ax_label(axis, index)]

    assert (
        plot.layout.title.text
        == "Partial dependencies for experiment 'experiment-name'"
    )

    assert plot.layout.coloraxis.colorbar.title.text == "Objective"
    assert plot.layout.yaxis.title.text == "Objective"

    yrange = _ax_layout("y", 0).range

    def all_indices():
        return {
            j * len(dims) + i + 1 for i in range(len(dims)) for j in range(i, len(dims))
        }

    def first_column():
        return {i * len(dims) + 1 for i in range(len(dims))}

    def last_row():
        return {len(dims) * (len(dims) - 1) + i + 1 for i in range(len(dims))}

    def diagonal():
        return {i * len(dims) + i + 1 for i in range(len(dims))}

    def assert_axis_log(axis, index, name):
        axis_type = _ax_layout(axis, index).type
        if name in log_dims:
            assert axis_type == "log"
        else:
            assert axis_type != "log"

    def assert_log_x():
        x_tested = set()
        for dim_i, dim_name in enumerate(dims):
            x_index = dim_i * len(dims) + dim_i
            for row in range(dim_i, len(dims)):
                assert_axis_log("x", x_index, dim_name)
                x_tested.add(x_index + 1)
                x_index += len(dims)

        assert x_tested == all_indices()

    assert_log_x()

    def assert_shared_y_on_diagonal():
        y_tested = set()
        for dim_i, dim_name in enumerate(dims):
            # Test shared y axis across the diagonal
            y_index = dim_i * len(dims) + dim_i
            assert _ax_layout("y", y_index).range == yrange
            y_tested.add(y_index + 1)

        assert y_tested == diagonal()

    assert_shared_y_on_diagonal()

    def assert_log_y():
        y_tested = set()
        for dim_i, dim_name in enumerate(dims):
            # Test shared y axis across the diagonal
            y_index = dim_i * len(dims) + dim_i
            # Should not be log
            assert_axis_log("y", y_index, None)
            y_tested.add(y_index + 1)

            y_index = dim_i * len(dims)
            for column in range(max(dim_i, 0)):
                assert_axis_log("y", y_index, dim_name)
                y_tested.add(y_index + 1)
                y_index += 1

        assert y_tested == all_indices()

    assert_log_y()

    def assert_x_labels():
        x_tested = set()
        for dim_i, dim_name in enumerate(dims):
            x_index = len(dims) * (len(dims) - 1) + dim_i
            assert _ax_layout("x", x_index).title.text == dim_name

        assert x_tested == last_row()

    def assert_y_labels():
        y_tested = set()
        for dim_i, dim_name in enumerate(dims):
            if dim_i > 0:
                # Test lable at left of row
                y_index = dim_i * len(dims)
                assert _ax_layout("y", y_index).title.text == dim_name
                y_tested.add(y_index + 1)
            else:
                assert _ax_layout("y", 0).title.text == "Objective"
                y_tested.add(1)

        assert y_tested == first_column()

    assert_y_labels()

    # assert x_tested == {1, 4, 5, 7, 8, 9}
    # assert y_tested == {1, 4, 5, 7, 8, 9}

    if custom_colorscale:
        assert plot.layout.coloraxis.colorscale[0][1] != "rgb(247,251,255)"
    else:
        assert plot.layout.coloraxis.colorscale[0][1] == "rgb(247,251,255)"

    data = plot.data
    data_index = 0
    for x_i, x_name in enumerate(dims):

        # Test scatter mean
        assert data[data_index].mode == "lines"
        assert data[data_index].showlegend is False
        assert len(data[data_index].x) == n_grid_points[x_name]
        assert len(data[data_index].y) == n_grid_points[x_name]
        data_index += 1
        # Test scatter var
        assert data[data_index].mode == "lines"
        assert data[data_index].fill == "toself"
        assert data[data_index].showlegend is False
        assert len(data[data_index].x) == 2 * n_grid_points[x_name]
        assert len(data[data_index].y) == 2 * n_grid_points[x_name]
        data_index += 1
        # Test scatter dots
        assert data[data_index].mode == "markers"
        assert data[data_index].showlegend is False
        assert data[data_index].customdata is not None
        assert data[data_index].hovertemplate is not None
        assert len(data[data_index].x) == n_samples
        assert len(data[data_index].y) == n_samples
        data_index += 1

        for y_i in range(x_i + 1, len(dims)):
            y_name = dims[y_i]

            # Test contour
            assert data[data_index].line.smoothing == smoothing
            # To share colorscale across subplots
            assert data[data_index].coloraxis == "coloraxis"
            assert len(data[data_index].x) == n_grid_points[x_name]
            assert len(data[data_index].y) == n_grid_points[y_name]
            assert data[data_index].z.shape == (
                n_grid_points[y_name],
                n_grid_points[x_name],
            )
            data_index += 1

            # Test scatter
            assert data[data_index].mode == "markers"
            assert data[data_index].showlegend is False
            assert len(data[data_index].x) == n_samples
            assert len(data[data_index].y) == n_samples
            data_index += 1

    # Make sure we covered all data
    assert len(data) == data_index


def assert_parallel_coordinates_plot(plot, order):
    """Checks the layout of a parallel coordinates plot"""
    assert (
        plot.layout.title.text
        == "Parallel Coordinates Plot for experiment 'experiment-name'"
    )

    trace = plot.data[0]
    for i in range(len(order)):
        assert trace.dimensions[i].label == order[i]
