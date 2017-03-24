import os

import bokeh.plotting
import bokeh.models
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelEncoder


def colors_from_column(hue_column):
    """Apply a color palette based on the values in a column.

    Parameters
    ----------
    hue_column : pandas.Series
        The column to map to a set of hex colors.

    Returns
    -------
    A pandas.Series containing the hex color values of each point.
    """
    encoder = LabelEncoder()
    labels = pd.Series(LabelEncoder().fit_transform(hue_column.values))
    pallette = sns.color_palette().as_hex()
    return labels.apply(lambda x: pallette[x]).tolist()


def select_filter_table(fig, column_name, source=None, name=None, **kwargs):
    """Adds a HTML table that is filtered based on a box select tool.

    Parameters
    ----------
    fig : bokeh.plotting.Figure
        Figure on which to plot
    column_name : str
        Name of the column in the figure's ColumnDataSource
    source : bokeh.models.ColumnDataSource
        The column data source of 'fig'
    name : str
        Bokeh series name to give to the selected data.
    **kwargs
        Any further arguments to be passed to fig.scatter

    Returns
    -------
    bokeh.plotting.Figure
        Figure containing the row contatenated `fig` and table.
    """
    # add an html table
    if source is None:
        source = fig.select(dict(type=bokeh.models.ColumnDataSource))[0]
    table_source = bokeh.models.ColumnDataSource(data=source.to_df())

    # Check if the figure as a box selector. If not add one.
    box_selector = fig.select(dict(type=bokeh.models.BoxSelectTool))
    if not box_selector:
        name = 'main' if name is None else name
        box_selector = bokeh.models.BoxSelectTool(name=name)
        fig.add_tools(box_selector)

    columns = [
        bokeh.models.widgets.TableColumn(field=column_name, title=column_name)
    ]
    data_table = bokeh.models.widgets.DataTable(
        source=table_source, columns=columns, **kwargs)

    selector_filename = 'filter_select.js'
    current_dir = os.path.dirname(__file__)
    with open(os.path.join(current_dir, selector_filename)) as cbk_js:
        callback_code = cbk_js.read() % column_name

    generic_callback = bokeh.models.CustomJS(
        args=dict(source=table_source, target_obj=data_table),
        code=callback_code
    )
    source.callback = generic_callback

    return bokeh.layouts.row([fig, data_table])


def hover_tooltip(fig, source, cols=None, name=None):
    """Add a hover tooltip that displays the value in `cols` of the
    selected point.

    Parameters
    ----------
    fig : bokeh.plotting.Figure
        Figure on which to plot
    source : bokeh.models.ColumnDataSource
        The column data source of 'fig'
    cols : list (default=None)
        Name of the columns displayed in the hover tool.
    name : str
        Bokeh series name to give to the selected data.

    Returns
    -------
    bokeh.plotting.Figure
        Figure with the hover tool added.
    """
    # Create the hover tool, and make sure it is only active with
    # the series we plotted in with name.
    name = 'main' if name is None else name
    hover = bokeh.models.HoverTool(names=[name])

    if cols is None:
        # Display *all* columns in the tooltips
        hover.tooltips = [(c, '@' + c) for c in source.column_names]
    else:
        # Display just the given columns in the tooltips
        hover.tooltips = [(c, '@' + c) for c in cols]

    hover.tooltips.append(('index', '$index'))

    # Finally add/enable the tool
    fig.add_tools(hover)

    return fig


def scatter_plot(x, y, data, hue=None,
                 table_column=None, hover_columns=None,
                 title=None, fig=None, name=None, marker='circle',
                 fig_width=500, fig_height=500,
                 hide_axes=True, hide_grid=True, **kwargs):
    """Plots an interactive scatter plot of `x` vs `y` using bokeh. Contains
    an additional table that will be filtered based on the selected data.

    Parameters
    ----------
    x : str
        Name of the column to use for the x-axis values
    y : str
        Name of the column to use for the y-axis values
    data : pandas.DataFrame
        DataFrame containing the data to be plotted.
    hue : str
        Name of the column to use to color code the scatter plot.
    table_column : str (default=None)
        The column to use to create the filterable table. If None then
        no table is displayed.
    fig : bokeh.plotting.Figure, optional
        Figure on which to plot (if not given then a new figure will be created)
    name : str
        Bokeh series name to give to the scattered data
    marker : str
        Name of marker to use for scatter plot
    **kwargs
        Any further arguments to be passed to fig.scatter

    Returns
    -------
    bokeh.plotting.Figure
        Figure (the same as given, or the newly created figure)
    """
    data = data.copy()

    # If we haven't been given a Figure obj then create it with default
    # size etc.
    if fig is None:
        tools = 'box_zoom,reset,help'
        fig = bokeh.plotting.figure(
            width=fig_width, height=fig_height, tools=tools,
            title=title)

        if hide_axes:
            fig.xaxis.visible = False
            fig.yaxis.visible = False

        if hide_grid:
            fig.xgrid.grid_line_color = None
            fig.ygrid.grid_line_color = None

    # add hue if necessary
    if hue:
        if hue not in data.columns:
            raise ValueError('Column `{}` specified for `hue` '
                             'not in dataframe. '.format(hue))
        data['hue'] = colors_from_column(data[hue])
        kwargs['color'] = 'hue'

    # We're getting data from the given dataframe
    source = bokeh.models.ColumnDataSource(data=data)

    # We need a name so that we can restrict hover tools to just this
    # particular 'series' on the plot. You can specify it (in case it
    # needs to be something specific for other reasons), otherwise
    # we just use 'main'
    if name is None:
        name = 'main'

    # Actually do the scatter plot
    # (other keyword arguments will be passed to this function)
    fig.scatter(x, y, source=source, name=name, marker=marker, **kwargs)

    if hover_columns is not None:
        fig = hover_tooltip(fig, source=source, cols=hover_columns)
    if table_column is not None:
        fig = select_filter_table(fig, table_column, source=source)

    return fig
