import dash
from dash import Dash, no_update
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import internal_functions as int_func

UPLOAD_DIRECTORY = "W:\\Neurophysiology-Storage1\\Wahl\\Anna\\"

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])



def format_fig(matrix, title="No session uploaded", zoom=False, center_coords_x=100, center_coords_y=100, is_tiff_mode=False):
    fig = px.imshow(matrix, color_continuous_scale=
        [[0.0, '#0d0887'],
        [0.0333333333333333, '#46039f'],
        [0.0444444444444444, '#7201a8'],
        [0.0555555555555555, '#9c179e'],
        [0.0666666666666666, '#bd3786'],
        [0.0888888888888888, '#d8576b'],
        [0.1111111111111111, '#ed7953'],
        [0.1333333333333333, '#fb9f3a'],
        [0.1444444444444444, '#fdca26'],
        [0.1777777777777777, '#f0f921'],
        [0.25, "white"], [0.4, "white"], [0.4, "grey"],[0.5, "grey"],
        [0.5, "red"], [0.6, "red"], [0.6, "green"], [0.7, "green"], [0.7, "pink"],[0.8, "pink"], [0.8, "black"],[1, "black"]], range_color=[0,5])
    if is_tiff_mode:
        matrix = np.interp(matrix, (matrix.min(), matrix.max()), (0, 1))
        fig = px.imshow(matrix, zmin=0, zmax=1)
        #fig=px.imshow(matrix, color_continuous_scale='gray', zmin=matrix.min(), zmax=matrix.max())
        fig.add_trace(go.Scatter(x=[center_coords_x], y=[center_coords_y], marker=dict(color='red', size=5)))
    #fig_neuron = px.imshow(footprints_1[0])
    fig.update_layout(#coloraxis_showscale=False,
                    autosize=False,
                    width=350, height=350,
                    margin=dict(l=5, r=5, t=25, b=5),
                    title={"text": title,
                        "yref": "paper",
                        "xref": "paper",
                        },
                    )
    fig.update_traces(hoverinfo='none',  hovertemplate=None )
    fig.update_xaxes(visible=False )
    fig.update_yaxes(visible=False)
    if zoom:
        xmin = max(center_coords_x-100,0) ; xmax = min(matrix.shape[0], center_coords_x+100)
        ymin = min(matrix.shape[1], center_coords_y+100); ymax = max(center_coords_y-100,0)
        fig.update_xaxes(range=[xmin, xmax], autorange=False)
        fig.update_yaxes(range=[ymin, ymax], autorange=False)
    return fig

# colour vars
cols = [1.5, 2.2, 2.8, 3.2, 3.8, 4.5]


empty_template = [[0]*512]*512
figs = [format_fig(empty_template)] * 6

sessions_data = [None] * 6
footprints = [None] * 6
orig_templates = [None] * 6
templates = [None] * 6
templates_tiff = [None] * 6
coms = [None]*6
print("here again")
pixel_ownerships = [np.full((512, 512), -1)]*6
cell_matching_df = []

filenames = [''] * 6
current_selected = [-1]*6
#selection_mode = False

#### controlss
controls = dbc.Card(
    [
        dbc.FormGroup(
            [
            dbc.Label("1. Upload npy files of the sessions"),
                #dbc.Spinner(html.Div(id="loading-output")),

             dbc.Row([
                    dcc.Upload(
                        id="upload_session_0",
                        children=html.Div(
                            dbc.Button('Upload 1', size="sm",  outline=True, color="secondary"),
                            #dbc.Button([dbc.Spinner(size="sm"), "Loading..."],'Upload 1', size="sm",  outline=True, color="secondary"),
                        ),
                        multiple=False,
                    ),
                    dcc.Upload(
                        id="upload_session_1",
                        children=html.Div(
                            dbc.Button('Upload 2', size="sm",  outline=True, color="secondary"),
                        ),
                        multiple=False,
                    ),
                    dcc.Upload(
                        id="upload_session_2",
                        children=html.Div(
                            dbc.Button('Upload 3', size="sm",  outline=True, color="secondary"),
                        ),
                        multiple=False,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dcc.Upload(
                        id="upload_session_3",
                        children=html.Div(
                            dbc.Button('Upload 4', size="sm",  outline=True, color="secondary"),
                        ),
                        multiple=False,
                    ),
                    dcc.Upload(
                        id="upload_session_4",
                        children=html.Div(
                            dbc.Button('Upload 5', size="sm",  outline=True, color="secondary"),
                        ),
                        multiple=False,
                    ),
                    dcc.Upload(
                        id="upload_session_5",
                        children=html.Div(
                            dbc.Button('Upload 6', size="sm",  outline=True, color="secondary"),
                        ),
                        multiple=False,
                    ),
                ]
            ),
        ]),
        dbc.FormGroup(
            [
                dbc.Label("2. Upload/Create neuron matching file"),
                dbc.Row(
                    html.Div([
                        dcc.Upload(
                            id="upload_csv_matching",  style={"visibility": "visible"},
                            children=html.Div(
                                dbc.Button('Upload csv', size="sm",  outline=True, color="secondary",
                                           style={"visibility": "visible"}),

                            ),
                            multiple=False,
                        ),
                        dbc.Button("Compute matchings", id="compute_matching_button", size="sm",  outline=True, active=False, color="secondary"),
                        html.Span(id="upload_confirmation", style={"vertical-align": "middle"}),
                        html.Span(id="compute_confirmation", style={"vertical-align": "middle"})
                    ])
                )
            ]),

    ],
    body=True,
)


select_controls = dbc.Card(
    [
        dbc.FormGroup(
            [
            dbc.Label("3. Choose selection mode"),

             dbc.Row(
                [   html.Div([
                    #html.Pre(id='placeholder', style={"display": "none"}) ,
                    #html.Pre(False, id='selection_mode', style={"display": "none"}) ,
                    dbc.Button("Selection mode", id="selection_mode_button", size="sm",  outline=True, active=True, color="secondary" ),
                    dbc.Button("Suggestion mode", id="suggestion_mode_button", size="sm",  outline=True, active=False, color="secondary" ),
                    ])
                ]
            ),
        ]),
        dbc.FormGroup(
            [
            dbc.Label("4. Display mode"),

             dbc.Row(
                [   html.Div([
                    #html.Pre(id='placeholder', style={"display": "none"}) ,
                    #html.Pre(False, id='selection_mode', style={"display": "none"}) ,
                    dbc.Button("Basic image", id="basic_mode_button", size="sm",  outline=True, active=True, color="secondary" ),
                    dbc.Button("Mean intensity image", id="tiff_mode_button", size="sm",  outline=True, active=False, color="secondary" ),
                    ])
                ]
            ),
        ]),
        dbc.FormGroup(
            [
            dbc.Label("5. Indicate if neuron does not exsit"),
            dbc.Row(
                [   html.Div([
                        dbc.Button("No match 1", id="no_match_1", size="sm", style={"visibility": "hidden"}, outline=True, color="secondary" ),
                        dbc.Button("No match 2", id="no_match_2", size="sm",  outline=True, color="secondary", active= False ),
                        dbc.Button("No match 3", id="no_match_3", size="sm",  outline=True, color="secondary", active= False ),
                        ])
                ]
            ),
            dbc.Row(
                [   html.Div([
                        dbc.Button("No match 4", id="no_match_4", size="sm",  outline=True, color="secondary", active= False ),
                        dbc.Button("No match 5", id="no_match_5", size="sm",  outline=True, color="secondary", active= False ),
                        dbc.Button("No match 6", id="no_match_6", size="sm",  outline=True, color="secondary", active= False ),
                        ])
                ]
            ),
        ]),
        dbc.FormGroup([
            dbc.Label("6. Click save after each matched neuron"),
            dbc.Row(
                [
                        html.Div([
                        dbc.Button("Save", id="save_button", block=True, size="sm",  outline=True, active=True, color="primary"),
                        html.Pre("Number of neurons to be matched", id='num_left_to_match')
                        ])
                ]
            )
        ])
    ],
    body=True,
)

app.layout = dbc.Container(
    [
        dbc.Row(dbc.Col(html.Div("CELL MATCHING ACROSS SESSIONS"), width=12), style={'padding': 10}),
        dbc.Row(
            [  dbc.Col(dcc.Graph(
                    id='graph_0',
                    figure=figs[0],
                    ), width=3),
                dbc.Col(dcc.Graph(
                    id='graph_1',
                    figure=figs[1]
                    ), width=3),
                dbc.Col(dcc.Graph(
                    id='graph_2',
                    figure=figs[2]
                    ), width=3),
                dbc.Col(controls, width={"size": 3, "offset": 9.5}),   # update
            ]
        ),
        dbc.Row(
            [   dbc.Col(dcc.Graph(
                    id='graph_3',
                    figure=figs[3]
                    ), width=3),
                dbc.Col(dcc.Graph(
                    id='graph_4',
                    figure=figs[4]
                    ), width=3),
                dbc.Col(dcc.Graph(
                    id='graph_5',
                    figure=figs[5]
                    ), width=3),
                dbc.Col(select_controls, width={"size": 3, "offset": 9.5}),
            ]
        ),
    ],
    fluid= True
)

################################### Callbacks ###################################
#################################################################################
#################################################################################

###### upload #######

def getdata_from_npyfile(filename, graph_id):
    print("click", UPLOAD_DIRECTORY + filename)
    session_data = np.load(UPLOAD_DIRECTORY + filename, allow_pickle=True).item()
    templates[graph_id] = session_data['template']
    orig_templates[graph_id] = session_data['template']
    print("basic template loaded")
    templates_tiff[graph_id] = session_data['mean_intensity_template']
    print("templates loaded")
    num_neurons = session_data['dff_trace'].shape[0]
    print("num_neurons")
    footprints[graph_id] = np.reshape(session_data['spatial_masks'].toarray(),
                                      (session_data['template'].shape[0],session_data['template'].shape[1] ,num_neurons),
                                      order='F') #
    footprints[graph_id] = np.transpose(footprints[graph_id], (2,0,1))
    print("transposed footprints")
    pixel_ownerships[graph_id] = int_func.pixel_neuron_ownership(footprints[graph_id])
    coms[graph_id] = int_func.compute_CoMs(footprints[graph_id])
    filenames[graph_id] = filename.partition(".")[0]
    print(type(templates), type(footprints), type(pixel_ownerships))
    return orig_templates[graph_id], templates[graph_id], footprints[graph_id], pixel_ownerships[graph_id]


def update_graph(tiff_mode, filename, graph_idx, curr_neuron):
    background = templates_tiff[graph_idx] if tiff_mode else templates[graph_idx]
    if curr_neuron == -1:
        new_graph = format_fig( background, title=filename.partition(".")[0], zoom=False, is_tiff_mode=tiff_mode)
    else:
        new_graph_matrix = np.where(footprints[graph_idx][curr_neuron] >= 0.01, cols[0], background)
        new_graph = format_fig(new_graph_matrix, title=filename.partition(".")[0], zoom=True,
                                        center_coords_x=int(coms[graph_idx][curr_neuron][1]),
                                        center_coords_y=int(coms[graph_idx][curr_neuron][0]),
                                        is_tiff_mode=tiff_mode)
    return new_graph


#graph updates

@app.callback([Output('graph_0', 'figure'),
               Output('graph_1', 'figure'),
               Output('graph_2', 'figure'),
               Output('graph_3', 'figure'),
               Output('graph_4', 'figure'),
               Output('graph_5', 'figure'),
               Output('num_left_to_match', 'children')
               ],
    [Input('upload_session_0', 'filename'),
     Input('upload_session_1', 'filename'),
     Input('upload_session_2', 'filename'),
     Input('upload_session_3', 'filename'),
     Input('upload_session_4', 'filename'),
     Input('upload_session_5', 'filename'),
     Input('graph_0', 'clickData'),
     Input('graph_1', 'clickData'),
     Input('graph_2', 'clickData'),
     Input('graph_3', 'clickData'),
     Input('graph_4', 'clickData'),
     Input('graph_5', 'clickData'),
     Input('save_button', 'n_clicks'),
     Input('suggestion_mode_button', 'active'),
     Input('tiff_mode_button', 'active')],
    [State('graph_0', 'figure'),
       State('graph_1', 'figure'),
       State('graph_2', 'figure'),
       State('graph_3', 'figure'),
       State('graph_4', 'figure'),
       State('graph_5', 'figure'),
       State('num_left_to_match', 'children')]
     , prevent_initial_call=True)
def update_graphs(fn0, fn1, fn2, fn3, fn4, fn5, cd0, cd1, cd2, cd3, cd4, cd5, click_save, suggestion_mode, tiff_mode,
                  st0, st1, st2, st3, st4, st5, state_cells_left ):
    fnames = [fn0, fn1, fn2, fn3, fn4, fn5]
    cdata = [cd0, cd1, cd2, cd3, cd4, cd5]
    states = [st0, st1, st2, st3, st4, st5]
    updates = [no_update]*6
    new_sugg = False
    cells_left_message = state_cells_left
    ctx = dash.callback_context
    global cell_matching_df
    global templates; global orig_templates

    if ctx.triggered:
        global current_selected
        trigger_action = ctx.triggered[0]['prop_id'].split('.')[1]
        trigger_firstarg = ctx.triggered[0]['prop_id'].split('.')[0]

        # 1. SAVE BUTTON CLICKED
        if trigger_action == "n_clicks" and trigger_firstarg == "save_button":
            if not filenames[0] :    # no files uploaded and/or nothing highlighted
                cells_left_message = "upload files first"
            else:
                # get index, update cells of matched neurons and recompute best options for other neurons
                col = cell_matching_df[fnames[0].partition(".")[0]]
                row_idx = cell_matching_df[col == current_selected[0]].index.values[0]
                cell_matching_df.at[row_idx, "confirmed"] = 1
                for graph_idx, filename in enumerate(filenames):
                    if filename:
                        cell_matching_df.at[row_idx, filename] = current_selected[graph_idx]
                        if graph_idx: # not ref graph
                            confirmed_neurons = list(cell_matching_df.loc[cell_matching_df['confirmed'] == 1][filename])
                            for i in cell_matching_df.index:
                                if cell_matching_df.at[i, "confirmed"] == 0:
                                    allowed_ranking = [neur for neur in cell_matching_df.at[i, filename + "ranking"]
                                                       if neur not in confirmed_neurons]
                                    cell_matching_df.at[i, filename] =  allowed_ranking[0] if allowed_ranking else 0
                cell_matching_df.to_csv(UPLOAD_DIRECTORY + "result.csv", index = False, header=True, sep=';')
                print("csv saved @ ", UPLOAD_DIRECTORY + "result.csv")

                # output number of cells left to match
                num_cells_left =  sum(cell_matching_df["confirmed"] == 0)
                cells_left_message =  str(num_cells_left) + " cells left to confirm"
                # color neuron in graphs
                color = cell_matching_df.at[row_idx, "color"]
                for graph_idx, filename in enumerate(filenames):
                    print(graph_idx)
                    curr_neuron = current_selected[graph_idx];
                    if not filename or curr_neuron == -1:
                        continue
                    print(curr_neuron, color)
                    templates[graph_idx] = np.where(footprints[graph_idx][curr_neuron] >= 0.01, cols[color], templates[graph_idx])
                    updates[graph_idx] = update_graph(tiff_mode, filename, graph_idx, -1)
                print(cells_left_message)
                new_sugg = suggestion_mode

            
        # SUGGEST RANDOM NEURON MATCHING
        if (ctx.triggered[0]['prop_id'] == "suggestion_mode_button.active") or new_sugg:
            if ctx.triggered[0]['value'] or new_sugg:
                print("new neuron matching suggested")
                if isinstance(cell_matching_df, pd.DataFrame):
                    col = cell_matching_df["confirmed"]
                    row_idx = cell_matching_df[col == 0].index.values[0]
                    for graph_idx, filename in enumerate(fnames):
                        if not filename:
                            continue
                        print(filename, fnames)
                        curr_neuron = cell_matching_df.iloc[row_idx][filename.partition(".")[0]]
                        if isinstance(curr_neuron, list):
                            print("list of tuples ", curr_neuron )
                            curr_neuron = curr_neuron[0][1]
                        print(curr_neuron); current_selected[graph_idx] = curr_neuron
                        updates[graph_idx] = update_graph(tiff_mode, filename, graph_idx, curr_neuron)
                        print(int(coms[graph_idx][curr_neuron][0]), int(coms[graph_idx][curr_neuron][1]))

        #  IMAGE MODE CHANGED
        elif ctx.triggered[0]['prop_id'] == "tiff_mode_button.active":
            print("***callback: IMAGE MODE tiff CHANGED to: ", tiff_mode)
            for graph_idx, filename in enumerate(fnames):
                if not filename:
                    continue
                print(filename, fnames, current_selected[graph_idx])
                updates[graph_idx] = update_graph(tiff_mode, filename, graph_idx, current_selected[graph_idx])


        # 2 GRAPH CLICKED
        elif trigger_action == "clickData":
            trigger_graph_id = int(trigger_firstarg[-1])
            print("***callback: IMAGE CLICKED trigger_id: ", trigger_graph_id)
            clickData = cdata[trigger_graph_id]
            print(clickData)
            y = clickData['points'][0]['x']
            x = clickData['points'][0]['y']
            print("click on something",  x,y, fnames[trigger_graph_id], np.amax(pixel_ownerships[trigger_graph_id]))
            neuron_id = pixel_ownerships[trigger_graph_id][x,y]
            print("clicked on neuron: ", neuron_id, "trigger graph: ", trigger_graph_id)
            if neuron_id != -1 and neuron_id != current_selected[trigger_graph_id]:
                current_selected[trigger_graph_id] = neuron_id

                # reselection of matching -> update highlighted neuron (of a non-reference graph)
                if trigger_graph_id != 0:
                    updates[trigger_graph_id] = update_graph(tiff_mode, fnames[trigger_graph_id], trigger_graph_id, current_selected[trigger_graph_id])

                # new ref neuron clicked -> update highlighted neuron in ref and non-ref graphs
                if trigger_graph_id == 0 and not suggestion_mode:
                    print(list(cell_matching_df.columns), fnames[trigger_graph_id].partition(".")[0])
                    col = cell_matching_df[fnames[trigger_graph_id].partition(".")[0]]
                    idx_vals = cell_matching_df[col == neuron_id].index.values
                    print("row index is: " ,idx_vals)
                    if len(idx_vals) >= 1:
                        idx = idx_vals[0]
                    else:
                        print("error: neuron not in ref")
                    for graph_idx, filename in enumerate(fnames):
                        if not filename:
                            continue
                        print(filename, fnames)
                        curr_neuron = cell_matching_df.iloc[idx][filename.partition(".")[0]]
                        if isinstance(curr_neuron, list):
                            print("list of tuples ", curr_neuron )
                            curr_neuron = curr_neuron[0][1]
                        print(curr_neuron); current_selected[graph_idx] = curr_neuron
                        updates[graph_idx] = update_graph(tiff_mode, filename, graph_idx, curr_neuron)
                        print(int(coms[graph_idx][curr_neuron][0]), int(coms[graph_idx][curr_neuron][1]))

            print("click on nothing?")
        # 3 UPLOAD BUTTON
        elif trigger_action == 'filename':
            print("***callback: LOAD FILE")
            trigger_graph_id = int(trigger_firstarg[-1])
            print("loading ", fnames[trigger_graph_id])
            updates[trigger_graph_id] = format_fig(getdata_from_npyfile(fnames[trigger_graph_id], trigger_graph_id)[1],  title=fnames[trigger_graph_id].partition(".")[0])
            print("done loading file")
            for i in range(6):
                if i != trigger_graph_id:
                    updates[i] = no_update
        else:
            pass
    print(cells_left_message)
    return tuple(updates) + (cells_left_message,)



@app.callback(
    Output('compute_confirmation', 'children'),
    [Input('compute_matching_button', 'n_clicks')], prevent_initial_call=True)
def matching(n_clicks):
    if n_clicks is None:
        raise dash.exceptions.PreventUpdate
    print("***callback: MATCHING COMPUTED")
    message = "upload npy_files first"
    if filenames[0]:
        #match_neurons_across_session()
        num_neurons_ref = footprints[0].shape[0]
        neurons_ref = {filenames[0]: [i for i in range(0, num_neurons_ref)]}
        matching_df = pd.DataFrame(neurons_ref, columns = [filenames[0]])
        print("num_neurons_ref in compute matching callback: ", num_neurons_ref)
        for idx, filename in enumerate(filenames):
            if idx == 0 or not filename:
                continue
            #matching_df[filename] = [i for i in range(0, num_neurons_ref)]
            closest = int_func.match_neurons_to_ref(footprints[0], footprints[idx], coms[0], coms[idx],
                                 templates[0], templates[idx])
            print(type(closest), type(closest[0]), len(closest), len(matching_df))
            matching_df[filename + "ranking"]  = closest
            matching_df[filename]  = [row[0] for row in closest]
            #print(type([i for i in range(0, num_neurons_ref)]), len([i for i in range(0, num_neurons_ref)]),
            #      type(list(closest[0])), len(list(closest[0])))

        matching_df["color"] = np.arange(len(matching_df)) % 5 + 1
        matching_df["confirmed"] = 0
        global cell_matching_df; cell_matching_df = matching_df
        message = "matching computed"
        print(type(cell_matching_df))
        cell_matching_df.to_csv(UPLOAD_DIRECTORY + "result.csv", index = False, header=True, sep=';')
    return message


# UPLOAD CSV

@app.callback(
    Output('upload_confirmation', 'children'),
    [Input('upload_csv_matching', 'filename')], prevent_initial_call=True)
def uploading_csv(filename):
    if not filename:
        raise dash.exceptions.PreventUpdate
    print("***callback: MATCHING COMPUTED: ", dash.callback_context.triggered)
    df = pd.read_csv(UPLOAD_DIRECTORY + filename, sep=';')
    # check if column for each uploaded session in csv
    loaded_filenames = [file for file in filenames if file]
    print("loaded filenames: ", loaded_filenames, " df.columns ", df.columns)
    all_sessions_in_csv = all(elem in df.columns for elem in loaded_filenames)
    print("all_sessions_in_csv ", all_sessions_in_csv)
    print(type(list(df.columns)), type(loaded_filenames))
    print(df.columns[0])
    has_correct_ref_session = list(df.columns) and loaded_filenames and loaded_filenames[0] == df.columns[0]
    print("has_correct_ref_session ", has_correct_ref_session)
    if not loaded_filenames:
        message = "load sessions first!"
    elif not all_sessions_in_csv:
        message = "invalid csv (col per session!)"
    elif not has_correct_ref_session:
        message = "ref sess msut be first col in csv"
    else:
        global cell_matching_df; cell_matching_df = df
        message = filename + " uploaded."
    return message


### keep buttons highlighted when clicked

# selection/suggestion buttons

@app.callback(
    [Output('selection_mode_button', 'active'), Output('suggestion_mode_button',  'active')],
    [Input('selection_mode_button',  'n_clicks'), Input('suggestion_mode_button',  'n_clicks')],
    [State('selection_mode_button',  'active'), State('suggestion_mode_button',  'active')],
    prevent_initial_call=True)
def update_selection_mode(c_sel, c_sug, state_sel, state_sug):
    if all(click is None for click in [c_sel, c_sug]):
        raise dash.exceptions.PreventUpdate
    print("***callback: UPDATE SELECTION MODE")
    states = [no_update]*2
    ctx = dash.callback_context
    if ctx.triggered:
        if((ctx.triggered[0]['prop_id'] == "suggestion_mode_button.n_clicks" and not state_sug)
            or (ctx.triggered[0]['prop_id'] == "selection_mode_button.n_clicks" and not state_sel)):
            states = [not state_sel, not state_sug]
    print("sel/sug states: ", states)
    return states

# basic/mean_intensity_mode button

@app.callback(
    [Output('basic_mode_button', 'active'), Output('tiff_mode_button',  'active')],
    [Input('basic_mode_button',  'n_clicks'), Input('tiff_mode_button',  'n_clicks')],
    [State('basic_mode_button',  'active'), State('tiff_mode_button',  'active')],
    prevent_initial_call=True)
def update_image_mode(c_basic, c_tiff, state_basic, state_tiff):
    if all(click is None for click in [c_basic, c_tiff]):
        raise dash.exceptions.PreventUpdate
    print("***callback: UPDATE IMAGE MODE")
    states = [no_update]*2
    ctx = dash.callback_context
    if ctx.triggered:
        if((ctx.triggered[0]['prop_id'] == "basic_mode_button.n_clicks" and not state_basic)
            or (ctx.triggered[0]['prop_id'] == "tiff_mode_button.n_clicks" and not state_tiff)):
            states = [not state_basic, not state_tiff]
    print(states)
    return states

### no match buttons

@app.callback(
    [Output('no_match_2', 'active'), Output('no_match_3', 'active'), Output('no_match_4', 'active'),
     Output('no_match_5', 'active'), Output('no_match_6', 'active'), ],
    [Input('no_match_2', 'n_clicks'), Input('no_match_3', 'n_clicks'), Input('no_match_4', 'n_clicks'),
    Input('no_match_5', 'n_clicks'), Input('no_match_6', 'n_clicks')],
    [State('no_match_2', 'active'), State('no_match_3', 'active'),
     State('no_match_4', 'active'), State('no_match_5', 'active'), State('no_match_6', 'active')],
    prevent_initial_call=True)
def update_no_match_buttons(click2, click3, click4, click5, click6, state2, state3, state4, state5, state6):
    clicks = [click2, click3, click4, click5, click6]
    if all(click is None for click in clicks) :
        raise dash.exceptions.PreventUpdate
    print("***callback: UPDATE MATCH BUTTONS: ", dash.callback_context.triggered)
    states = [state2, state3, state4, state5, state6]
    ctx = dash.callback_context
    if ctx.triggered:
        print(ctx.triggered[0]["prop_id"].split(".")[0][-1])
        triggered_id = int(ctx.triggered[0]["prop_id"].split(".")[0][-1])
        print(states[triggered_id-2])
        states[triggered_id-2] = not states[triggered_id-2]
        global current_selected;
        current_selected[triggered_id-1] = -1
        print(current_selected)
    return states


if __name__ == '__main__':
    app.run_server(debug=True, threaded=True)
    #app.run_server(debug=False)

