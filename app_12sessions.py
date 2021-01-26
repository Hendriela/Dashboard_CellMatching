import dash
from dash import Dash, no_update
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import json
import pickle
import internal_functions as int_func
from content_layout import select_controls_12sessions, controls_12sessions

# **********************************************************************************************************************
#               USER-SPECIFIC SETTINGS
# **********************************************************************************************************************

# UPLOAD_DIRECTORY = "W:\\Neurophysiology-Storage1\\Wahl\\Anna\\"
UPLOAD_DIRECTORY = "/home/anna/Documents/Neuron_imaging_validationsets/"
filename_result_csv = 'result_matching.csv'

# True -> if the classifier predicts "no neuron", the closest neuron is shown
# False -> no neuron selected
always_predict_match = True

# portion of original FoV that will be shown when a neuron is selected (0.5 = zoom to half of FoV, 1 = no zoom)
zoom_ratio = 0.5

# colours of the matched neurons
cols = [1.5, 2.2, 2.8, 3.2, 3.8, 4.5]

# classifier settings
classifier_path = 'pickle_model.pkl'
num_features = 12  # needs to match the loaded clf!
num_neurs_clf_input = 3  # needs to match the loaded clf!

# **********************************************************************************************************************
#               INITIALIZATION
# **********************************************************************************************************************


# initialization of global variables
num_plots_max = 12

cell_matching_df = []
filenames = [''] * num_plots_max
sessions_data = [None] * num_plots_max
footprints = [None] * num_plots_max
orig_templates = [None] * num_plots_max
templates = [None] * num_plots_max
templates_tiff = [None] * num_plots_max
coms = [None] * num_plots_max
current_selected = [-1] * num_plots_max
pixel_ownerships = [np.full((512, 512), -1)] * num_plots_max
empty_template = [[0] * 512] * 512
figs = [int_func.format_fig(empty_template)] * num_plots_max
clf = pickle.load(open(classifier_path, 'rb'))

# **********************************************************************************************************************
#               DASHBOARD
# **********************************************************************************************************************

print("the app reset, all data must be reloaded!")
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# **********************************************************************************************************************
#               LAYOUT (see content_layout.py for more details)
# **********************************************************************************************************************


app.layout = dbc.Container(
    [
        dbc.Row(dbc.Col(html.Div("CELL MATCHING ACROSS SESSIONS"), width=12), style={'padding': 10}),
        dbc.Row(
            [dbc.Col(dcc.Graph(
                id='graph_0',
                figure=figs[0],
                ), width=2),
                dbc.Col(dcc.Graph(
                    id='graph_1',
                    figure=figs[1]
                ), width=2),
                dbc.Col(dcc.Graph(
                    id='graph_2',
                    figure=figs[2]
                ), width=2),
                dbc.Col(dcc.Graph(
                    id='graph_3',
                    figure=figs[3]
                ), width=2),
                dbc.Col(controls_12sessions, width={"size": 3, "offset": 1}),
            ]
        ),
        dbc.Row(
            [dbc.Col(dcc.Graph(
                id='graph_4',
                figure=figs[4]
                ), width=2.),
                dbc.Col(dcc.Graph(
                    id='graph_5',
                    figure=figs[5]
                ), width=2),
                dbc.Col(dcc.Graph(
                    id='graph_6',
                    figure=figs[6]
                ), width=2),
                dbc.Col(dcc.Graph(
                    id='graph_7',
                    figure=figs[7]
                ), width=2),
                dbc.Col(select_controls_12sessions, width={"size": 3, "offset": 1 }), #"order": "last", "offset": 3
            ]
        ),
        dbc.Row(
            [dbc.Col(dcc.Graph(
                id='graph_8',
                figure=figs[8]
                ), width=2),
                dbc.Col(dcc.Graph(
                    id='graph_9',
                    figure=figs[9]
                ), width=2),
                dbc.Col(dcc.Graph(
                    id='graph_10',
                    figure=figs[10]
                ), width=2),
                dbc.Col(dcc.Graph(
                    id='graph_11',
                    figure=figs[11]
                ), width=2)
            ]
        ),
    ],
    fluid=True
)


# **********************************************************************************************************************
#               HELPER FUNCTIONS
# **********************************************************************************************************************

def getdata_from_npyfile(filename, graph_id):
    """ Load the data from the npy file of the session data (obtained by running prepare_data_dashboard.py) """
    print("click", UPLOAD_DIRECTORY + filename)
    session_data = np.load(UPLOAD_DIRECTORY + "\\" + filename, allow_pickle=True).item()
    print("loading...", UPLOAD_DIRECTORY + filename)
    templates[graph_id] = session_data['template']
    orig_templates[graph_id] = session_data['template']

    print("basic template loaded")
    templates_tiff[graph_id] = session_data['mean_intensity_template']
    print("templates loaded")
    num_neurons = session_data['dff_trace'].shape[0]
    footprints[graph_id] = np.reshape(session_data['spatial_masks'].toarray(),
                                      (session_data['template'].shape[0], session_data['template'].shape[1],
                                       num_neurons),
                                      order='F')  #
    footprints[graph_id] = np.transpose(footprints[graph_id], (2, 0, 1))
    print("transposed footprints")
    pixel_ownerships[graph_id] = int_func.pixel_neuron_ownership(footprints[graph_id])
    coms[graph_id] = int_func.compute_CoMs(footprints[graph_id])
    # update template
    for com in coms[graph_id]:
        templates[graph_id][int(com[0])][int(com[1])] = 255

    filenames[graph_id] = filename.partition(".")[0]
    print(type(templates), type(footprints), type(pixel_ownerships))
    return orig_templates[graph_id], templates[graph_id], footprints[graph_id], pixel_ownerships[graph_id]


def update_graph(tiff_mode, filename, graph_idx, curr_neuron):
    background = templates_tiff[graph_idx] if tiff_mode else templates[graph_idx]
    if curr_neuron == -1:
        new_graph = int_func.format_fig(background, title=filename.partition(".")[0], zoom=False,
                                        is_tiff_mode=tiff_mode)
    else:
        if not tiff_mode:
            background = np.where(np.logical_and(footprints[graph_idx][curr_neuron] >= 0.01, background != 255),
                                  # to keep com dot
                                  cols[0], background)
        new_graph = int_func.format_fig(background, title=filename.partition(".")[0], zoom=True, zoom_ratio=zoom_ratio,
                                        center_coords_x=int(coms[graph_idx][curr_neuron][1]),
                                        center_coords_y=int(coms[graph_idx][curr_neuron][0]),
                                        is_tiff_mode=tiff_mode)
    return new_graph


def get_predicted_neuron(clf, clf_input):
    features_flat_list = [item for sublist in clf_input for item in sublist]
    print(features_flat_list)
    prediction = clf.predict([features_flat_list])[0]
    print('pred ', prediction)
    curr_neuron = -1
    if prediction in [0, 1, 2]:
        curr_neuron = features_flat_list[prediction * num_features]
    elif always_predict_match and prediction == num_neurs_clf_input:
        curr_neuron = features_flat_list[0]
    print('curr_neuron ', curr_neuron)
    return curr_neuron


# **********************************************************************************************************************
#               CALLBACKS (interactive functionality)
# **********************************************************************************************************************

# main callback (suggest neuron, click on neuron, save match)

@app.callback([Output('graph_0', 'figure'),
               Output('graph_1', 'figure'),
               Output('graph_2', 'figure'),
               Output('graph_3', 'figure'),
               Output('graph_4', 'figure'),
               Output('graph_5', 'figure'),
               Output('graph_6', 'figure'),
               Output('graph_7', 'figure'),
               Output('graph_8', 'figure'),
               Output('graph_9', 'figure'),
               Output('graph_10', 'figure'),
               Output('graph_11', 'figure'),
               Output('num_left_to_match', 'children')
               ],
              [Input('upload_session_0', 'filename'),
               Input('upload_session_1', 'filename'),
               Input('upload_session_2', 'filename'),
               Input('upload_session_3', 'filename'),
               Input('upload_session_4', 'filename'),
               Input('upload_session_5', 'filename'),
               Input('upload_session_6', 'filename'),
               Input('upload_session_7', 'filename'),
               Input('upload_session_8', 'filename'),
               Input('upload_session_9', 'filename'),
               Input('upload_session_10', 'filename'),
               Input('upload_session_11', 'filename'),
               Input('graph_0', 'clickData'),
               Input('graph_1', 'clickData'),
               Input('graph_2', 'clickData'),
               Input('graph_3', 'clickData'),
               Input('graph_4', 'clickData'),
               Input('graph_5', 'clickData'),
               Input('graph_6', 'clickData'),
               Input('graph_7', 'clickData'),
               Input('graph_8', 'clickData'),
               Input('graph_9', 'clickData'),
               Input('graph_10', 'clickData'),
               Input('graph_11', 'clickData'),
               Input('save_button', 'n_clicks'),
               Input('suggestion_mode_button', 'active'),
               Input('tiff_mode_button', 'active')],
              [State('graph_0', 'figure'),
               State('graph_1', 'figure'),
               State('graph_2', 'figure'),
               State('graph_3', 'figure'),
               State('graph_4', 'figure'),
               State('graph_5', 'figure'),
               State('graph_6', 'figure'),
               State('graph_7', 'figure'),
               State('graph_8', 'figure'),
               State('graph_9', 'figure'),
               State('graph_10', 'figure'),
               State('graph_11', 'figure'),
               State('num_left_to_match', 'children')],
              prevent_initial_call=True)
def update_graphs(fn0, fn1, fn2, fn3, fn4, fn5, fn6, fn7, fn8, fn9, fn10, fn11,
                  cd0, cd1, cd2, cd3, cd4, cd5, cd6, cd7, cd8, cd9, cd10, cd11,
                  click_save, suggestion_mode, tiff_mode,
                  st0, st1, st2, st3, st4, st5, st6, st7, st8, st9, st10, st11,
                  state_cells_left):
    fnames = [fn0, fn1, fn2, fn3, fn4, fn5, fn6, fn7, fn8, fn9, fn10, fn11]
    cdata = [cd0, cd1, cd2, cd3, cd4, cd5, cd6, cd7, cd8, cd9, cd10, cd11]
    states = [st0, st1, st2, st3, st4, st5, st6, st7, st8, st9, st10, st11]
    updates = [no_update] * num_plots_max
    new_sugg = False
    cells_left_message = state_cells_left
    ctx = dash.callback_context
    global cell_matching_df
    global templates
    global orig_templates

    if ctx.triggered:
        global current_selected
        trigger_action = ctx.triggered[0]['prop_id'].split('.')[1]
        trigger_firstarg = ctx.triggered[0]['prop_id'].split('.')[0]

        # 1. SAVE BUTTON CLICKED
        if trigger_action == "n_clicks" and trigger_firstarg == "save_button":
            print("***callback: SAVE")
            if not filenames[0]:  # no files uploaded and/or nothing highlighted
                cells_left_message = "upload files first"
            else:
                # get index, update cells of matched neurons and recompute best options for other neurons
                col = cell_matching_df[fnames[0].partition(".")[0]]
                row_idx = cell_matching_df[col == current_selected[0]].index.values[0]
                cell_matching_df.at[row_idx, "confirmed"] = 1
                for graph_idx, filename in enumerate(filenames):
                    if filename:
                        cell_matching_df.at[row_idx, filename] = current_selected[graph_idx]
                        if graph_idx:  # not ref graph
                            confirmed_neurons = list(cell_matching_df.loc[cell_matching_df['confirmed'] == 1][filename])
                            print("confirmed_neurons: ", confirmed_neurons)
                            """
                            for i in cell_matching_df.index:
                                print(type(json.loads(cell_matching_df.at[i, filename + "ranking"])))

                                if cell_matching_df.at[i, "confirmed"] == 0:
                                    allowed_ranking = [neuron for neuron in
                                                       json.loads(cell_matching_df.at[i, filename + "ranking"])
                                                       if neuron not in confirmed_neurons]

                                    cell_matching_df.at[i, filename] = allowed_ranking[0] if allowed_ranking else 0
                                """
                cell_matching_df.to_csv(UPLOAD_DIRECTORY + "\\" + filename_result_csv, index=False, header=True, sep=';')
                print("csv saved @ ", UPLOAD_DIRECTORY + "\\" + filename_result_csv)

                # output number of cells left to match
                num_cells_left = sum(cell_matching_df["confirmed"] == 0)
                cells_left_message = str(num_cells_left) + " cells left to confirm"
                # color neuron in graphs
                color = cell_matching_df.at[row_idx, "color"]
                for graph_idx, filename in enumerate(filenames):
                    print(graph_idx)
                    curr_neuron = current_selected[graph_idx]
                    if not filename or curr_neuron == -1:
                        continue
                    print(curr_neuron, color)
                    templates[graph_idx] = np.where(
                        np.logical_and(footprints[graph_idx][curr_neuron] >= 0.01, templates[graph_idx] != 255),
                        cols[color], templates[graph_idx])
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
                        if graph_idx > 0:
                            clf_input = cell_matching_df.iloc[row_idx][filename.partition(".")[0] + 'feature_vals']
                            if not isinstance(clf_input, list):
                                clf_input = json.loads(clf_input)
                            print('clf_input ', clf_input)
                            curr_neuron = get_predicted_neuron(clf, clf_input[:3])
                        if isinstance(curr_neuron, list):
                            print("list of tuples ", curr_neuron)
                            curr_neuron = curr_neuron[0][1]
                        print(curr_neuron);
                        current_selected[graph_idx] = curr_neuron
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
            click_data = cdata[trigger_graph_id]
            print(click_data)
            y = click_data['points'][0]['x']
            x = click_data['points'][0]['y']
            print("click on something", x, y, fnames[trigger_graph_id], np.amax(pixel_ownerships[trigger_graph_id]))
            neuron_id = pixel_ownerships[trigger_graph_id][x, y]
            print("clicked on neuron: ", neuron_id, "trigger graph: ", trigger_graph_id)
            if neuron_id != -1 and neuron_id != current_selected[trigger_graph_id]:
                current_selected[trigger_graph_id] = neuron_id

                # reselection of matching -> update highlighted neuron (of a non-reference graph)
                if trigger_graph_id != 0:
                    print("newly selected: ", current_selected[trigger_graph_id])
                    updates[trigger_graph_id] = update_graph(tiff_mode, fnames[trigger_graph_id], trigger_graph_id,
                                                             current_selected[trigger_graph_id])

                # new ref neuron clicked -> update highlighted neuron in ref and non-ref graphs
                if trigger_graph_id == 0 and not suggestion_mode:
                    print(list(cell_matching_df.columns), fnames[trigger_graph_id].partition(".")[0])
                    col = cell_matching_df[fnames[trigger_graph_id].partition(".")[0]]
                    row_idx = cell_matching_df[col == neuron_id].index.values[0]
                    print("row index is: ", row_idx)
                    for graph_idx, filename in enumerate(fnames):
                        if not filename:
                            continue
                        print(filename, fnames)
                        curr_neuron = cell_matching_df.iloc[row_idx][filename.partition(".")[0]]
                        if graph_idx > 0:
                            clf_input = cell_matching_df.iloc[row_idx][filename.partition(".")[0] + 'feature_vals']
                            if not isinstance(clf_input, list):
                                clf_input = json.loads(clf_input)
                            print('clf_input ', clf_input)
                            curr_neuron = get_predicted_neuron(clf, clf_input[:3])
                        print(curr_neuron)
                        current_selected[graph_idx] = curr_neuron
                        updates[graph_idx] = update_graph(tiff_mode, filename, graph_idx, curr_neuron)
                        print(int(coms[graph_idx][curr_neuron][0]), int(coms[graph_idx][curr_neuron][1]))

            print("click on nothing?")
        # 3 UPLOAD BUTTON
        elif trigger_action == 'filename':
            print("***callback: LOAD FILE")
            print(ctx.triggered)
            trigger_graph_id = int(trigger_firstarg.rsplit('_', 1)[-1])
            print("loading ", fnames[trigger_graph_id])
            updates[trigger_graph_id] = int_func.format_fig(
                getdata_from_npyfile(fnames[trigger_graph_id], trigger_graph_id)[1],
                title=fnames[trigger_graph_id].partition(".")[0])
            print("done loading file")
            for i in range(num_plots_max):
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
    print("***callback: COMPUTING MATCHING ")
    message = "upload npy_files first"
    if filenames[0]:
        # match_neurons_across_session()
        num_neurons_ref = footprints[0].shape[0]
        neurons_ref = {filenames[0]: [i for i in range(0, num_neurons_ref)]}
        # if only place cells: neurons_ref = {filenames[0]: [i for i in place_cells_ref]}
        matching_df = pd.DataFrame(neurons_ref, columns=[filenames[0]])
        print("num_neurons_ref in compute matching callback: ", num_neurons_ref)
        for idx, filename in enumerate(filenames):
            if idx == 0 or not filename:
                continue
            # matching_df[filename] = [i for i in range(0, num_neurons_ref)]
            closest, feature_vals = int_func.match_neurons_to_ref(footprints[0], footprints[idx], coms[0], coms[idx],
                                                                  templates[0], templates[idx])
            # print(type(closest), type(closest[0]), len(closest), len(matching_df))
            matching_df[filename + "ranking"] = closest
            matching_df[filename] = [row[0] for row in closest]
            matching_df[filename + "feature_vals"] = feature_vals
            # print(type([i for i in range(0, num_neurons_ref)]), len([i for i in range(0, num_neurons_ref)]),
            #      type(list(closest[0])), len(list(closest[0])))

        matching_df["color"] = np.arange(len(matching_df)) % 5 + 1
        matching_df["confirmed"] = 0
        global cell_matching_df;
        cell_matching_df = matching_df
        message = "matching computed"
        cell_matching_df.to_csv(UPLOAD_DIRECTORY + filename_result_csv, index=False, header=True, sep=';')
    return message


# UPLOAD CSV

@app.callback(
    Output('upload_confirmation', 'children'),
    [Input('upload_csv_matching', 'filename')], prevent_initial_call=True)
def uploading_csv(filename):
    if not filename:
        raise dash.exceptions.PreventUpdate
    print("***callback: MATCHING COMPUTED: ", dash.callback_context.triggered)
    df = pd.read_csv(UPLOAD_DIRECTORY + "\\" + filename, sep=';')
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
        message = "ref sess must be first col in csv"
    else:
        global cell_matching_df;
        cell_matching_df = df

        # update coloured templates:
        global templates;
        confirmed_neurons_row_idx = list(cell_matching_df.loc[cell_matching_df['confirmed'] == 1][filenames[0]])
        print("confirmed_neurons_ref", confirmed_neurons_row_idx)
        for row_idx in confirmed_neurons_row_idx:
            color = cell_matching_df.at[row_idx, "color"]
            print("color: ", color)
            for graph_idx, file in enumerate(filenames):
                if not file:
                    continue
                neuron = cell_matching_df.at[row_idx, file]
                print(graph_idx)
                print(type(footprints), type(cols), type(templates[graph_idx]))
                templates[graph_idx] = np.where(np.logical_and(footprints[graph_idx][neuron] >= 0.01,
                                                               templates[graph_idx] != 255),  # to keep com dot
                                                cols[color], templates[graph_idx])

        message = filename + " uploaded."
    return message


### keep buttons highlighted when clicked

# selection/suggestion buttons

@app.callback(
    [Output('selection_mode_button', 'active'), Output('suggestion_mode_button', 'active')],
    [Input('selection_mode_button', 'n_clicks'), Input('suggestion_mode_button', 'n_clicks')],
    [State('selection_mode_button', 'active'), State('suggestion_mode_button', 'active')],
    prevent_initial_call=True)
def update_selection_mode(c_sel, c_sug, state_sel, state_sug):
    if all(click is None for click in [c_sel, c_sug]):
        raise dash.exceptions.PreventUpdate
    print("***callback: UPDATE SELECTION MODE")
    states = [no_update] * 2
    ctx = dash.callback_context
    if ctx.triggered:
        if ((ctx.triggered[0]['prop_id'] == "suggestion_mode_button.n_clicks" and not state_sug)
                or (ctx.triggered[0]['prop_id'] == "selection_mode_button.n_clicks" and not state_sel)):
            states = [not state_sel, not state_sug]
    print("sel/sug states: ", states)
    return states


# basic/mean_intensity_mode button

@app.callback(
    [Output('basic_mode_button', 'active'), Output('tiff_mode_button', 'active')],
    [Input('basic_mode_button', 'n_clicks'), Input('tiff_mode_button', 'n_clicks')],
    [State('basic_mode_button', 'active'), State('tiff_mode_button', 'active')],
    prevent_initial_call=True)
def update_image_mode(c_basic, c_tiff, state_basic, state_tiff):
    if all(click is None for click in [c_basic, c_tiff]):
        raise dash.exceptions.PreventUpdate
    print("***callback: UPDATE IMAGE MODE")
    states = [no_update] * 2
    ctx = dash.callback_context
    if ctx.triggered:
        if ((ctx.triggered[0]['prop_id'] == "basic_mode_button.n_clicks" and not state_basic)
                or (ctx.triggered[0]['prop_id'] == "tiff_mode_button.n_clicks" and not state_tiff)):
            states = [not state_basic, not state_tiff]
    print(states)
    return states


### no match buttons

@app.callback(
    [Output('no_match_2', 'active'), Output('no_match_3', 'active'), Output('no_match_4', 'active'),
     Output('no_match_5', 'active'), Output('no_match_6', 'active'),
     Output('no_match_7', 'active'), Output('no_match_8', 'active'), Output('no_match_9', 'active'),
     Output('no_match_10', 'active'), Output('no_match_11', 'active'), Output('no_match_12', 'active')],
    [Input('no_match_2', 'n_clicks'), Input('no_match_3', 'n_clicks'), Input('no_match_4', 'n_clicks'),
     Input('no_match_5', 'n_clicks'), Input('no_match_6', 'n_clicks'),
     Input('no_match_7', 'n_clicks'), Input('no_match_8', 'n_clicks'), Input('no_match_9', 'n_clicks'),
     Input('no_match_10', 'n_clicks'), Input('no_match_11', 'n_clicks'),  Input('no_match_12', 'n_clicks')],
    [State('no_match_2', 'active'), State('no_match_3', 'active'),
     State('no_match_4', 'active'), State('no_match_5', 'active'), State('no_match_6', 'active'),
     State('no_match_7', 'active'), State('no_match_8', 'active'), State('no_match_9', 'active'),
     State('no_match_10', 'active'), State('no_match_11', 'active'), State('no_match_12', 'active')],
    prevent_initial_call=True)
def update_no_match_buttons(click2, click3, click4, click5, click6,
                            click7, click8, click9, click10, click11, click12,
                            state2, state3, state4, state5, state6,
                            state7, state8, state9, state10, state11, state12):
    clicks = [click2, click3, click4, click5, click6, click7, click8, click9, click10, click11, click12]
    if all(click is None for click in clicks):
        raise dash.exceptions.PreventUpdate
    print("***callback: UPDATE MATCH BUTTONS: ", dash.callback_context.triggered)
    states = [state2, state3, state4, state5, state6, state7, state8, state9, state10, state11, state12]
    ctx = dash.callback_context
    if ctx.triggered:
        print(ctx.triggered[0]["prop_id"].split(".")[0].rsplit('_', 1)[-1])
        triggered_id = int(ctx.triggered[0]["prop_id"].split(".")[0].rsplit('_', 1)[-1])
        print(states[triggered_id - 2])
        states[triggered_id - 2] = not states[triggered_id - 2]
        global current_selected;
        current_selected[triggered_id - 1] = -1
        print(current_selected)
    return states


if __name__ == '__main__':
    app.run_server(debug=True, threaded=True)
    # app.run_server(debug=False)
