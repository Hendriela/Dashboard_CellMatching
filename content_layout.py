import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html


# top right controls (upload buttons)

controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("1. Upload npy files of the sessions"),
                # dbc.Spinner(html.Div(id="loading-output")),

                dbc.Row([
                    dcc.Upload(
                        id="upload_session_0",
                        children=html.Div(
                            dbc.Button('Upload 1', size="sm", outline=True, color="secondary"),
                            # dbc.Button([dbc.Spinner(size="sm"), "Loading..."],'Upload 1', size="sm",  outline=True, color="secondary"),
                        ),
                        multiple=False,
                    ),
                    dcc.Upload(
                        id="upload_session_1",
                        children=html.Div(
                            dbc.Button('Upload 2', size="sm", outline=True, color="secondary"),
                        ),
                        multiple=False,
                    ),
                    dcc.Upload(
                        id="upload_session_2",
                        children=html.Div(
                            dbc.Button('Upload 3', size="sm", outline=True, color="secondary"),
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
                                dbc.Button('Upload 4', size="sm", outline=True, color="secondary"),
                            ),
                            multiple=False,
                        ),
                        dcc.Upload(
                            id="upload_session_4",
                            children=html.Div(
                                dbc.Button('Upload 5', size="sm", outline=True, color="secondary"),
                            ),
                            multiple=False,
                        ),
                        dcc.Upload(
                            id="upload_session_5",
                            children=html.Div(
                                dbc.Button('Upload 6', size="sm", outline=True, color="secondary"),
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
                            id="upload_csv_matching", style={"visibility": "visible"},
                            children=html.Div(
                                dbc.Button('Upload csv', size="sm", outline=True, color="secondary",
                                           style={"visibility": "visible"}),

                            ),
                            multiple=False,
                        ),
                        dbc.Button("Compute matchings", id="compute_matching_button", size="sm", outline=True,
                                   active=False, color="secondary"),
                        html.Span(id="upload_confirmation", style={"vertical-align": "middle"}),
                        html.Span(id="compute_confirmation", style={"vertical-align": "middle"})
                    ])
                )
            ]),

    ],
    body=True,
)

# bottom right controls (selection mode, display mode, no match, save buttons)

select_controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("3. Choose selection mode"),

                dbc.Row(
                    [html.Div([
                        # html.Pre(id='placeholder', style={"display": "none"}) ,
                        # html.Pre(False, id='selection_mode', style={"display": "none"}) ,
                        dbc.Button("Selection mode", id="selection_mode_button", size="sm", outline=True, active=True,
                                   color="secondary"),
                        dbc.Button("Suggestion mode", id="suggestion_mode_button", size="sm", outline=True,
                                   active=False, color="secondary"),
                    ])
                    ]
                ),
            ]),
        dbc.FormGroup(
            [
                dbc.Label("4. Display mode"),

                dbc.Row(
                    [html.Div([
                        # html.Pre(id='placeholder', style={"display": "none"}) ,
                        # html.Pre(False, id='selection_mode', style={"display": "none"}) ,
                        dbc.Button("Basic image", id="basic_mode_button", size="sm", outline=True, active=True,
                                   color="secondary"),
                        dbc.Button("Mean intensity image", id="tiff_mode_button", size="sm", outline=True, active=False,
                                   color="secondary"),
                    ])
                    ]
                ),
            ]),
        dbc.FormGroup(
            [
                dbc.Label("5. Indicate if neuron does not exsit"),
                dbc.Row(
                    [html.Div([
                        dbc.Button("No match 1", id="no_match_1", size="sm", style={"visibility": "hidden"},
                                   outline=True, color="secondary"),
                        dbc.Button("No match 2", id="no_match_2", size="sm", outline=True, color="secondary",
                                   active=False),
                        dbc.Button("No match 3", id="no_match_3", size="sm", outline=True, color="secondary",
                                   active=False),
                    ])
                    ]
                ),
                dbc.Row(
                    [html.Div([
                        dbc.Button("No match 4", id="no_match_4", size="sm", outline=True, color="secondary",
                                   active=False),
                        dbc.Button("No match 5", id="no_match_5", size="sm", outline=True, color="secondary",
                                   active=False),
                        dbc.Button("No match 6", id="no_match_6", size="sm", outline=True, color="secondary",
                                   active=False),
                    ])
                    ]
                ),
            ]),
        dbc.FormGroup([
            dbc.Label("6. Click save after each matched neuron"),
            dbc.Row(
                [
                    html.Div([
                        dbc.Button("Save", id="save_button", block=True, size="sm", outline=True, active=True,
                                   color="primary"),
                        html.Pre("Number of neurons to be matched", id='num_left_to_match')
                    ])
                ]
            )
        ])
    ],
    body=True,
)




controls_16sessions = dbc.FormGroup(
            [
                dbc.Label("1. Upload npy files of the sessions"),
                # dbc.Spinner(html.Div(id="loading-output")),

                dbc.Row([
                    dcc.Upload(
                        id="upload_session_0",
                        children=html.Div(
                            dbc.Button(' Upload 01', size="sm", outline=True, color="secondary"),
                            # dbc.Button([dbc.Spinner(size="sm"), "Loading..."],'Upload 1', size="sm",  outline=True, color="secondary"),
                        ),
                        multiple=False,
                    ),
                    dcc.Upload(
                        id="upload_session_1",
                        children=html.Div(
                            dbc.Button("Upload " + "02", size="sm", outline=True, color="secondary"),
                        ),
                        multiple=False,
                    ),
                    dcc.Upload(
                        id="upload_session_2",
                        children=html.Div(
                            dbc.Button('Upload 03', size="sm", outline=True, color="secondary"),
                        ),
                        multiple=False,
                    ),
                    dcc.Upload(
                        id="upload_session_3",
                        children=html.Div(
                            dbc.Button('Upload 04', size="sm", outline=True, color="secondary"),
                        ),
                        multiple=False,
                    ),
                ]
                ),
                dbc.Row([
                        dcc.Upload(
                            id="upload_session_4",
                            children=html.Div(
                                dbc.Button('Upload 05', size="sm", outline=True, color="secondary"),
                            ),
                            multiple=False,
                        ),
                        dcc.Upload(
                            id="upload_session_5",
                            children=html.Div(
                                dbc.Button('Upload 06', size="sm", outline=True, color="secondary"),
                            ),
                            multiple=False,
                        ),
                        dcc.Upload(
                            id="upload_session_6",
                            children=html.Div(
                                dbc.Button('Upload 07', size="sm", outline=True, color="secondary"),
                            ),
                            multiple=False,
                        ),
                        dcc.Upload(
                            id="upload_session_7",
                            children=html.Div(
                                dbc.Button('Upload 08', size="sm", outline=True, color="secondary"),
                            ),
                            multiple=False,
                        ),
                    ]
                ),
                dbc.Row([
                        dcc.Upload(
                            id="upload_session_8",
                            children=html.Div(
                                dbc.Button('Upload 09', size="sm", outline=True, color="secondary"),
                            ),
                            multiple=False,
                        ),
                        dcc.Upload(
                            id="upload_session_9",
                            children=html.Div(
                                dbc.Button('Upload 10', size="sm", outline=True, color="secondary"),
                            ),
                            multiple=False,
                        ),
                        dcc.Upload(
                            id="upload_session_10",
                            children=html.Div(
                                dbc.Button('Upload 11', size="sm", outline=True, color="secondary"),
                            ),
                            multiple=False,
                        ),
                        dcc.Upload(
                            id="upload_session_11",
                            children=html.Div(
                                dbc.Button('Upload 12', size="sm", outline=True, color="secondary"),
                            ),
                            multiple=False,
                        ),
                    ]
                ),
                dbc.Row([
                    dcc.Upload(
                        id="upload_session_12",
                        children=html.Div(
                            dbc.Button('Upload 13', size="sm", outline=True, color="secondary"),
                        ),
                        multiple=False,
                    ),
                    dcc.Upload(
                        id="upload_session_13",
                        children=html.Div(
                            dbc.Button('Upload 14', size="sm", outline=True, color="secondary"),
                        ),
                        multiple=False,
                    ),
                    dcc.Upload(
                        id="upload_session_14",
                        children=html.Div(
                            dbc.Button('Upload 15', size="sm", outline=True, color="secondary"),
                        ),
                        multiple=False,
                    ),
                    dcc.Upload(
                        id="upload_session_15",
                        children=html.Div(
                            dbc.Button('Upload 16', size="sm", outline=True, color="secondary"),
                        ),
                        multiple=False,
                    ),
                ]
                ),
                dbc.Label("2. Upload/Create neuron matching file"),
                dbc.Row(
                    html.Div([
                        dcc.Upload(
                            id="upload_csv_matching", style={"visibility": "visible"},
                            children=html.Div(
                                dbc.Button('Upload csv', size="sm", outline=True, color="secondary",
                                           style={"visibility": "visible"}),

                            ),
                            multiple=False,
                        ),
                        dbc.Button("Compute matchings", id="compute_matching_button", size="sm", outline=True,
                                   active=False, color="secondary"),
                        html.Span(id="upload_confirmation", style={"vertical-align": "middle"}),
                        html.Span(id="compute_confirmation", style={"vertical-align": "middle"})
                    ])
                ),
                dbc.Label("3. Choose selection mode"),
                dbc.Row(
                    [html.Div([
                        # html.Pre(id='placeholder', style={"display": "none"}) ,
                        # html.Pre(False, id='selection_mode', style={"display": "none"}) ,
                        dbc.Button("Selection mode", id="selection_mode_button", size="sm", outline=True, active=True,
                                   color="secondary"),
                        dbc.Button("Suggestion mode", id="suggestion_mode_button", size="sm", outline=True,
                                   active=False, color="secondary"),
                    ])
                    ]
                )
            ])







# bottom right controls (selection mode, display mode, no match, save buttons) for 16 sessions

select_controls_16sessions =     [

        dbc.FormGroup(
            [
                dbc.Label("4. Display mode"),

                dbc.Row(
                    [html.Div([
                        # html.Pre(id='placeholder', style={"display": "none"}) ,
                        # html.Pre(False, id='selection_mode', style={"display": "none"}) ,
                        dbc.Button("Basic image", id="basic_mode_button", size="sm", outline=True, active=True,
                                   color="secondary"),
                        dbc.Button("Mean intensity image", id="tiff_mode_button", size="sm", outline=True, active=False,
                                   color="secondary"),
                    ])
                    ]
                ),
            ]),
        dbc.FormGroup(
            [
                dbc.Label("5. Indicate if neuron does not exsit"),
                dbc.Row(
                    [html.Div([
                        dbc.Button("No match 01 ", id="no_match_1", size="sm", style={"visibility": "hidden"},
                                   outline=True, color="secondary"),
                        dbc.Button("No match 02 ", id="no_match_2", size="sm", outline=True, color="secondary",
                                   active=False),
                        dbc.Button("No match 03 ", id="no_match_3", size="sm", outline=True, color="secondary",
                                   active=False),
                        dbc.Button("No match 04 ", id="no_match_4", size="sm", outline=True, color="secondary",
                                   active=False),
                    ])
                    ]
                ),
                dbc.Row(
                    [html.Div([
                        dbc.Button("No match 05", id="no_match_5", size="sm", outline=True, color="secondary",
                                   active=False),
                        dbc.Button("No match 06", id="no_match_6", size="sm", outline=True, color="secondary",
                                   active=False),
                        dbc.Button("No match 07", id="no_match_7", size="sm", outline=True, color="secondary",
                                   active=False),
                        dbc.Button("No match 08", id="no_match_8", size="sm", outline=True, color="secondary",
                                   active=False),
                    ])
                    ]
                ),
                dbc.Row(
                    [html.Div([
                        dbc.Button("No match 09", id="no_match_9", size="sm", outline=True, color="secondary",
                                   active=False),
                        dbc.Button("No match 10", id="no_match_10", size="sm", outline=True, color="secondary",
                                   active=False),
                        dbc.Button("No match 11", id="no_match_11", size="sm", outline=True, color="secondary",
                                   active=False),
                        dbc.Button("No match 12", id="no_match_12", size="sm", outline=True, color="secondary",
                                   active=False),
                    ])
                    ]
                ),
                dbc.Row(
                    [html.Div([
                        dbc.Button("No match 13", id="no_match_13", size="sm", outline=True, color="secondary",
                                   active=False),
                        dbc.Button("No match 14", id="no_match_14", size="sm", outline=True, color="secondary",
                                   active=False),
                        dbc.Button("No match 15", id="no_match_15", size="sm", outline=True, color="secondary",
                                   active=False),
                        dbc.Button("No match 16", id="no_match_16", size="sm", outline=True, color="secondary",
                                   active=False),
                    ])
                    ]
                ),
            ]),
        dbc.FormGroup([
            dbc.Label("6. Click save after each matched neuron"),
            dbc.Row(
                [
                    html.Div([
                        dbc.Button("Save", id="save_button", block=True, size="sm", outline=True, active=True,
                                   color="primary"),
                        html.Pre("Number of neurons to be matched", id='num_left_to_match')
                    ])
                ]
            )
        ])
    ]




controls_12sessions = dbc.FormGroup(
            [
                dbc.Label("1. Upload npy files of the sessions"),
                # dbc.Spinner(html.Div(id="loading-output")),

                dbc.Row([
                    dcc.Upload(
                        id="upload_session_0",
                        children=html.Div(
                            dbc.Button(' Upload 01', size="sm", outline=True, color="secondary"),
                            # dbc.Button([dbc.Spinner(size="sm"), "Loading..."],'Upload 1', size="sm",  outline=True, color="secondary"),
                        ),
                        multiple=False,
                    ),
                    dcc.Upload(
                        id="upload_session_1",
                        children=html.Div(
                            dbc.Button("Upload " + "02", size="sm", outline=True, color="secondary"),
                        ),
                        multiple=False,
                    ),
                    dcc.Upload(
                        id="upload_session_2",
                        children=html.Div(
                            dbc.Button('Upload 03', size="sm", outline=True, color="secondary"),
                        ),
                        multiple=False,
                    ),
                    dcc.Upload(
                        id="upload_session_3",
                        children=html.Div(
                            dbc.Button('Upload 04', size="sm", outline=True, color="secondary"),
                        ),
                        multiple=False,
                    ),
                ]
                ),
                dbc.Row([
                        dcc.Upload(
                            id="upload_session_4",
                            children=html.Div(
                                dbc.Button('Upload 05', size="sm", outline=True, color="secondary"),
                            ),
                            multiple=False,
                        ),
                        dcc.Upload(
                            id="upload_session_5",
                            children=html.Div(
                                dbc.Button('Upload 06', size="sm", outline=True, color="secondary"),
                            ),
                            multiple=False,
                        ),
                        dcc.Upload(
                            id="upload_session_6",
                            children=html.Div(
                                dbc.Button('Upload 07', size="sm", outline=True, color="secondary"),
                            ),
                            multiple=False,
                        ),
                        dcc.Upload(
                            id="upload_session_7",
                            children=html.Div(
                                dbc.Button('Upload 08', size="sm", outline=True, color="secondary"),
                            ),
                            multiple=False,
                        ),
                    ]
                ),
                dbc.Row([
                        dcc.Upload(
                            id="upload_session_8",
                            children=html.Div(
                                dbc.Button('Upload 09', size="sm", outline=True, color="secondary"),
                            ),
                            multiple=False,
                        ),
                        dcc.Upload(
                            id="upload_session_9",
                            children=html.Div(
                                dbc.Button('Upload 10', size="sm", outline=True, color="secondary"),
                            ),
                            multiple=False,
                        ),
                        dcc.Upload(
                            id="upload_session_10",
                            children=html.Div(
                                dbc.Button('Upload 11', size="sm", outline=True, color="secondary"),
                            ),
                            multiple=False,
                        ),
                        dcc.Upload(
                            id="upload_session_11",
                            children=html.Div(
                                dbc.Button('Upload 12', size="sm", outline=True, color="secondary"),
                            ),
                            multiple=False,
                        ),
                    ]
                ),
                dbc.Label("2. Upload/Create neuron matching file"),
                dbc.Row(
                    html.Div([
                        dcc.Upload(
                            id="upload_csv_matching", style={"visibility": "visible"},
                            children=html.Div(
                                dbc.Button('Upload csv', size="sm", outline=True, color="secondary",
                                           style={"visibility": "visible"}),

                            ),
                            multiple=False,
                        ),
                        dbc.Button("Compute matchings", id="compute_matching_button", size="sm", outline=True,
                                   active=False, color="secondary"),
                        html.Span(id="upload_confirmation", style={"vertical-align": "middle"}),
                        html.Span(id="compute_confirmation", style={"vertical-align": "middle"})
                    ])
                ),
                dbc.Label("3. Choose selection mode"),
                dbc.Row(
                    [html.Div([
                        # html.Pre(id='placeholder', style={"display": "none"}) ,
                        # html.Pre(False, id='selection_mode', style={"display": "none"}) ,
                        dbc.Button("Selection mode", id="selection_mode_button", size="sm", outline=True, active=True,
                                   color="secondary"),
                        dbc.Button("Suggestion mode", id="suggestion_mode_button", size="sm", outline=True,
                                   active=False, color="secondary"),
                    ])
                    ]
                )
            ])







# bottom right controls (selection mode, display mode, no match, save buttons) for 16 sessions

select_controls_12sessions =     [

        dbc.FormGroup(
            [
                dbc.Label("4. Display mode"),

                dbc.Row(
                    [html.Div([
                        # html.Pre(id='placeholder', style={"display": "none"}) ,
                        # html.Pre(False, id='selection_mode', style={"display": "none"}) ,
                        dbc.Button("Basic image", id="basic_mode_button", size="sm", outline=True, active=True,
                                   color="secondary"),
                        dbc.Button("Mean intensity image", id="tiff_mode_button", size="sm", outline=True, active=False,
                                   color="secondary"),
                    ])
                    ]
                ),
            ]),
        dbc.FormGroup(
            [
                dbc.Label("5. Indicate if neuron does not exsit"),
                dbc.Row(
                    [html.Div([
                        dbc.Button("No match 01 ", id="no_match_1", size="sm", style={"visibility": "hidden"},
                                   outline=True, color="secondary"),
                        dbc.Button("No match 02 ", id="no_match_2", size="sm", outline=True, color="secondary",
                                   active=False),
                        dbc.Button("No match 03 ", id="no_match_3", size="sm", outline=True, color="secondary",
                                   active=False),
                        dbc.Button("No match 04 ", id="no_match_4", size="sm", outline=True, color="secondary",
                                   active=False),
                    ])
                    ]
                ),
                dbc.Row(
                    [html.Div([
                        dbc.Button("No match 05", id="no_match_5", size="sm", outline=True, color="secondary",
                                   active=False),
                        dbc.Button("No match 06", id="no_match_6", size="sm", outline=True, color="secondary",
                                   active=False),
                        dbc.Button("No match 07", id="no_match_7", size="sm", outline=True, color="secondary",
                                   active=False),
                        dbc.Button("No match 08", id="no_match_8", size="sm", outline=True, color="secondary",
                                   active=False),
                    ])
                    ]
                ),
                dbc.Row(
                    [html.Div([
                        dbc.Button("No match 09", id="no_match_9", size="sm", outline=True, color="secondary",
                                   active=False),
                        dbc.Button("No match 10", id="no_match_10", size="sm", outline=True, color="secondary",
                                   active=False),
                        dbc.Button("No match 11", id="no_match_11", size="sm", outline=True, color="secondary",
                                   active=False),
                        dbc.Button("No match 12", id="no_match_12", size="sm", outline=True, color="secondary",
                                   active=False),
                    ])
                    ]
                ),
            ]),
        dbc.FormGroup([
            dbc.Label("6. Click save after each matched neuron"),
            dbc.Row(
                [
                    html.Div([
                        dbc.Button("Save", id="save_button", block=True, size="sm", outline=True, active=True,
                                   color="primary"),
                        html.Pre("Number of neurons to be matched", id='num_left_to_match')
                    ])
                ]
            )
        ])
    ]