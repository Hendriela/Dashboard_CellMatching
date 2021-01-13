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
