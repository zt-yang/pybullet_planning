from os.path import join, abspath

content_dir = join('content')
media_dir = join('media')
planning_tree_path = join(media_dir, 'planning_tree.png')
log_name = 'txt_file.txt'

chat_prompt = """
                <div class="message user1">
                    <p class="username">Prompt:</p>
                    <p>{text}</p>
                </div>
"""

chat_answer = """
                <div class="message user2">
                    <p class="username">Answer:</p>
                    <p>{text}</p>
                </div>
"""

chat_processed = """
                <div class="message user3">
                    <p class="username">Post-processed:</p>
                    <p>{text}</p>
                </div>
"""

chat_templates = {'prompt': chat_prompt, 'answer': chat_answer, 'processed': chat_processed}

row_template = """
        <tr>
            <td class="left-column">{left}
            </td>
            <td class="right-column">{right}
            </td>
        </tr>
"""

image_row = """
                <img src="{img_path}"/><br>
"""

toggled_label = """
                <button onclick="loadAndToggleContent('{toggle_id}', '{txt_path}', '{img_path}')" class="{style}">
                {label}</button>&nbsp;
"""

toggled_hidden = """
                <br><span id="{span_id}" class="hidden-text"></span><br>
"""

toggled_txt_text = """
                <button onclick="loadAndToggleText('{toggle_id}', '{txt_path}')"">{label}</button><br>
                <span id="{span_id}" class="hidden-text"></span><br>
"""

toggled_text = """
                <button onclick="toggleText({toggle_id})">{label}</button><br>
                <span id="{span_id}" class="hidden-text">{hidden}</span><br>
"""

merged_row_template = """
        <tr>
            <td colspan="2" class="merged">Round {count}</td>
        </tr>
"""

html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VLM + Planning History</title>
    <!--link rel="stylesheet" href="log/style.css">
    <script src="log/scripts.js"></script--> 
    <style>
    {style}
    </style>
    <script>
    {script}
    </script>
</head>
<body>
    {notes}
    <table>
        {rows}
    </table>

</body>
</html>
"""

included_style = """
table {
    width: 100%;
    border-collapse: collapse;
}

th, td {
    border: 1px solid black;
    padding: 10px;
    vertical-align: top;
    text-align: left;
    /*width: 50%;*/
    overflow-x: auto; /* Add horizontal scroll for overflow */
    box-sizing: border-box; /* Ensures padding is included in the width */
}

.left-column {
    width: 33%;
    font-size: 12px;
}

.right-column {
    width: 67%;
    font-size: 12px;
}

.hidden-text {
    display: none;
    white-space: pre-wrap; /* Wraps the text and preserves white spaces and formatting */
    overflow-x: auto; /* Enable horizontal scrolling for overflow */
    font-size: 12px;
}

button {
    font-size: 14px !important;
    margin: 4px 0px;
    display: inline-block;
}

.button-started {

}

.button-solved {
    color: #27ae60 !important;
}

.button-failed {
    color: #c0392b !important;
}

.button-already {
    color: #3498db !important;
}

.button-ungrounded {
    color: #f1c40f !important;
}

.message {
    border-radius: 5px;
    padding: 10px;
    margin-bottom: 10px;
}

.user1 {
    background-color: #dbedf9;
    margin-right: 15%;
    color: #333;
}

.user2 {
    background-color: #fae5d2;
    margin-left: 15%;
    color: #333;
}

.user3 {
    background-color: #dbf7e7;
    margin-left: 10%;
    margin-right: 10%;
    color: #333;
}

.merged {
    background-color: #dde0e2;
}

.username {
    font-weight: bold;
}

img {
    max-width: 100%;
}

.full_column {
    width: 100% !important;
}
"""

included_scripts = """
function toggleText(number) {
    var text = document.getElementById("text-" + number);
    text.style.display = (text.style.display === "none" || text.style.display === "") ? "inline" : "none";
}

function loadAndToggleText(number, filePath) {
    toggleImagw(number);

    var textElement = document.getElementById("text-" + number);

    // Fetch the content from the file
    fetch(filePath)
        .then(response => response.text())
        .then(data => {
            textElement.innerText = data; // Use innerText to preserve formatting
            toggleText(number); // Use the existing toggle function
        })
        .catch(error => console.error('Error fetching file:', error));
}

function loadAndToggleContent(number, filePath, imgPath) {
    var elem = document.getElementById("content-" + number);

    // Clear previous content
    elem.innerHTML = '';

    // // Create and append the image element
    // var img = new Image();
    // img.src = planTreePath;
    // elem.appendChild(img);

    // Create and append the image element
    var img2 = new Image();
    img2.src = imgPath;
    elem.appendChild(img2);

    var br = document.createElement("br");
    elem.appendChild(br);

    // Fetch and display the text content
    fetch(filePath)
        .then(response => response.text())
        .then(data => {
            var textNode = document.createElement("span");
            textNode.innerText = data;
            elem.appendChild(textNode);
            toggleText(number); // Use the existing toggle function
        })
        .catch(error => console.error('Error fetching file:', error, filePath, planTreePath, imgPath));

    elem.style.display = (elem.style.display === "none" || elem.style.display === "") ? "inline" : "none";
}
"""


def launch_log_page(log_dir, port=8000):
    import http.server
    import socketserver
    import _thread as thread
    import warnings

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=log_dir, **kwargs)

    def start_server():
        with warnings.catch_warnings():
            with socketserver.TCPServer(("", port), Handler) as httpd:
                print(f"serving {abspath(log_dir)} at port", port)
                httpd.serve_forever()

    # start the server in a background thread
    thread.start_new_thread(start_server, ())


## ---------------------------------------------------------------------------------------------------

def _default_make_cell_text(d, row):
    color = 'black'
    return f'<span style="color: {color}">{d}</span>'


def _make_html_table_from_rows(rows, get_text_fn=_default_make_cell_text, table_style=''):
    header = rows[0]
    table = '\n'.join([f'<th style="background-color: #dde0e2;">{h}</th>' for h in header])
    table = f"<tr>{table}</tr>"
    for row in rows[1:]:
        cells = '\n'.join([f"<td>{get_text_fn(d, row)}</td>" for d in row])
        table += f'\n<tr>\n{cells}\n</tr>\n'
    return f"\n<table{table_style}>\n{table}\n</table>\n"


def _make_summary_table_from_rows(summary_table):
    def _make_cell_text(d, row):
        color = 'black'
        return f'<span style="color: {color}">{d}</span>'
    table_style = (' style="position: absolute; width: 600px; left: 800px; top: 10px; '
                   'font-size:10pt; background-color: white;z-index: 120;"')
    summary_table = [[]+row for row in summary_table]  ## so that all columns of the subgoal groups will be grey
    return _make_html_table_from_rows(summary_table, table_style=table_style)


def _make_progress_table_from_rows(progress_table, run_name):
    from vlm_tools.vlm_utils import STATUS_TO_COLOR

    def _make_cell_text(d, row):
        status = row[3]
        color = STATUS_TO_COLOR[status] if status in STATUS_TO_COLOR else 'black'
        return f'<span style="color: {color}">{d}</span>'

    table_style = ' style="font-size: 10pt; "'
    labels = ['Continuous Success', 'Task Progress', 'Planning Success']
    link_to_page = """<a href='{link}' target='_blank'>{text}</a>"""

    for i, row in enumerate(progress_table):
        if i == len(progress_table) - 1:
            progress_table[i][1: 4] = [labels[j] + '<br>' + progress_table[i][1+j] for j in range(len(labels))]
        elif i != 0:
            agent_state = progress_table[i][-1]
            if len(agent_state) > 0:
                if '/states/' in agent_state:
                    loaded_run_name, state_name = agent_state.split('/states/')
                    text = "/loaded_states/" + state_name
                    link = agent_state
                else:
                    text = "/states/" + agent_state
                    link = run_name + text
                progress_table[i][-1] = link_to_page.format(link=link, text=text)
    return _make_html_table_from_rows(progress_table, get_text_fn=_make_cell_text, table_style=table_style)


def output_html(log_rounds, output_path, memory_path=None, progress_table=None):

    rows = []
    for i, info in enumerate(log_rounds):
        chat_round, planner_round = info.values()

        ## the chat history is on the left
        left = ''
        for typ, text in chat_round:
            left += chat_templates[typ].format(text=text.replace('\\n', '<br>'))

        ## the planner logs are on the right
        right = image_row.format(img_path=planning_tree_path)

        ## sub-planning problems
        for j, (subgoal, txt_path, obs_path, status) in enumerate(planner_round):
            right += toggled_label.format(toggle_id=f'{i}', txt_path=txt_path,  ## plan_tree_path=plan_tree_path,
                                          img_path=obs_path, style=f'button-{status}', label=subgoal)
        span_id = f'content-{i}'
        right += toggled_hidden.format(span_id=span_id)

        rows += [merged_row_template.format(count=str(i+1)), row_template.format(left=left, right=right)]

    run_name = '/'.join(output_path.split('/')[-4:-2])
    memory_name = memory_path
    notes = f"<p>{run_name}</p><p>{memory_name}</p>"

    ## first row is header, last row is summary
    if progress_table is not None and len(progress_table) > 2:
        notes += '<br>' + _make_progress_table_from_rows(progress_table, run_name)

    with open(output_path, 'w') as f:
        f.write(html_template.format(style=included_style, script=included_scripts, notes=notes,
                                     rows=''.join(rows)).replace('\\n', '<br>'))
