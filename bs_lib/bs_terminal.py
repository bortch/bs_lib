from rich import table
from rich import style
from rich.console import Console
from rich.theme import Theme
from rich.style import Style
from rich.text import Text
from rich.table import Column, Table

font_color = 'black'
bg_color = 'bright_white'

custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": Style(color="red", blink=True, bold=True),
    "title": Style(color=font_color, bgcolor=bg_color, bold=True,
                   encircle=True),
    "subtitle": Style(color=font_color, bgcolor=bg_color, italic=True),
    "rule": Style(color=font_color, bgcolor=bg_color),
    "list_index": Style(color=font_color),
    "column": Style(color=font_color, bgcolor=bg_color),
    "border": Style(color=font_color, bgcolor=bg_color),
    "header": Style(color=font_color, bgcolor=bg_color),
    "row": Style(color=font_color, bgcolor=bg_color),
    "table": Style(color=font_color, bgcolor=bg_color)
})


def article(title, content):
    """print a title and a content

    Args:
        title (str): Title of the article
        content (dict, list of dict): a content object or list of dict
    """    
    console = Console(theme=custom_theme)
    title_ = Text(title.upper())
    title_.stylize('title')
    console.rule(title=title_, style='rule', align='left')
    if isinstance(content, list):
        contents = parse_contents(content)
        for c in contents:
            console.print(c)
    else:
        content = parse_content(content=content)
        console.print(content)
    # console.print(title_)


def parse_content(content):
    result = None
    if 'table' in content:
        result = get_table(content=content)
    if 'list' in content:
        if 'ordered' in content['list']['type']:
            result = get_ordered_list(content=content['list'])
    if 'section' in content:
        result = get_section(content['section'])
    if 'dataframe' in content:
        result = get_dataframe(content['dataframe'])
    return result


def parse_contents(contents):
    results = []
    for content in contents:
        results.append(parse_content(content))
    return results


def get_ordered_list(content):
    for key, value in content.items():
        index = Text(key, style='list_index')
    #content = {"list":{'chi2':chi2,"pvalue":pvalue,"cramer v":cramer_v}}


def create_column(name, options={}):
    return {"name": name, 'options': options}


def create_table(title, columns, data, options=[]):
    columns_ = []
    for c in range(len(columns)):
        options_ = {}
        if len(options) > 0:
            options_ = options[c]
        columns_.append(create_column(columns[c], options_))
    table = {"table": {'title': title,
                       'columns': columns_,
                       'data': data
                       }}
    return table


def get_table(content):
    # {"table":{
    #           "title":'The Table Title',
    #           "columns":[{
    #               "name":"Column_1_name",
    #               "options":{"justify":"center"}
    #               }],
    #           "data":[[value_1,value_2,value_3]]
    #           }
    # }
    content = content['table']
    table = Table(title=content['title'],
                  style='table', highlight=True, title_style='subtitle', title_justify='left', header_style='header'
                  # border_style='border', header_style='header', row_styles='row'
                  )

    # create columns
    default_options = {"justify": "center", "no_wrap": True}
    columns = content['columns']
    for column in columns:
        # merge column options: column['options'] replace default_option
        options = {**default_options, **column['options']}
        table.add_column(column['name'], **options)

    data = content['data']
    for row in data:
        row_ = []
        for value in row:
            if isinstance(value, float):
                value = f'{value:.4f}'
            row_.append(str(value))
        table.add_row(*row_)
    return table


def create_section(title='title', content='content'):
    return {"section": {"title": title, "content": str(content)}}


def get_section(section_):
    result = Text()
    result.append(section_['title']+'\n', style='subtitle')
    result.append(section_['content'], style='content')
    return result


def create_dataframe(title, content):
    return {"dataframe": {"title": title, "dataframe": content}}


def get_dataframe(content):
    df = content['dataframe']
    # {‘index’ -> [index], ‘columns’ -> [columns], ‘data’ -> [values]}
    dictionnary = df.to_dict('split')

    # prepend with empty column for index
    dictionnary['columns'].insert(0, '')

    # format columns as expected
    formatted_columns = []
    for column in dictionnary['columns']:
        formatted_columns.append(create_column(column))

    # create rows
    data = dictionnary['data']
    for row_index in range(len(data)):
        data[row_index].insert(
            0, dictionnary['index'][row_index])

    table = create_table(
        title=content['title'], columns=dictionnary['columns'], data=data, options=[])
    return get_table(table)
